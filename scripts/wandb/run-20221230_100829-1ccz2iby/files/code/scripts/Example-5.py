import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from argparse import ArgumentParser   
from scipy import interpolate

from hcpinnseikonal.utils import *
from hcpinnseikonal.model import *
from hcpinnseikonal.plot import *
from hcpinnseikonal.arguments import *
from hcpinnseikonal.distributed import *

if __name__ == "__main__":

    args = parser.parse_args()

    dict_args = vars(args)
    print(dict_args)

    # Change these lines for the wandb setup
    if args.use_wandb=='y':
        wandb.init(project=args.project_name)
        wandb.run.log_code(".")
        wandb_dir = wandb.run.dir
    else:
        args.save_folder='../saves/saves_lightning3d'
        from pathlib import Path
        Path(args.save_folder).mkdir(parents=True, exist_ok=True)
        wandb_dir = args.save_folder
        
    # Setup
    pl.seed_everything(dict_args['seed'])

    model = HCEikonalPINNsModel(dict_args)
    nx, nz, ns = model.x.shape[0], model.z.shape[0], model.sx.shape[0]    
    data = HCEikonalPINNsData(dict_args, batch_size=int(nx*nz*ns)//200)
    X, Y, Z, SX, SY, SZ, taud, taudx, taudy, T0, px0, py0, pz0, index = data.input_list

    # Setup
    pl.seed_everything(dict_args['seed'])

    model = HCEikonalPINNsModel(dict_args)
    nx, nz, ns = model.x.shape[0], model.z.shape[0], model.sx.shape[0]    
    data = HCEikonalPINNsData(dict_args, batch_size=int(nx*nz*ns)//200)
    X, Y, Z, SX, SY, SZ, taud, taudx, taudy, T0, px0, py0, pz0, index = data.input_list

    id_sou_z = np.array(dict_args['zid_source'])
    id_rec_z = np.array(dict_args['zid_receiver'])
    id_sou_x = np.arange(0,len(X[0,:,0]),dict_args['sou_spacing'])
    id_rec_x = np.arange(0,len(X[0,:,0]),dict_args['rec_spacing'])

    BATCH_SIZE = Z.size//200 if torch.cuda.is_available() else 64
    NUM_WORKERS = int(os.cpu_count() / 2)

    if dict_args['fast_loader']=='y':
        data_loader = FastTensorDataLoader(
            torch.from_numpy(np.vstack((data.input_list[0], data.input_list[1], data.input_list[2])).T).ravel().float(),
            torch.from_numpy(data.input_list[3]).ravel().float(),
            torch.from_numpy(data.input_list[4]).ravel().float(),
            torch.from_numpy(data.input_list[5]).ravel().float(),
            torch.from_numpy(data.input_list[6]).ravel().float(),
            torch.from_numpy(data.input_list[7]).ravel().float(),
            torch.from_numpy(data.input_list[8]).ravel().float(),
            torch.from_numpy(data.input_list[9]).ravel().float(),
            torch.from_numpy(data.input_list[10]).ravel().float(),
            torch.from_numpy(data.input_list[11]).ravel().float(),
            torch.from_numpy(data.input_list[12]).ravel().float(),
            batch_size=BATCH_SIZE, 
            shuffle=True
    )
    else:
        data_loader = torch.utils.data.DataLoader(
            data.input_dataset,
            batch_size=BATCH_SIZE,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True,
            num_workers=NUM_WORKERS
    )

    init_loader = FastTensorDataLoader(
        torch.from_numpy(np.vstack((data.input_list[0].ravel(), data.input_list[1].ravel(), data.input_list[2].ravel())).T).float(),
        torch.from_numpy(data.input_list[3].ravel()).float(),
        torch.from_numpy(data.input_list[4].ravel()).float(),
        torch.from_numpy(data.input_list[5].ravel()).float(),
        torch.from_numpy(data.input_list[6].ravel()).float(),
        torch.from_numpy(data.input_list[7].ravel()).float(),
        torch.from_numpy(data.input_list[8].ravel()).float(),
        torch.from_numpy(data.input_list[9].ravel()).float(),
        torch.from_numpy(data.input_list[10].ravel()).float(),
        torch.from_numpy(data.input_list[11].ravel()).float(),
        torch.from_numpy(data.input_list[12].ravel()).float(),
        batch_size=BATCH_SIZE, 
        shuffle=True
    )

    v_init = model.evaluate_velocity(init_loader,batch_size=BATCH_SIZE,num_pts=X.size)
    tau_init = model.evaluate_tau(init_loader,batch_size=BATCH_SIZE,num_pts=X.size)

    v_init = v_init.detach()
    v_init = v_init.reshape(nz,nx,-1)
    tau_init = tau_init.detach()
    tau_init = tau_init.reshape(nz,nx,-1)

    # Training
    wandb_logger = WandbLogger(log_model="all")

    if dict_args['mixed_precision']=='y':
        trainer = Trainer(
            accelerator="gpu",
            strategy="ddp_find_unused_parameters_false",
            devices="auto",  # limiting got iPython runs
            max_epochs=dict_args['num_epochs'],
            precision=16,
            callbacks=[
                TQDMProgressBar(refresh_rate=20), 
                ModelCheckpoint(monitor="train_pde_loss", mode="min")],
            logger=wandb_logger
        )

    else:
        trainer = Trainer(
            accelerator="gpu",
            strategy="ddp_find_unused_parameters_false",
            devices="auto",  # limiting got iPython runs
            max_epochs=dict_args['num_epochs'],
            callbacks=[
                TQDMProgressBar(refresh_rate=20), 
                ModelCheckpoint(monitor="train_pde_loss", mode="min")],
            logger=wandb_logger
        )
    trainer.fit(model, datamodule=data)

    # Inference
    v_pred = model.evaluate_velocity(init_loader,batch_size=BATCH_SIZE,num_pts=X.size)
    tau_pred = model.evaluate_tau(init_loader,batch_size=BATCH_SIZE,num_pts=X.size)

    tau_pred = tau_pred.detach()
    tau_pred = tau_pred.reshape(nz,nx,ns)

    T_pred = (torch.tensor(taud).reshape(nz,nx,ns) + torch.tensor(Z.reshape(nz,nx,ns))*tau_pred)*torch.tensor(T0).reshape(nz,nx,ns)

    v_pred = v_pred.detach()
    v_pred = v_pred.reshape(nz,nx,ns)[:,:,0]
    v_init = v_init.detach()
    v_init = v_init.reshape(nz,nx,ns)[:,:,0]
    v_true = data.velmodel.reshape(Z.shape)[::1,:,0]

    if args['rescale_plot']=='y':
        earth_radi = dict_args['plotting_factor'] # Average in km
        xmin, xmax, deltax = earth_radi*model.x.min(), earth_radi*model.x.max(), earth_radi*dict_args['lateral_spacing']
        zmin, zmax, deltaz = earth_radi*model.z.min(), earth_radi*model.z.max(), earth_radi*dict_args['vertical_spacing']

        # Creating grid, extending the velocity model, and prepare list of grid points for training (X_star)
        z = np.arange(zmin,zmax+deltaz,deltaz)
        x = np.arange(xmin,xmax+deltax,deltax)

        # Point-source locations
        sx = x[id_sou_x]
        sz = z[id_sou_z]*np.ones_like(sx)

        Z,X,SX = np.meshgrid(z,x,sx,indexing='ij')

        SZ = np.ones(SX.shape)*sz # Creating an array of sources along z with same size as SX

        T_pred, T_data, T0 = T_pred*dict_args['plotting_factor'], T_data*dict_args['plotting_factor'], T0*dict_args['plotting_factor']

    # Save model
    torch.save({
            'tau_model_state_dict': model.tau_model.state_dict(),
            'v_model_state_dict': model.v_model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict()
    }, wandb.run.dir + '/checkpoint')

    # # To load
    # checkpoint = torch.load( wandb.run.dir + '/checkpoint')
    # tau_model.load_state_dict(checkpoint['tau_model_state_dict'])
    # v_model.load_state_dict(checkpoint['v_model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])