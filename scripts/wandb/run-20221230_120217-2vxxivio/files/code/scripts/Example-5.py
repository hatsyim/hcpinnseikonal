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
    xmin, ymin, zmin = 0, 0, 0
    xmax, ymax, zmax = dict_args['max_offset'], dict_args['max_offset'], dict_args['max_depth']

    data.id_sou_z = np.array(dict_args['zid.id_source'])
    data.id_rec_z = np.array(dict_args['zid.id_receiver'])
    data.id_sou_x = np.arange(0,len(X[0,:,0]),dict_args['sou_spacing'])
    data.id_rec_x = np.arange(0,len(X[0,:,0]),dict_args['rec_spacing'])

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

    # ZX plane after
    plot_section(vel3d[:,10,:], 'v_true_zx.png', vmin=np.nanmin(data.velmodel)+0.1, 
                 vmax=np.nanmax(data.velmodel)-0.5, save_dir=wandb_dir, aspect='equal',
                 xmin=xmin, xmax=xmax, zmin=zmin, zmax=zmax, 
                 sx=X[:,:,:,0].reshape(-1)[data.id_sou],sz=Z[:,:,:,0].reshape(-1)[data.id_sou],rx=X[:,:,:,0].reshape(-1)[data.id_rec],rz=Z[:,:,:,0].reshape(-1)[data.id_rec])

    # XY plane
    plot_section(vel3d[5,:,:], 'v_true_xy.png', vmin=np.nanmin(data.velmodel)+0.1, 
                 vmax=np.nanmax(data.velmodel)-0.5, save_dir=wandb_dir, aspect='equal',
                 xmin=xmin, xmax=xmax, zmin=xmin, zmax=xmax, 
                 sx=X[:,:,:,0].reshape(-1)[data.id_sou],sz=Y[:,:,:,0].reshape(-1)[data.id_sou],rx=X[:,:,:,0].reshape(-1)[data.id_rec],rz=Y[:,:,:,0].reshape(-1)[data.id_rec])

    # ZY plane
    plot_section(vel3d[:,:,10], 'v_true_zy.png', vmin=np.nanmin(data.velmodel)+0.1, 
                 vmax=np.nanmax(data.velmodel)-0.5, save_dir=wandb_dir, aspect='equal',
                 xmin=xmin, xmax=xmax, zmin=zmin, zmax=zmax, 
                 sx=Y[:,:,:,0].reshape(-1)[data.id_sou],sz=Z[:,:,:,0].reshape(-1)[data.id_sou],rx=Y[:,:,:,0].reshape(-1)[data.id_rec],rz=Z[:,:,:,0].reshape(-1)[data.id_rec])
    
    v_init = model.evaluate_velocity(init_loader,batch_size=BATCH_SIZE,num_pts=X.size)

    # ZX plane after
    plot_section(v_init.reshape(X.shape)[:,10,:,0], 'v_init_zx.png', vmin=np.nanmin(data.velmodel)+0.1, 
                 vmax=np.nanmax(data.velmodel)-0.5, save_dir=wandb_dir, aspect='equal',
                 xmin=xmin, xmax=xmax, zmin=zmin, zmax=zmax, 
                 sx=X[:,:,:,0].reshape(-1)[data.id_sou],sz=Z[:,:,:,0].reshape(-1)[data.id_sou],rx=X[:,:,:,0].reshape(-1)[data.id_rec],rz=Z[:,:,:,0].reshape(-1)[data.id_rec])

    # XY plane
    plot_section(v_init.reshape(X.shape)[5,:,:,0], 'v_init_xy.png', vmin=np.nanmin(data.velmodel)+0.1, 
                 vmax=np.nanmax(data.velmodel)-0.5, save_dir=wandb_dir, aspect='equal',
                 xmin=xmin, xmax=xmax, zmin=xmin, zmax=xmax, 
                 sx=X[:,:,:,0].reshape(-1)[data.id_sou],sz=Y[:,:,:,0].reshape(-1)[data.id_sou],rx=X[:,:,:,0].reshape(-1)[data.id_rec],rz=Y[:,:,:,0].reshape(-1)[data.id_rec])

    # ZY plane
    plot_section(v_init.reshape(X.shape)[:,:,10,0], 'v_init_zy.png', vmin=np.nanmin(data.velmodel)+0.1, 
                 vmax=np.nanmax(data.velmodel)-0.5, save_dir=wandb_dir, aspect='equal',
                 xmin=xmin, xmax=xmax, zmin=zmin, zmax=zmax, 
                 sx=Y[:,:,:,0].reshape(-1)[data.id_sou],sz=Z[:,:,:,0].reshape(-1)[data.id_sou],rx=Y[:,:,:,0].reshape(-1)[data.id_rec],rz=Z[:,:,:,0].reshape(-1)[data.id_rec])
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

    # ZX plane after
    plot_section(v_pred.reshape(X.shape)[:,0,:,i], 'v_pred_zx.png', vmin=np.nanmin(data.velmodel)+0.1, 
                 vmax=np.nanmax(data.velmodel)-0.5, save_dir=wandb_dir, aspect='equal',
                 xmin=xmin, xmax=xmax, zmin=zmin, zmax=zmax, 
                 sx=X[:,:,:,i].reshape(-1)[data.id_sou],sz=Z[:,:,:,i].reshape(-1)[data.id_sou],rx=X[:,:,:,i].reshape(-1)[data.id_rec],rz=Z[:,:,:,i].reshape(-1)[data.id_rec])

    # XY plane
    plot_section(v_pred.reshape(X.shape)[args.zid.id_source,:,:,i], 'v_pred_xy.png', vmin=np.nanmin(data.velmodel)+0.1, 
                 vmax=np.nanmax(data.velmodel)-0.5, save_dir=wandb_dir, aspect='equal',
                 xmin=xmin, xmax=xmax, zmin=xmin, zmax=xmax, 
                 sx=X[:,:,:,i].reshape(-1)[data.id_sou],sz=Y[:,:,:,i].reshape(-1)[data.id_sou],rx=X[:,:,:,i].reshape(-1)[data.id_rec],rz=Y[:,:,:,i].reshape(-1)[data.id_rec])

    # ZY plane
    plot_section(v_pred.reshape(X.shape)[:,:,0,i], 'v_pred_zy.png', vmin=np.nanmin(data.velmodel)+0.1, 
                 vmax=np.nanmax(data.velmodel)-0.5, save_dir=wandb_dir, aspect='equal',
                 xmin=xmin, xmax=xmax, zmin=zmin, zmax=zmax, 
                 sx=Y[:,:,:,i].reshape(-1)[data.id_sou],sz=Z[:,:,:,i].reshape(-1)[data.id_sou],rx=Y[:,:,:,i].reshape(-1)[data.id_rec],rz=Z[:,:,:,i].reshape(-1)[data.id_rec])

    # Save model
    torch.save({
            'tau_model_state_dict': tau_model.state_dict(),
            'v_model_state_dict': v_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_history
    }, wandb_dir+'/saved_model')
    
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