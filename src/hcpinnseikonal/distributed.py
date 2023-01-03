import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io 
import time
import random
import os
import pytorch_lightning as pl
import torch

from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load style @hatsyim
# plt.style.use("~/science.mplstyle")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
plt.rcParams['figure.figsize'] =  [6.4, 4.8]

from hcpinnseikonal.train3d import *
from hcpinnseikonal.model import *
from hcpinnseikonal.utils import *
from hcpinnseikonal.plot import *

def setup_medium(args):  
    
    import numpy as np
    from scipy import interpolate
    import pykonal
    
    # Medium
    data_type = args['data_type']
    deltar = args['rec_spacing']
    deltas = args['sou_spacing']
    
    # Computational model parameters
    zmin = -0.1 if args['field_synthetic']=='y' else 0; zmax = args['max_depth'] #; deltaz = args['vertical_spacing'];
    ymin = 0.; ymax = args['max_offset'] #; deltay = args['lateral_spacing'];
    xmin = 0.; xmax = args['max_offset'] #; deltax = args['lateral_spacing'];
    deltax, deltay, deltaz = args['sampling_rate']*0.00625/2*4, args['sampling_rate']*0.00625/2*4, args['sampling_rate']*0.00625/2

    if args['earth_scale']=='y':
        earth_radi = 6371/args['scale_factor'] # Average in km
        xmin, xmax, deltax = earth_radi*xmin, earth_radi*xmax, earth_radi*deltax
        ymin, ymax, deltay = earth_radi*ymin, earth_radi*ymax, earth_radi*deltay
        zmin, zmax, deltaz = earth_radi*zmin, earth_radi*zmax, earth_radi*deltaz

    # Creating grid, extending the velocity model, and prepare list of grid points for training (X_star)
    z = np.arange(zmin,zmax,deltaz)
    nz = z.size

    y = np.arange(ymin,ymax,deltay)
    ny = y.size

    x = np.arange(xmin,xmax,deltax)
    nx = x.size

    Z,Y,X = np.meshgrid(z,y,x,indexing='ij')

    # Number of training points
    num_tr_pts = 4000

    if args['field_synthetic']=='y':
        import pandas as pd
        import pygmt
        import numpy as np

        import pandas as pd

        # Earthquake events location
        location = pd.read_csv('/home/taufikmh/KAUST/fall_2022/GFATT_PINNs/data/fang_etal_2020/sjfzcatlog.csv')

        # Recorded traveltime data
        traveltime = pd.read_table('/home/taufikmh/KAUST/fall_2022/GFATT_PINNs/data/fang_etal_2020/sjfz_traveltime.dat', delim_whitespace='y')

        # Rounding to make the coordinates rounding the same
        location, traveltime = location.round(3), traveltime.round(3)

        # Merge
        data = pd.merge(traveltime, location,  how='left', left_on=['evlat','evlon','evdep'], right_on = ['evlat','evlon','evdep'])

        # Create earthquake group
        data['event_id'] = data.groupby(['evlat', 'evlon', 'evdep']).cumcount() + 1
        data['station_id'] = data.groupby(['stlat', 'stlon', 'stele']).cumcount() + 1

        # Station only
        sta_only = data.drop_duplicates(subset=['stlat', 'stlon'], keep='last')

        # Event only
        eve_only = data.drop_duplicates(subset=['evlat', 'evlon'], keep='last')

        region = [-118, -115, 32.5, 34.50]
        x0,x1,y0,y1 = -117.45, -115.55, 34.15, 32.76

        # eve_only['dist_to_line'] = 
        p1=np.array([(360+x0)*np.ones_like(eve_only.event_id.values), y0*np.ones_like(eve_only.event_id.values)])
        p2=np.array([(360+x1)*np.ones_like(eve_only.event_id.values), y1*np.ones_like(eve_only.event_id.values)])
        p3=np.array([eve_only.evlon, eve_only.evlat])

        d = pd.DataFrame(np.cross((p2-p1).T,(p3-p1).T)/np.linalg.norm((p2-p1).T))
        eve_only.loc[:, 'closest_event'] = np.copy(d[0].values)

        # sta_only['dist_to_line'] = 
        p1=np.array([(360+x0)*np.ones_like(sta_only.station_id.values), y0*np.ones_like(sta_only.station_id.values)])
        p2=np.array([(360+x1)*np.ones_like(sta_only.station_id.values), y1*np.ones_like(sta_only.station_id.values)])
        p3=np.array([sta_only.stlon, sta_only.stlat])

        d = pd.DataFrame(np.cross((p2-p1).T,(p3-p1).T)/np.linalg.norm((p2-p1).T))
        sta_only.loc[:, 'closest_station'] = np.copy(d[0].values)

        closest_sta = sta_only[np.abs(sta_only['closest_station'])<0.003]
        closest_eve = eve_only[np.abs(eve_only['closest_event'])<0.00003]

        grid = pygmt.datasets.load_earth_relief(resolution="03m", region=region)

        points = pd.DataFrame(
            data=np.linspace(start=(x0, y0), stop=(x1, y1), num=len(x)),
            columns=["x", "y"],
        )

        track = pygmt.grdtrack(points=points, grid=grid, newcolname="elevation")
        xtop = track.x.values + 360
        ztop = track.elevation.values*1e-3

        xsta = closest_sta.stlon.values
        zsta = closest_sta.stele.values

        xeve = closest_eve.evlon.values
        zeve = closest_eve.evdep.values

        xtop,xsta,xeve = xtop-xtop.min(),xsta-xsta.min(),xeve-xeve.min()
        xtop,xsta,xeve = xtop/xtop.max()*xmax,xsta/xsta.max()*xmax,xeve/xeve.max()*xmax

        ytop,ysta,yeve = ytop-ytop.min(),ysta-ysta.min(),yeve-yeve.min()
        ytop,ysta,yeve = ytop/ytop.max()*ymax,ysta/ysta.max()*ymax,yeve/yeve.max()*ymax

        ztop,zsta,zeve = ztop-ztop.min(),zsta-zsta.min(),zeve-zeve.min()
        ztop,zsta,zeve = args['station_factor']*ztop/ztop.max()+zmin,args['station_factor']*zsta/zsta.max()+zmin,zmax-args['event_factor']*zeve/zeve.max()

        xsta,xeve = xsta[(xsta>xtop.min()) & (xsta<xtop.max())], xeve[(xeve>xtop.min()) & (xeve<xtop.max())]
        ysta,yeve = ysta[(ysta>ytop.min()) & (ysta<ytop.max())], yeve[(yeve>ytop.min()) & (yeve<ytop.max())]
        zsta,zeve = zsta[(xsta>xtop.min()) & (xsta<xtop.max())],zeve[(xeve>xtop.min()) & (xeve<xtop.max())]

        if args['exclude_topo']=='y':
            ztop, zsta = zmin*np.ones_like(ztop), zmin*np.ones_like(zsta)

        ztop, zsta = zmin-ztop, zmin-zsta

        id_sou_z = np.array([]).astype(int)

        for szi in zeve.round(2):
            sid = np.where(np.abs(z.round(3)-szi)<1e-6)
            id_sou_z = np.append(id_sou_z,sid)

        id_rec_z = np.array([]).astype(int)

        for rzi in zsta.round(2):
            sid = np.where(np.abs(z.round(3)-rzi)<1e-6)
            id_rec_z = np.append(id_rec_z,sid)

        id_sou_y = np.array([]).astype(int)

        for syi in yeve.round(2):
            sid = np.where(np.abs(y.round(3)-syi)<1.5e-2)
            id_sou_y = np.append(id_sou_y,sid)

        id_rec_y = np.array([]).astype(int)

        for ryi in ysta.round(2):
            sid = np.where(np.abs(y.round(3)-ryi)<1.5e-2)
            id_rec_y = np.append(id_rec_y,sid)

        id_sou_x = np.array([]).astype(int)

        for sxi in xeve.round(2):
            sid = np.where(np.abs(x.round(3)-sxi)<1.5e-2)
            id_sou_x = np.append(id_sou_x,sid)

        id_rec_x = np.array([]).astype(int)

        for rxi in xsta.round(2):
            sid = np.where(np.abs(x.round(3)-rxi)<1.5e-2)
            id_rec_x = np.append(id_rec_x,sid)

        id_top_x = []
        id_top_y = []
        id_top_z = []

        for h in range(len(xtop)):

            for i in range(len(x)):
                if np.abs(xtop[h]-x[i])<1e-2:
                    id_top_x.append(i)

            for i in range(len(y)):
                if np.abs(ytop[h]-y[i])<1e-2:
                    id_top_y.append(i)

            for j in range(len(z)):    
                if np.abs(ztop[h]-z[j])<5e-3:
                    id_top_z.append(j)

        if args['regular_station']=='y':
            id_rec_x = id_top_x[::args['rec_spacing']]
            id_rec_y = id_top_y[::args['rec_spacing']]
            id_rec_z = id_top_z[::args['rec_spacing']]

        if args['append_shot']=='y':
            for i in range(8):
                id_sou_x = np.append(id_sou_x, len(x)-1-2*i)
                id_sou_y = np.append(id_sou_y, len(y)-1-2*i)
                id_sou_z = np.append(id_sou_z, len(z)-1-int(0.5*i))

    else:
        zeve, yeve, xeve = z[args['zid_source']]*np.ones_like(x[::deltas]), y[::deltas], x[::deltas]
        zsta, ysta, xsta = z[args['zid_receiver']]*np.ones_like(x[::deltar]), y[::deltar], x[::deltar]
        ztop, ytop, xtop = zmin*np.ones_like(x), np.copy(y), np.copy(x)

        idx_all = np.arange(X.size).reshape(X.shape)

        # Sources indices
        id_sou = idx_all[args['zid_source'], ::deltas, ::deltas].reshape(-1)

        # Receivers indices
        id_rec = idx_all[args['zid_receiver'], ::deltar, ::deltar].reshape(-1)

    # Keeping the number of shots fixed while centering the shots location
    if args['middle_shot']=='y':
        id_sou_left = x.shape[0]//2-len(id_sou_x)//2
        id_sou_x = np.array(range(id_sou_left, id_sou_left+len(id_sou_x)))
        id_sou_y = np.array(range(id_sou_left, id_sou_left+len(id_sou_y)))

    if args['explode_reflector']=='y':
        id_sou_x = np.arange(0, len(x), args['sou_spacing'])
        id_sou_y = np.arange(0, len(y), args['sou_spacing'])
        id_sou_z = np.ones_like(id_sou_x)*(len(z)-1)

    if args['empty_middle']=='y':
        id_sou, id_rec = (np.array(id_sou_x)<=(len(x)//2-50))|(np.array(id_sou_x)>=(len(x)//2+50)), (np.array(id_rec_x)<=(len(x)//2-50))|(np.array(id_rec_x)>=(len(x)//2+50))
        if args['field_synthetic']=='n':
            id_sou_x = np.array(id_sou_x)[id_sou]
            id_sou_y = np.array(id_sou_y)[id_sou]
            id_sou_z = np.array(id_sou_z)[id_sou]
        id_rec_x = np.array(id_rec_x)[id_rec]
        id_rec_y = np.array(id_rec_y)[id_rec]
        id_rec_z = np.array(id_rec_z)[id_rec]

    sz = Z.reshape(-1)[id_sou]
    sy = Y.reshape(-1)[id_sou]
    sx = X.reshape(-1)[id_sou]

    Z,Y,X,SX = np.meshgrid(z,y,x,sx,indexing='ij')
    _,_,_,SY = np.meshgrid(z,y,x,sy,indexing='ij')
    _,_,_,SZ = np.meshgrid(z,y,x,sz,indexing='ij')
    _,_,_,ID = np.meshgrid(z,y,x,np.arange(sx.size),indexing='ij')

    ## Sources location checkpointing
    # for i in range(len(id_sou)):
    #     print(np.unique(SX[:,:,:,i]), np.unique(SY[:,:,:,i]), np.unique(SZ[:,:,:,i]))

    if args['model_type']=='marmousi':
        vel = np.fromfile('../data/marmousi.bin', np.float32).reshape(221, 601)
        x1 = np.linspace(0, 5, 601)
        z1 = np.linspace(0, 1, 221) 
        x2 = np.linspace(0.25, 5, len(x))
        z2 = np.linspace(0.09, 0.55, len(z)) 
        f = interpolate.interp2d(x1, z1, vel, kind='cubic')
        vel = f(x2, z2)
        # Augment a 3D velocity volume frdeom 2D data
        vel3d = np.repeat(vel[:, np.newaxis, :], len(y), axis=1)
    elif args['model_type']=='seam':
        vel = np.load('/home/taufikmh/KAUST/spring_2022/constrained_eikonal/notebooks/PINNtomo/inputs/seam_model/vel_seam.npy')
        x1 = np.arange(0,1+0.01,0.01)
        z1 = np.arange(0,1+0.01,0.01)
        from scipy import interpolate
        f = interpolate.interp2d(x1, z1, vel, kind='cubic')
        vel = f(x, z)
        # Augment a 3D velocity volume from 2D data
        vel3d = np.repeat(vel[:, np.newaxis, :], len(y), axis=1)
    elif args['model_type']=='constant':
        vel = 4*np.ones((nz,nx))
    elif args['model_type']=='gradient':
        vel = 1 + 7*np.meshgrid(x,z)[1]
    elif args['model_type']=='arid':
        vel = np.fromfile('../data/seam_arid', np.float32).reshape(400,400,600)/1000
        vel3d = np.moveaxis(vel[::args['sampling_rate'],::args['sampling_rate'],::args['sampling_rate']], -1, 0)

    # Extending the velocity model in thirs dimension byy repeatin the array
    velmodel = np.repeat(vel3d[...,np.newaxis], sx.size,axis=2)

    if args['depth_shift']=='y':
        zmin, zmax, z, sz, Z, SZ = zmin+5, zmax+5, z+5, sz+5, Z+5, SZ+5

    X_star = [Z.reshape(-1,1), Y.reshape(-1,1), X.reshape(-1,1), SY.reshape(-1,1), SX.reshape(-1,1)] # Grid points for prediction 

    # Numerical traveltime
    T_data3d = numerical_traveltime3d(vel3d, len(x), len(y), len(z), len(id_sou), 
                                      xmin, ymin, zmin, deltax, deltay, deltaz, 
                                      [np.where(x==X[:,:,:,0].reshape(-1)[id_sou[i]])[0][0] for i in range(len(id_sou))], 
                                      [np.where(y==Y[:,:,:,0].reshape(-1)[id_sou[i]])[0][0] for i in range(len(id_sou))], 
                                      [np.where(z==Z[:,:,:,0].reshape(-1)[id_sou[i]])[0][0] for i in range(len(id_sou))])

    # ZX plane after
    plot_section(vel3d[:,10,:], 'v_true_zx.png', vmin=np.nanmin(velmodel)+0.1, 
                 vmax=np.nanmax(velmodel)-0.5, save_dir=args['save_folder'], aspect='equal',
                 xmin=xmin, xmax=xmax, zmin=zmin, zmax=zmax, 
                 sx=X[:,:,:,0].reshape(-1)[id_sou],sz=Z[:,:,:,0].reshape(-1)[id_sou],rx=X[:,:,:,0].reshape(-1)[id_rec],rz=Z[:,:,:,0].reshape(-1)[id_rec])

    # XY plane
    plot_section(vel3d[5,:,:], 'v_true_xy.png', vmin=np.nanmin(velmodel)+0.1, 
                 vmax=np.nanmax(velmodel)-0.5, save_dir=args['save_folder'], aspect='equal',
                 xmin=xmin, xmax=xmax, zmin=xmin, zmax=xmax, 
                 sx=X[:,:,:,0].reshape(-1)[id_sou],sz=Y[:,:,:,0].reshape(-1)[id_sou],rx=X[:,:,:,0].reshape(-1)[id_rec],rz=Y[:,:,:,0].reshape(-1)[id_rec])

    # ZY plane
    plot_section(vel3d[:,:,10], 'v_true_zy.png', vmin=np.nanmin(velmodel)+0.1, 
                 vmax=np.nanmax(velmodel)-0.5, save_dir=args['save_folder'], aspect='equal',
                 xmin=xmin, xmax=xmax, zmin=zmin, zmax=zmax, 
                 sx=Y[:,:,:,0].reshape(-1)[id_sou],sz=Z[:,:,:,0].reshape(-1)[id_sou],rx=Y[:,:,:,0].reshape(-1)[id_rec],rz=Z[:,:,:,0].reshape(-1)[id_rec])

    # Plots
    if args['model_type']=='checkerboard':
        plot_section((6 + 6.5217391304347826*Z[:,:,0])/args['scale_factor'], "v_back.png", 
                     save_dir=args['save_folder'], aspect='equal',
                     xmin=xmin, xmax=xmax, zmin=zmin, zmax=zmax, 
                     sx=x[id_sou_x],sz=z[id_sou_z],rx=x[id_rec_x],rz=z[id_rec_z])
        plot_section(velpert[:,:,0]/args['scale_factor'], "v_pert.png", 
                     save_dir=args['save_folder'], aspect='equal',
                     xmin=xmin, xmax=xmax, zmin=zmin, zmax=zmax, 
                     sx=x[id_sou_x],sz=z[id_sou_z],rx=x[id_rec_x],rz=z[id_rec_z])

    # Interpolation
    Td_nn = np.zeros_like(T_data3d)
    taudx_nn = np.zeros_like(T_data3d)

    Ti_data = np.zeros((len(id_rec)*len(id_sou)))
    xri = np.tile(X.reshape(-1)[id_rec], len(id_sou))
    yri = np.tile(Y.reshape(-1)[id_rec], len(id_sou))
    zri = np.tile(Z.reshape(-1)[id_rec], len(id_sou))

    xsi = np.repeat(X.reshape(-1)[id_sou], len(id_rec))
    ysi = np.repeat(Y.reshape(-1)[id_sou], len(id_rec))
    zsi = np.repeat(Z.reshape(-1)[id_sou], len(id_rec))

    for i in range(len(id_sou)):
        Ti_data[i*len(id_rec):(i+1)*len(id_rec)] = T_data3d[:,:,:,i].reshape(-1)[id_rec]

    rand_idx = np.random.permutation(np.arange(len(Ti_data)))

    X_ori = np.vstack((xri, yri, zri, xsi, ysi, zsi)).T
    y_ori = Ti_data

    X_all = X_ori[rand_idx,:]
    y_all = y_ori[rand_idx]

    X_all = torch.from_numpy(X_all).float()
    y_all = torch.from_numpy(y_all).float()

    X_ori = torch.from_numpy(X_ori).float()

    all_dataset = torch.utils.data.TensorDataset(X_all, y_all)

    # Use Pytorch's functionality to load data in batches. Here we use full-batch training again.
    all_loader = torch.utils.data.DataLoader(all_dataset, batch_size=16, shuffle=True)

    if data_type=='nn':
        torch.manual_seed(8888)
        model = FullyConnectedNetwork(6, 1, n_hidden=[args['data_neurons']]*args['data_layers'], act='elu')
        # optimizer = torch.optim.Adam(model.parameters(), lr=5e-4) # best
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=int(4*args['reduce_after']), verbose=True)
        criterion = torch.nn.MSELoss()
        model.train()
        loss_data = []
        for epoch in range(int(5e3)):
            total_loss = 0.
            model.train()
            loss = 0
            for x_i, y_i in all_loader:
                optimizer.zero_grad()
                yest = model(x_i).view(-1)
                loss = criterion(yest, y_i)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 50 == 0 and epoch > 0:
                print(f'Epoch: {epoch}, Loss: {total_loss / X_all[:,0].detach().numpy().size}')
            mean_loss = total_loss / X_all[:,0].detach().numpy().size
            # wandb.log({"data_loss": mean_loss})
            scheduler.step(mean_loss)
            loss_data.append(mean_loss)

        Td_nn = np.zeros_like(T_data3d)

        if args['field_synthetic']=='y':
            X_all = [np.tile(x[id_top_x], len(sx)), np.tile(z[id_top_z], len(sz)), 
                     np.repeat(x[id_sou_x], len(x[id_top_x])), 
                     np.repeat(z[id_sou_z], len(z[id_top_z]))]
        else:
            X_all = [np.tile(x, len(sx)), 
                     np.tile(y, len(sy)),
                     np.tile(z[args['zid_receiver']]*np.ones_like(x), len(sz)), 
                     np.repeat(x[id_sou_x], len(x)), 
                     np.repeat(y[id_sou_y], len(y)), 
                     np.repeat(z[id_sou_z], len(z))]

        model.eval()
        Td_pred = model(torch.FloatTensor(X_all).T)

        for i in range(len(id_sou)):
            Td_nn[:,:,:,i] = Td_pred[i*len(x):(i+1)*len(x)].detach().numpy().reshape(-1)

        # Convergence history plot for verification
        fig = plt.figure()
        ax = plt.axes()
        ax.semilogy(loss_data)

        ax.set_xlabel('Epochs',fontsize=14)

        plt.xticks(fontsize=11)

        ax.set_ylabel('Loss',fontsize=14)
        plt.yticks(fontsize=11);
        plt.grid()
        plt.savefig(os.path.join(wandb_dir, "data_loss.png"), format='png', bbox_inches="tight")

        # Save model
        torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_data
        }, wandb_dir+'/saved_data_model')

    # Analytical solution for the known traveltime part
    if args['depth_shift']=='y':
        vs = args['initial_velocity'] #velmodel[np.round((SZ-5)/deltaz).astype(int),np.round(SX/deltax).astype(int),0]
    else:
        vs = vel3d[np.round(SZ/deltaz).astype(int),np.round(SY/deltay).astype(int),np.round(SX/deltax).astype(int)]

    T0 = np.sqrt((Z-SZ)**2 + (Y-SY)**2 + (X-SX)**2)/vs;
    px0 = np.divide(X-SX, T0*vs**2, out=np.zeros_like(T0), where=T0!=0)
    py0 = np.divide(Y-SY, T0*vs**2, out=np.zeros_like(T0), where=T0!=0)
    pz0 = np.divide(Z-SZ, T0*vs**2, out=np.zeros_like(T0), where=T0!=0)

    if args['field_synthetic']=='y':
        xf = np.arange(xmin,xmax+0.1*deltax,0.1*deltax)
        zf = np.arange(zmin,zmax+0.1*deltaz,0.1*deltaz)
        T_topo = np.zeros((len(zf), len(xf), len(id_sou_x)))
        for i in range(len(id_sou_x)):
            f = interpolate.interp2d(x, z, T_data3d[:,:,i], kind='cubic')
            T_topo[:,:,i] = f(xf, zf)
        id_top_x = []
        id_top_z = []

        for h in range(len(xtop)):

            for i in range(len(x)):
                if np.abs(xtop[h]-x[i])<1e-2:
                    id_top_x.append(i)

            for j in range(len(z)):    
                if np.abs(ztop[h]-z[j])<5e-3:
                    id_top_z.append(j)

        taud_topo = np.divide(T_data3d, T0, where=T0!=0)[id_top_z, id_top_x, :]
        T_topo = T_data[id_top_z, id_top_x, :]

        taud_topo = np.repeat(taud_topo, nz).reshape(nx,len(id_sou_x),nz).swapaxes(1,2).swapaxes(0,1)
        T_topo = np.repeat(T_topo, nz).reshape(nx,len(id_sou_x),nz).swapaxes(1,2).swapaxes(0,1)

    Td_hc = np.zeros_like(T0)
    T0_hc = np.zeros_like(T0)
    taud_hc = np.zeros_like(T0)
    taudx_hc = np.zeros_like(T0)
    taudy_hc = np.zeros_like(T0)

    for i in range(len(id_sou)):
        T0_hc[:,:,:,i] = np.moveaxis(np.tile(T0.reshape(X.shape)[args['zid_receiver'],:,:,i], nz).reshape(ny,nz,nx), 1, 0)
        # np.tile(T0[args['zid_receiver'],:,:,i], nz).reshape(nz,ny,nx)

        # Numerical
        if data_type=='full':
            Td_hc[:,:,:,i] = np.moveaxis(np.tile(T_data3d[args['zid_receiver'],:,:,i], nz).reshape(ny,nz,nx), 1, 0)
        # np.tile(T_data3d[args['zid_receiver'],:,:,i], nz).reshape(nz,ny,nx)

        # NN-based interpolation
        elif data_type=='nn':
            Td_hc[:,:,:,i] = Td_nn[:,:,:,i].reshape(nz,ny,nx)

        if args['factorization_type']=='multiplicative':   
            taud_hc[:,:,:,i] = np.divide(Td_hc[:,:,:,i], T0_hc[:,:,:,i], out=np.ones_like(T0_hc[:,:,:,i]),
                                       where=T0_hc[:,:,:,i]!=0)
        else:
            taud_hc[:,:,:,i] = Td_hc[:,:,:,i] - T0_hc[:,:,:,i]

        # Numerical
        if data_type=='full':
            taudy_hc[:,:,:,i] = np.gradient(taud_hc.reshape(X.shape)[:,:,:,i], deltay, axis=1)
            taudx_hc[:,:,:,i] = np.gradient(taud_hc.reshape(X.shape)[:,:,:,i], deltax, axis=2)

        # NN-based interpolation
        elif data_type=='nn':
            taudy_hc[:,:,:,i] = np.gradient(taud_hc.reshape(X.shape)[:,:,:,i], deltay, axis=1)        
            taudx_hc[:,:,:,i] = np.gradient(taud_hc.reshape(X.shape)[:,:,:,i], deltax, axis=2)

    if args['field_synthetic']=='y':

        NAN = np.ones_like(X)
        for i in range(z.shape[0]):
            for j in range(x.shape[0]):
                if z[i] < Z[id_top_z, id_top_x, 0][j]:
                    NAN[i,j,:] = float("Nan")

    # Locate source boolean
    import time
    start_time = time.time()

    sids = id_sou

    # Locate source boolean
    isource = np.ones_like(X_star[0]).reshape(-1,).astype(bool)
    isource[sids] = False

    velmodel = vel3d.reshape(-1,1)
    px0 = px0.reshape(-1,1)
    py0 = py0.reshape(-1,1)
    pz0 = pz0.reshape(-1,1)
    T0 = T0.reshape(-1,1)
    T_data = T_data3d.reshape(-1,1)
    taud = taud_hc.reshape(-1,1)

    if args['factorization_type']=='multiplicative':
        taud[~isource] = 1.    
    taudx = taudx_hc.reshape(-1,1)
    taudy = taudy_hc.reshape(-1,1)
    index = ID.reshape(-1,1)

    perm_id = np.random.permutation(X.size-sx.size)
    
    rx=X[:,:,:,0].reshape(-1)[id_rec]
    ry=Y[:,:,:,0].reshape(-1)[id_rec]
    rz=Z[:,:,:,0].reshape(-1)[id_rec]
    
    return X, Y, Z, SX, SY, SZ, taud, taudx, taudy, T0, px0, py0, pz0, index, sx, sy, sz, rx, ry, rz, vel3d, id_rec, id_sou

# Lightning Dataset
import copy
class HCEikonalPINNsData(pl.LightningDataModule):
    def __init__(
        self,
        args,
        batch_size: int = 2**26 ,
        num_workers: int = int(os.cpu_count() / 2),
        permute=False
    ):
        super().__init__()
        self.fast_loader = args['fast_loader']
        self.batch_size = batch_size
        self.num_workers = num_workers
        X, Y, Z, SX, SY, SZ, taud, taudx, taudy, T0, px0, py0, pz0, index, self.sx, self.sy, self.sz, rx, ry, rz, self.vel3d, self.id_rec, self.id_sou = setup_medium(args)
        self.input_list = [X, Y, Z, SX, SY, SZ, taud, taudx, taudy, T0, px0, py0, pz0, index]
        if permute:
            perm_id = np.random.permutation(X.size-self.sx.size)
            self.input_list = [i[perm_id] for i in self.input_list]
        self.input_dataset = self.create_dataset([i.ravel() for i in self.input_list])

    def prepare_data(self):
        dummy_data = None
        
    def create_dataset(self, input_list):
        
        XYZ = torch.from_numpy(np.vstack((input_list[0].ravel(), input_list[1].ravel(), input_list[2].ravel())).T).float()
        SX = torch.from_numpy(input_list[3]).ravel().float()
        SY = torch.from_numpy(input_list[4]).ravel().float()
        SZ = torch.from_numpy(input_list[5]).ravel().float()

        taud = torch.from_numpy(input_list[6]).ravel().float()
        taud_dx = torch.from_numpy(input_list[7]).ravel().float()
        taud_dy = torch.from_numpy(input_list[8]).ravel().float()

        tana = torch.from_numpy(input_list[9]).ravel().float()
        tana_dx = torch.from_numpy(input_list[10]).ravel().float()
        tana_dy = torch.from_numpy(input_list[11]).ravel().float()
        tana_dz = torch.from_numpy(input_list[12]).ravel().float()

        index = torch.from_numpy(input_list[13]).ravel().float()
    
        return TensorDataset(XYZ, SX, SY, SZ, taud, taud_dx, taud_dy, tana, tana_dx, tana_dy, tana_dz, index)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_input = self.input_dataset
            
    def train_dataloader(self):
        
        if self.fast_loader=='y':
            return FastTensorDataLoader(
            torch.from_numpy(np.vstack((self.input_list[0].ravel(), self.input_list[1].ravel(), self.input_list[2].ravel())).T).float(),
            torch.from_numpy(self.input_list[3]).ravel().float(),
            torch.from_numpy(self.input_list[4]).ravel().float(),
            torch.from_numpy(self.input_list[5]).ravel().float(),
            torch.from_numpy(self.input_list[6]).ravel().float(),
            torch.from_numpy(self.input_list[7]).ravel().float(),
            torch.from_numpy(self.input_list[8]).ravel().float(),
            torch.from_numpy(self.input_list[9]).ravel().float(),
            torch.from_numpy(self.input_list[10]).ravel().float(),
            torch.from_numpy(self.input_list[11]).ravel().float(),
            torch.from_numpy(self.input_list[12]).ravel().float(),
            batch_size=self.batch_size, 
            shuffle=True
    )
        
        else:
            return DataLoader(
                self.train_input,
                batch_size=self.batch_size,
                pin_memory=True,
                persistent_workers=True,
                shuffle=True,
                num_workers=self.num_workers,
            )

# Lightning Model
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

class HCEikonalPINNsModel(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
                                                               
        self.optim_type = args['optimizer']
        
        self.register_buffer("seed", torch.tensor(args['seed']))
        self.register_buffer("learning_rate", torch.tensor(args['learning_rate']))    

        # Computational model parameters    
        deltar = args['rec_spacing']
        deltas = args['sou_spacing']
        zmin = 0.; zmax = args['max_depth']
        ymin = 0.; ymax = args['max_offset']
        xmin = 0.; xmax = args['max_offset']
        deltax, deltay, deltaz = args['sampling_rate']*0.00625/2*4, args['sampling_rate']*0.00625/2*4, args['sampling_rate']*0.00625/2
        
        z = torch.arange(zmin,zmax,deltaz)
        y = torch.arange(ymin,ymax,deltay)
        x = torch.arange(xmin,xmax,deltax)
        
        Z,Y,X = np.meshgrid(z,y,x,indexing='ij')
        
        idx_all = np.arange(X.size).reshape(X.shape)
        id_sou = idx_all[args['zid_source'], ::deltas, ::deltas].reshape(-1)
        id_rec = idx_all[args['zid_receiver'], ::deltar, ::deltar].reshape(-1)
        
        self.register_buffer("x", x)
        self.register_buffer("y", y)
        self.register_buffer("z", z)

        self.register_buffer("sx", torch.tensor(X.reshape(-1)[id_sou]))
        self.register_buffer("sy", torch.tensor(Y.reshape(-1)[id_sou]))
        self.register_buffer("sz", torch.tensor(Z.reshape(-1)[id_sou]))
        
        self.register_buffer("rx", torch.tensor(X.reshape(-1)[id_rec]))
        self.register_buffer("ry", torch.tensor(Y.reshape(-1)[id_rec]))
        self.register_buffer("rz", torch.tensor(Z.reshape(-1)[id_rec]))
        self.register_buffer("sid", torch.arange(torch.tensor(self.sx.shape[0])))
        
        self.register_buffer("bias", torch.tensor(0.2))
        self.register_buffer("mean", torch.tensor(0.01))
        self.register_buffer("std", torch.tensor(0.05))
        self.register_buffer("v_scaler", torch.tensor(1.))
        self.register_buffer("num_epochs", torch.tensor(args['num_epochs']))
        self.register_buffer("reduce_after", torch.tensor(args['reduce_after']))
        
        # network
        if args['residual_network']=='n':
            self.tau_model = FullyConnectedNetwork(4, 1, [args['num_neurons']]*args['num_layers'], last_act=args['tau_act'], act=args['activation'], lay='linear', last_multiplier=args['tau_multiplier'])
            
            self.v_model = FullyConnectedNetwork(3, 1, [args['num_neurons']//2]*args['num_layers'], act='relu', lay='linear', last_act='relu', last_multiplier=args['v_multiplier'])
        else:
            self.tau_model = ResidualNetwork(4, 1, num_neurons=args['num_neurons'], num_layers=args['num_layers'], act=args['activation'], lay='linear', last_multiplier=args['tau_multiplier'])
            
            self.v_model = ResidualNetwork(3, 1, num_neurons=args['num_neurons']//2, act='relu', last_act='relu', num_layers=args['num_layers'], lay='linear', last_multiplier=args['v_multiplier'])
        
        self.v_model = self.v_model.apply(lambda m: init_weights(m, init_type=args['initialization'], bias=self.bias, mean=self.mean, std=self.std))

    def configure_optimizers(self):
        if self.optim_type == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-5)
        elif self.optim_type == 'lbfgs':
            self.optimizer = torch.optim.LBFGS(list(self.tau_model.parameters()) + list(self.v_model.parameters()), line_search_fn="strong_wolfe")
        scheduler = {'scheduler': ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=self.num_epochs//self.reduce_after, verbose=True), 
                     'interval': 'epoch', 
                     'monitor':'train_pde_loss'}
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch[0],batch[1],batch[2],batch[3],batch[4],batch[5],batch[6],batch[7],batch[8],batch[9],batch[10],batch[11])
        self.log("train_pde_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def evaluate_velocity(self, data_loader, batch_size, num_pts):
        self.v_model.eval()

        # Prepare input
        with torch.no_grad():
            V = torch.empty(num_pts)
            for i, X in enumerate(data_loader):

                # Compute v
                batch_end = (i+1)*batch_size if (i+1)*batch_size<num_pts else i*batch_size + X[0].shape[0]
                V[i*batch_size:batch_end] = self.v_model(X[0]).view(-1)

        return V
    
    def evaluate_tau(self, data_loader, batch_size, num_pts):
        self.tau_model.eval()

        # Prepare input
        with torch.no_grad():
            T = torch.empty(num_pts)
            for i, X in enumerate(data_loader, 0):

                xyzs = torch.hstack((X[0], X[-1].view(-1,1)))
                batch_end = (i+1)*batch_size if (i+1)*batch_size<num_pts else i*batch_size + X[0].shape[0]
                T[i*batch_size:batch_end] = self.tau_model(xyzs).view(-1)

        return T
    
    def forward(self, xyz, sx, sy, sz, taud, taud_dx, taud_dy, t0, t0_dx, t0_dy, t0_dz, idx):
        
        ic = torch.hstack((self.sx.view(-1,1), self.sy.view(-1,1), self.sz.view(-1,1))).float()
        
        # Number of source
        num_sou = len(ic[:,0])

        xyz.requires_grad = True

        # Input for the velocity network
        xyzic = torch.cat([xyz, ic])

        # Source location
        sxic = torch.cat([sx, ic[:,0]])
        syic = torch.cat([sy, ic[:,1]])
        szic = torch.cat([sz, ic[:,2]])
        sidx = torch.cat([idx, self.sid])

        # Input for the data network
        # xyzsic = torch.hstack((xyzic, sxic.view(-1,1), syic.view(-1,1), szic.view(-1,1)))
        xyzsic = torch.hstack((xyzic, sidx.view(-1,1)))

        # Compute T
        tau = self.tau_model(xyzsic).view(-1)

        # Compute v
        v = self.v_model(xyzsic[:, :3]).view(-1)

        # Gradients
        gradient = torch.autograd.grad(tau, xyzsic, torch.ones_like(tau), create_graph=True)[0]
        
        tau_dx = gradient[:, 0]
        tau_dy = gradient[:, 1]
        tau_dz = gradient[:, 2]
        
        rec_op = xyz[:,2]
        rec_op_dz = 1
                
        T_dx = rec_op*tau_dx[:-num_sou] + taud_dx + t0_dx
        T_dy = rec_op*tau_dy[:-num_sou] + taud_dy + t0_dy
        T_dz = rec_op*tau_dz[:-num_sou] + rec_op_dz*tau[:-num_sou] + t0_dz
        
        vscaler = 1
        
        pde_lhs = (T_dx**2 + T_dy**2 + T_dz**2) * vscaler

        pde = pde_lhs - vscaler / (v[:-num_sou] ** 2)
        
        wl2=1        
        
        return torch.mean(wl2*pde**2)