import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io 
import time
import random
import os
import wandb

from torch.optim.lr_scheduler import ReduceLROnPlateau
from hcpinnseikonal.utils import create_dataloaderwell, SaveBestModels

# Load style @hatsyim
# plt.style.use("~/science.mplstyle")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
plt.rcParams['figure.figsize'] =  [6.4, 4.8]

from scipy import interpolate

# Velocity model
def velocity_model(xmin, xmax, zmin, zmax, x, z, sx, sz, args):
    if args['model_type']=='seam':

        vel = np.load('/home/taufikmh/KAUST/spring_2022/constrained_eikonal/notebooks/PINNtomo/inputs/seam_model/vel_seam.npy')

        x1 = np.arange(xmin,1+deltax,deltax)
        z1 = np.arange(zmin,1+deltaz,deltaz)
        from scipy import interpolate
        f = interpolate.interp2d(x1, z1, vel, kind='cubic')
        vel = f(x, z)

        # Extending the velocity model in thirs dimension byy repeatin the array
        velmodel = np.repeat(vel[...,np.newaxis],sx.size,axis=2)

    elif args['model_type']=='ek137':

        # Load data
        colnames=['depth', 'vp', 'vs', 'rho']
        ek137 = pd.read_csv('/home/taufikmh/KAUST/fall_2022/GFATT_PINNs/data/ek137.tvel', skiprows=2, header=None, delim_whitespace=1, names=colnames)
        depth_ek137 = ek137.depth.values
        vp_ek137 = np.repeat(ek137.vp.values, ek137.vp.values.shape[0]).reshape(ek137.vp.values.shape[0], -1)

        # Interpolate the input to match the medium size
        if args['until_cmb']:
            id_depth = 67 
        else:
            id_depth = depth_ek137.shape[0]
        print((ek137.vp.values[:id_depth].shape[0]-1)*deltaz+deltaz)
        x1 = np.arange(xmin,(ek137.vp.values[:id_depth].shape[0]-1)*deltax+deltax,deltax)
        z1 = np.arange(xmin,(ek137.vp.values[:id_depth].shape[0]-1)*deltaz+deltaz,deltaz)
        from scipy import interpolate
        f = interpolate.interp2d(x1, z1, vp_ek137[:id_depth, :id_depth], kind='cubic')
        vel = f(x, z)/args['scale_factor']

        # Extending the velocity model in thirs dimension byy repeatin the array
        velmodel = np.repeat(vel[...,np.newaxis],sx.size,axis=2)

    elif args['model_type']=='overthrust':

        # Load data
        filename = '/home/taufikmh/KAUST/spring_2022/constrained_eikonal/data/overthrust_vz.csv'
        data = pd.read_csv(filename, index_col=None, header=None)
        temp = np.zeros((101,401))
        data = np.array(data)
        temp[:-1,:] = data
        temp[-1,:] = data[-1,:]
        vel = temp[:, 10:211]

        # Interpolate the input to match the medium size
        x1 = np.arange(xmin,5+5/2*deltax,5/2*deltax)
        z1 = np.arange(zmin,1+deltaz,deltaz)
        from scipy import interpolate
        f = interpolate.interp2d(x1, z1, vel, kind='cubic')
        vel = f(x, z)

        # Extending the velocity model in thirs dimension byy repeatin the array
        velmodel = np.repeat(vel[...,np.newaxis],sx.size,axis=2)
    elif args['model_type']=='checkerboard':

        velpert = 0.5*np.sin(5*X)*np.sin(5*Z)
        vel = 6 + 6.5217391304347826*Z + velpert
        vel = vel[:,:,0]/args['scale_factor']

        # Extending the velocity model in thirs dimension byy repeatin the array
        velmodel = np.repeat(vel[...,np.newaxis],sx.size,axis=2)

    elif args['model_type']=='knox':

        vel = np.ones((351,351))

        # Discretize points depth and velocity
        zd_lvl = np.array([5,12,40,77,140,350])
        vd_lvl = np.array([610,610,1860,670,1340,2240])

        # Normalize depth to 0-1 km and velocity to 0-5 km/s
        zd_lvl = zd_lvl/zd_lvl.max()
        vd_lvl = vd_lvl*2e-3

        # Build continuous 1-D velocity
        z_lvl = np.linspace(0,1,351)
        v_lvl = np.linspace(0,1,351)
        i=0
        for idz in range(len(z_lvl)):
            for idp in range(len(zd_lvl)):
                if z_lvl[idz] == zd_lvl[idp]:
                    j = idz+1
                    v_lvl[i:j] = vd_lvl[idp]
                    i = j

        # Interpolate the input to match the medium size
        for i in range(350):
            vel[:,i] = v_lvl
        x1 = np.linspace(0,10,351)
        z1 = np.linspace(0,1,351)
        from scipy import interpolate
        f = interpolate.interp2d(x1, z1, vel, kind='cubic')
        vel = f(x, z)        

        # Extending the velocity model in thirs dimension byy repeatin the array
        velmodel = np.repeat(vel[...,np.newaxis],sx.size,axis=2)

    elif args['model_type']=='undulation':

        vel = np.load('/home/taufikmh/KAUST/fall_2022/GFATT_PINNs/data/undulation.npy')

        x1 = np.linspace(xmin,xmax,vel.shape[1])
        z1 = np.linspace(zmin,zmax,vel.shape[0])
        from scipy import interpolate
        f = interpolate.interp2d(x1, z1, vel, kind='cubic')
        vel = f(x, z)*args['scale_factor']

        # Extending the velocity model in thirs dimension byy repeatin the array
        velmodel = np.repeat(vel[...,np.newaxis],sx.size,axis=2)

    elif args['model_type']=='knox_salt':

        vel = np.ones((351,351))

        # Discretize points depth and velocity
        zd_lvl = np.array([5,12,40,77,140,350])
        vd_lvl = np.array([610,610,1860,670,1340,2240])

        # Normalize depth to 0-1 km and velocity to 0-5 km/s
        zd_lvl = zd_lvl/zd_lvl.max()
        vd_lvl = vd_lvl*2e-3

        # Build continuous 1-D velocity
        z_lvl = np.linspace(0,1,351)
        v_lvl = np.linspace(0,1,351)
        i=0
        for idz in range(len(z_lvl)):
            for idp in range(len(zd_lvl)):
                if z_lvl[idz] == zd_lvl[idp]:
                    j = idz+1
                    v_lvl[i:j] = vd_lvl[idp]
                    i = j

        # Interpolate the input to match the medium size
        for i in range(350):
            vel[:,i] = v_lvl
        x1 = np.linspace(0,10,351)
        z1 = np.linspace(0,1,351)
        from scipy import interpolate
        f = interpolate.interp2d(x1, z1, vel, kind='cubic')
        vel = f(x, z)        

        a = 25.0
        b = 5.0

        ellipse = (X[:,:,0]-x[len(x)//2])**2/ a**2 + (Z[:,:,0]-z[(len(z)//2-20)])**2/ b**2
        grey = np.zeros((nz,nx), dtype=np.int32)
        grey[ellipse < 0.0025] = 1

        vel = vel + 2*grey

        # Extending the velocity model in thirs dimension byy repeatin the array
        velmodel = np.repeat(vel[...,np.newaxis],sx.size,axis=2)

    elif args['model_type']=='arid':

        from scipy import interpolate
        vel = np.fromfile('../data/seam_arid', np.float32).reshape(400,400,600)[90,:,25:325].T/1000
        x1 = np.linspace(xmin, xmax, 400)
        z1 = np.linspace(zmin, zmax, 300) 
        x2 = np.linspace(xmin, xmax, len(x))
        z2 = np.linspace(zmin, zmax, len(z)) 
        f = interpolate.interp2d(x1, z1, vel, kind='cubic')
        vel = f(x2, z2)

        # Extending the velocity model in thirs dimension byy repeatin the array
        velmodel = np.repeat(vel[...,np.newaxis],sx.size,axis=2)
        
    return vel, velmodel

def setup_acquisition(xmin, xmax, zmin, zmax, x, z, deltar, deltas, args):
    if args['field_synthetic']=='y':
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

        ztop,zsta,zeve = ztop-ztop.min(),zsta-zsta.min(),zeve-zeve.min()
        ztop,zsta,zeve = args['station_factor']*ztop/ztop.max()+zmin,args['station_factor']*zsta/zsta.max()+zmin,zmax-args['event_factor']*zeve/zeve.max()

        xsta,xeve,zsta,zeve = xsta[(xsta>xtop.min()) & (xsta<xtop.max())],xeve[(xeve>xtop.min()) & (xeve<xtop.max())],zsta[(xsta>xtop.min()) & (xsta<xtop.max())],zeve[(xeve>xtop.min()) & (xeve<xtop.max())]

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

        id_sou_x = np.array([]).astype(int)

        for sxi in xeve.round(2):
            sid = np.where(np.abs(x.round(3)-sxi)<1.5e-2)
            id_sou_x = np.append(id_sou_x,sid)

        id_rec_x = np.array([]).astype(int)

        for rxi in xsta.round(2):
            sid = np.where(np.abs(x.round(3)-rxi)<1.5e-2)
            id_rec_x = np.append(id_rec_x,sid)

        id_top_x = []
        id_top_z = []

        for h in range(len(xtop)):

            for i in range(len(x)):
                if np.abs(xtop[h]-x[i])<1e-2:
                    id_top_x.append(i)

            for j in range(len(z)):    
                if np.abs(ztop[h]-z[j])<5e-3:
                    id_top_z.append(j)

        if args['regular_station']=='y':
            id_rec_x = id_top_x[::args['rec_spacing']]
            id_rec_z = id_top_z[::args['rec_spacing']]

        if args['append_shot']=='y':
            for i in range(8):
                id_sou_x = np.append(id_sou_x, len(x)-1-2*i)
                id_sou_z = np.append(id_sou_z, len(z)-1-int(0.5*i))

        plt.plot(args['plotting_factor']*(xtop-xtop.min()), args['plotting_factor']*ztop)
        plt.scatter(args['plotting_factor']*(xeve-xtop.min()), args['plotting_factor']*zeve)
        plt.scatter(x[id_rec_x], z[id_rec_z], c='y', marker='v')
        plt.title('Cross-section')
        plt.xlabel('X (km)')
        plt.ylabel('Z (km)')
        plt.gca().invert_yaxis()
        plt.axis('tight')
        plt.savefig(os.path.join(wandb_dir, 'cross_section.png'), format='png', bbox_inches="tight")
    else:
        zeve, xeve = z[args['zid_source']]*np.ones_like(x[::deltas]), x[::deltas]
        zsta, xsta = z[args['zid_receiver']]*np.ones_like(x[::deltar]), x[::deltar]
        ztop, xtop = zmin*np.ones_like(x), np.copy(x)

        id_sou_z = np.array([]).astype(int)

        for szi in zeve.round(2):
            sid = np.where(np.abs(z.round(4)-szi)<1e-6)
            id_sou_z = np.append(id_sou_z,sid)

        id_sou_x = np.array([]).astype(int)

        for sxi in xeve.round(2):
            sid = np.where(np.abs(x.round(4)-sxi)<1e-6)
            id_sou_x = np.append(id_sou_x,sid)

        id_rec_z = np.array([]).astype(int)

        for szi in zsta.round(2):
            sid = np.where(np.abs(z.round(4)-szi)<1e-6)
            id_rec_z = np.append(id_rec_z,sid)

        id_rec_x = np.array([]).astype(int)

        for sxi in xsta.round(2):
            sid = np.where(np.abs(x.round(4)-sxi)<1e-6)
            id_rec_x = np.append(id_rec_x,sid)

    # Keeping the number of shots fixed while centering the shots location
    if args['middle_shot']=='y':
        id_sou_left = x.shape[0]//2-len(id_sou_x)//2
        id_sou_x = np.array(range(id_sou_left, id_sou_left+len(id_sou_x)))

    if args['explode_reflector']=='y':
        id_sou_x = np.arange(0, len(x), args['sou_spacing'])
        id_sou_z = np.ones_like(id_sou_x)*(len(z)-1)

    if args['empty_middle']=='y':
        id_sou, id_rec = (np.array(id_sou_x)<=(len(x)//2-50))|(np.array(id_sou_x)>=(len(x)//2+50)), (np.array(id_rec_x)<=(len(x)//2-50))|(np.array(id_rec_x)>=(len(x)//2+50))
        if args['field_synthetic']=='n':
            id_sou_x = np.array(id_sou_x)[id_sou]
            id_sou_z = np.array(id_sou_z)[id_sou]
        id_rec_x = np.array(id_rec_x)[id_rec]
        id_rec_z = np.array(id_rec_z)[id_rec]
        
    return zeve, xeve, zsta, xsta, ztop, xtop, id_rec_x, id_rec_z, id_sou_x, id_sou_z
    
def numerical_traveltime(vel, nx, nz, ns, xmin, zmin, deltax, deltaz, id_sou_x, id_sou_z):
    
    import pykonal

    T_data_surf = np.zeros((nz,nx,ns))

    for i in range(ns):

        solver = pykonal.EikonalSolver(coord_sys="cartesian")
        solver.velocity.min_coords = zmin, xmin, zmin
        solver.velocity.node_intervals = deltaz, deltax, deltaz
        solver.velocity.npts = nz, nx, 1
        solver.velocity.values = vel.reshape(nz,nx,1)

        src_idx = id_sou_z[i], id_sou_x[i], 0
        
        solver.traveltime.values[src_idx] = 0
        solver.unknown[src_idx] = False
        solver.trial.push(*src_idx)

        solver.solve()

        teik = solver.traveltime.values

        T_data_surf[:,:,i] = teik[:,:,0]
        
    return T_data_surf

# Training functions
import random
import numpy as np
from zmq import device
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def velocity_well(v_well, v_nn, x_well, x, args):
    
    first_x, second_term = 1, 0
    
    for i in range(x_well.size()[0]):
        
        if args['v_function']=='l2':
            well_op = ((x-x_well[i])/(x-x_well).max())**2
        elif args['v_function']=='l1':
            well_op = 10*((x-x_well[i])/((x-x_well[i]).max()))
        elif args['v_function']=='exp':
            well_op = 10*(1-torch.exp(((x-x_well[i]))))
        elif args['v_function']=='try':
            well_op = 1
        
        first_x *= well_op #(x-x_well[i])
        second_x = 1
        
        for j in range(x_well.size()[0]):
            if x_well.size()[0]>1:
                if i!=j:
                    second_x *= (x-x_well[j])/((x_well[i]-x_well[j]))
            else:
                second_x = 1
                
        second_term += second_x * v_well
    
    # print(first_x)
    
    return first_x * v_nn + second_term

def train(tau_model, v_model, v_optimizer, tau_optimizer, data_loader, ic, vic, x_well, args, vscaler=1., device='cuda',wandb=False):
    tau_model.train()
    v_model.train()
    loss = []
    
    for xz, s, taud, taud_dx, t0, t0_dx, t0_dz, v0, nuw in data_loader:
        
        # Number of source
        num_sou = len(ic[:,0])

        xz.requires_grad = True

        # Input for the velocity network
        xzic = torch.cat([xz, ic])

        # Source X's location
        sic = torch.cat([s, ic[:,0]])

        # Input for the data network
        xzsic = torch.hstack((xzic, sic.view(-1,1)))
                
        # Compute T
        tau = tau_model(xzsic).view(-1)

        # Compute v
        v0ic = torch.cat((v0, vic[:,0]))
        nuwic = torch.cat((nuw, vic[:,1]))
        
        # Compute v
        if args['with_well']=='y':
            v = velocity_well(nuwic*v0ic, v_model(xzsic[:, :2]).view(-1), x_well, xzsic[:,0], args)
            
            # # Compute v
            # if args['v_function']=='l2':
            #     v = ((xzsic_nograd[:,0] - x_well)/(xzsic_nograd[:,0]-x_well).max())**2*v_model(xzsic[:, :2]).view(-1) + nuwic*v0ic #v_model(xzsic[:, :2]).view(-1)
            # elif args['v_function']=='l1':
            #     v = 10*((xzsic_nograd[:,0] - x_well)/(xzsic_nograd[:,0]-x_well).max())*v_model(xzsic[:, :2]).view(-1) + nuwic*v0ic #v_model(xzsic[:, :2]).view(-1)
            # elif args['v_function']=='exp':
            #     v = 4*(1-torch.exp((xzsic_nograd[:,0] - x_well)))*v_model(xzsic[:, :2]).view(-1) + nuwic*v0ic
            # elif args['v_function']=='try':
            #     v = v_model(xzsic_nograd[:, :2]).view(-1) + nuwic*v0ic
        else:
            v = v_model(xzsic[:, :2]).view(-1)
            
        # Gradients
        gradient = torch.autograd.grad(tau, xzsic, torch.ones_like(tau), create_graph=True)[0]
        
        tau_dx = gradient[:, 0]
        tau_dz = gradient[:, 1]
    
        # Loss function based on the factored isotropic eikonal equation
        if args['tau_function']=='exp':
            rec_op = (1-torch.exp((xz[:,1])**args['exp_factor']))

            # Initialize output tensor with desired value
            rec_op_dz = torch.full_like(xz[:,1], fill_value=0.)

            # Zero mask
            mask = (xz[:,1] != 0)

            rec_op_dz[mask] = (-args['exp_factor']*torch.exp((xz[:,1][mask])**args['exp_factor'])/(xz[:,1][mask]**(1-args['exp_factor'])))
        else:
            rec_op = xz[:,1]
            rec_op_dz = 1
            
        if args['factorization_type']=='multiplicative':
            T_dx = (rec_op*tau_dx[:-num_sou] + taud_dx)*t0 + (rec_op*tau[:-num_sou] + taud)*t0_dx
            T_dz = (rec_op*tau_dz[:-num_sou] + rec_op_dz*tau[:-num_sou])*t0 + (rec_op*tau[:-num_sou] + taud)*t0_dz
        else:
            T_dx = rec_op*tau_dx[:-num_sou] + taud_dx + t0_dx
            T_dz = rec_op*tau_dz[:-num_sou] + rec_op_dz*tau[:-num_sou] + t0_dz
        
        pde_lhs = (T_dx**2 + T_dz**2) * vscaler
        pde = pde_lhs - vscaler / (v[:-num_sou] ** 2)
        ls_pde = torch.mean(pde**2)
        
        # if args['with_well']=='y':
        #     ls = ls_pde + torch.mean(pde**2)
        # else:
        ls = ls_pde

        loss.append(ls.item())
        
        ls.backward()
            
        v_optimizer.step()
        if tau_optimizer is not None:
            tau_optimizer.step()
            
        v_optimizer.zero_grad()
        if tau_optimizer is not None:
            tau_optimizer.zero_grad()

        del xz, xzic, sic, xzsic, taud, taud_dx, t0, t0_dx, t0_dz, ls, v, tau, tau_dx, tau_dz, gradient, num_sou, T_dx, T_dz, pde_lhs, ls_pde, pde, rec_op, rec_op_dz

    loss = np.sum(loss) / len(data_loader)

    return loss

def evaluate_tau(tau_model, grid_loader):
    tau_model.eval()
    with torch.no_grad():
        xz, s, taud, taud_dx, t0, t0_dx, t0_dz, v0, nuw = next(iter(grid_loader)) #next(iter(grid_loader)) #iter(grid_loader).next()
        
        xz.requires_grad = True
        xzs = torch.hstack((xz, s.view(-1,1)))

        T = tau_model(xzs)
    return T

def evaluate_velocity(v_model, grid_loader, x_well, args):
    v_model.eval()
    
    xz, s, taud, taud_dx, t0, t0_dx, t0_dz, v0, nuw = next(iter(grid_loader)) #next(iter(grid_loader)) #iter(grid_loader).next()

    xz.requires_grad = True
    
    xzs = torch.hstack((xz, s.view(-1,1)))

    # Compute v             
    v = velocity_well(nuw*v0, v_model(xzs[:, :2]).view(-1), x_well, xzs[:,0], args)

    return v

def evaluate_nu(v_model, grid_loader):
    v_model.eval()
    
    xz, s, taud, taud_dx, t0, t0_dx, t0_dz, v0, nuw = next(iter(grid_loader)) #next(iter(grid_loader)) #iter(grid_loader).next()

    xz.requires_grad = True

    return v_model(xz[:, :2]).view(-1)

def training_loop(input_wosrc, sx, sz,
                  tau_model, v_model, v_optimizer, tau_optimizer, epochs, x_well, args,
                  batch_size=200**3,
                  vscaler= 1., device='cuda', v_scheduler=None,
                  tau_scheduler=None, fast_loader=False, wandb=False):

    # Create dataloader
    data_loader, ic, vic = create_dataloaderwell(input_wosrc, sx, sz, batch_size=batch_size, device=device, fast_loader=fast_loader)
    print('Number of points used per epoch:%d' % int(1 * len(input_wosrc[0])))

    # Initialize SaveBestModel class
    save_best_model = SaveBestModels(args['save_folder'])
    
    loss_history = []

    for i in range(epochs):

        # Train step
        loss = train(tau_model, v_model, v_optimizer, tau_optimizer, data_loader, ic, vic, x_well, args, vscaler=vscaler, device=device, wandb=wandb)
        if wandb=='y':
            wandb.log({"loss": mean_loss})
        loss_history.append(loss)

        if i % 3 == 0:
            print(f'Epoch {i}, Loss {loss:.7f}')
        
        if v_scheduler is not None:
            v_scheduler.step(loss)
        if tau_scheduler is not None:
            tau_scheduler.step(loss)
        if i%(args['num_epochs']//20)==0:
            if args['dual_optimizer']=='y':
                save_best_model(
                loss, i, tau_model, v_model, v_optimizer, nn.MSELoss(), tau_optimizer
                )
            else:
                save_best_model(
                loss, i, tau_model, v_model, v_optimizer, nn.MSELoss()
                )

    return loss_history
