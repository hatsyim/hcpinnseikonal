import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io 
import time
import random
import os
import wandb

from torch.optim.lr_scheduler import ReduceLROnPlateau
from hcpinnseikonal.utils import create_dataloader3d

# Load style @hatsyim
# plt.style.use("~/science.mplstyle")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
plt.rcParams['figure.figsize'] =  [6.4, 4.8]
           
def numerical_traveltime3d(vel, nx, ny, nz, ns, xmin, ymin, zmin, deltax, deltay, deltaz, id_sou_x, id_sou_y, id_sou_z):
    
    import pykonal

    T_data_surf = np.zeros((nz,ny,nx,ns))

    for i in range(ns):

        solver = pykonal.EikonalSolver(coord_sys="cartesian")
        solver.velocity.min_coords = zmin, ymin, xmin
        solver.velocity.node_intervals = deltaz, deltay, deltax
        solver.velocity.npts = nz, ny, nx
        solver.velocity.values = vel.reshape(nz, ny, nx)

        src_idx = id_sou_z[i], id_sou_y[i], id_sou_x[i]
        
        solver.traveltime.values[src_idx] = 0
        solver.unknown[src_idx] = False
        solver.trial.push(*src_idx)

        solver.solve()

        T_data_surf[:,:,:,i] = solver.traveltime.values
        
    return T_data_surf

# Training functions
import random
import numpy as np
from zmq import device
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def train3d(input_wosrc, sx, sy, sz,
          tau_model, v_model, optimizer, epoch,
          batch_size, vscaler, scheduler, fast_loader, device, args):
    tau_model.train()
    v_model.train()
    loss = []
    
    # Create dataloader
    weights = torch.Tensor(torch.ones(len(input_wosrc[0]))).to(device)
    data_loader, ic = create_dataloader3d(input_wosrc, sx, sy, sz, batch_size, shuffle='y', device=device, fast_loader=fast_loader, perm_id=None)
                
    # input_wsrc = [X, Y, Z, SX, SY, SZ, taud, taudx, taudy, T0, px0, py0, pz0, idx]
    sid = torch.arange(sx.size).float().to(device)
        
    for xyz, sx, sy, sz, taud, taud_dx, taud_dy, t0, t0_dx, t0_dy, t0_dz, idx in data_loader:
        
        # Number of source
        num_sou = len(ic[:,0])

        xyz.requires_grad = True

        # Input for the velocity network
        xyzic = torch.cat([xyz, ic])

        # Source location
        sxic = torch.cat([sx, ic[:,0]])
        syic = torch.cat([sy, ic[:,1]])
        szic = torch.cat([sz, ic[:,2]])
        sidx = torch.cat([idx, sid])

        # Input for the data network
        # xyzsic = torch.hstack((xyzic, sxic.view(-1,1), syic.view(-1,1), szic.view(-1,1)))
        xyzsic = torch.hstack((xyzic, sidx.view(-1,1)))

        # Compute T
        tau = tau_model(xyzsic).view(-1)

        # Compute v
        v = v_model(xyzsic[:, :3]).view(-1)

        # Gradients
        gradient = torch.autograd.grad(tau, xyzsic, torch.ones_like(tau), create_graph=True)[0]
        
        tau_dx = gradient[:, 0]
        tau_dy = gradient[:, 1]
        tau_dz = gradient[:, 2]
        
        # print(tau_dx, tau_dy, tau_dz)
    
        # Loss function based on the factored isotropic eikonal equation
        if args['exp_function']=='y':
            rec_op = (1-torch.exp((xz[:,1])**args['exp_factor']))

            # Initialize output tensor with desired value
            rec_op_dz = torch.full_like(xz[:,1], fill_value=0.)

            # Zero mask
            mask = (xz[:,1] != 0)

            rec_op_dz[mask] = (-args['exp_factor']*torch.exp((xz[:,1][mask])**args['exp_factor'])/(xz[:,1][mask]**(1-args['exp_factor'])))
        else:
            rec_op = xyz[:,2]
            rec_op_dz = 1
                
        if args['factorization_type']=='multiplicative':
            T_dx = (rec_op*tau_dx[:-num_sou] + taud_dx)*t0 + (rec_op*tau[:-num_sou] + taud)*t0_dx
            T_dz = (rec_op*tau_dz[:-num_sou] + rec_op_dz*tau[:-num_sou])*t0 + (rec_op*tau[:-num_sou] + taud)*t0_dz
        else:
            T_dx = rec_op*tau_dx[:-num_sou] + taud_dx + t0_dx
            T_dy = rec_op*tau_dy[:-num_sou] + taud_dy + t0_dy
            T_dz = rec_op*tau_dz[:-num_sou] + rec_op_dz*tau[:-num_sou] + t0_dz
        
        pde_lhs = (T_dx**2 + T_dy**2 + T_dz**2) * vscaler

        if args['velocity_loss']=='y':
            pde = torch.sqrt(1/pde_lhs) - v[:-num_sou]
        else:
            pde = pde_lhs - vscaler / (v[:-num_sou] ** 2)
        
        # No causality
        if args['causality_weight']=='type_0':
            wl2=1        
        # Stationary causality
        elif args['causality_weight']=='type_1':
            wl2 = torch.exp(-args['causality_factor']*torch.sqrt((xz[:,0]-s)**2+(xz[:,1]-z[args['zid_source']])**2))
        # Anti-causal causal non-stationary
        elif args['causality_weight']=='type_2':
            delt = (1-0.01)/args['num_epochs']
            wl2 = torch.exp((-1+delt*epoch)*args['causality_factor']*torch.sqrt((xz[:,0]-s)**2+(xz[:,1]-z[args['zid_source']])**2))
        # Anti-causal
        elif args['causality_weight']=='type_3':
            wl2 = 1.05*(1-torch.exp(-args['causality_factor']*torch.sqrt((xz[:,0]-s)**2+(xz[:,1]-z[args['zid_source']])**2)))
        # Anti-causal causal non-stationary
        elif args['causality_weight']=='type_4':
            delt = (1-0.01)/args['num_epochs']
            wl2 = torch.abs(torch.exp((-1+delt*epoch)*args['causality_factor']*torch.sqrt((xz[:,0]-s)**2+(xz[:,1]-z[args['zid_source']])**2))-1) + (1-torch.exp(torch.tensor(-5e-4*epoch)))
        
        ls_pde = torch.mean(wl2*pde**2)
        ls = ls_pde
        loss.append(ls.item())
        ls.backward()
        optimizer.step()
        optimizer.zero_grad()

        del idx, xyz, xyzic, sxic, xyzsic, taud, taud_dx, taud_dy, t0, t0_dx, t0_dy, t0_dz, ls, v, tau, tau_dx, tau_dy, tau_dz, gradient, num_sou, T_dx, T_dz, pde_lhs, ls_pde, pde, rec_op, rec_op_dz

    mean_loss = np.sum(loss) / len(data_loader)

    return mean_loss

def evaluate_tau3d(tau_model, grid_loader, num_pts, batch_size, device):
    tau_model.eval()
    
    # with torch.no_grad():
    #     T = []
    #     for xyz, sx, sy, sz, taud, taud_dx, taud_dy, t0, t0_dx, t0_dy, t0_dz, idx in grid_loader:
    #         xyz.requires_grad = True
    #         xyzs = torch.hstack((xyz, idx.view(-1,1)))    
    #         T.append(tau_model(xyzs).view(-1))
            
    with torch.no_grad():
        T = torch.empty(num_pts, device=device)
        for i, X in enumerate(grid_loader, 0):

            xyzs = torch.hstack((X[0], X[-1].view(-1,1)))
            batch_end = (i+1)*batch_size if (i+1)*batch_size<num_pts else i*batch_size + X[0].shape[0]
            T[i*batch_size:batch_end] = tau_model(xyzs).view(-1)
        
    return T

def evaluate_velocity3d(v_model, grid_loader, num_pts, batch_size, device):
    v_model.eval()
    
    # Prepare input
    with torch.no_grad():
        V = torch.empty(num_pts, device=device)
        for i, X in enumerate(grid_loader):

            # Compute v
            batch_end = (i+1)*batch_size if (i+1)*batch_size<num_pts else i*batch_size + X[0].shape[0]
            V[i*batch_size:batch_end] = v_model(X[0]).view(-1)

    return V

def training_loop3d(input_wosrc, sx, sy, sz,
                  tau_model, v_model, optimizer, epochs,
                  batch_size=200**3,
                  vscaler= 1., scheduler=None, fast_loader='n', device='cuda', wandb=None, args=None):

    loss_history = []

    for epoch in range(epochs):

        # Train step
        mean_loss = train3d(input_wosrc, sx, sy, sz,
                  tau_model, v_model, optimizer, epoch,
                  batch_size,
                  vscaler, scheduler, fast_loader, device, args)
        if wandb is not None:
            wandb.log({"loss": mean_loss})
        loss_history.append(mean_loss)

        if epoch % 3 == 0:
            print(f'Epoch {epoch}, Loss {mean_loss:.7f}')
        
        if scheduler is not None:
            scheduler.step(mean_loss)

    return loss_history
