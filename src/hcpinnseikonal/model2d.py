import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io 
import time
import random
import os
import wandb

from torch.optim.lr_scheduler import ReduceLROnPlateau
from hcpinnseikonal.utils import create_dataloader2d

# Load style @hatsyim
# plt.style.use("~/science.mplstyle")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
plt.rcParams['figure.figsize'] =  [6.4, 4.8]
           
def numerical_traveltime2d(vel, nx, nz, ns, xmin, zmin, deltax, deltaz, id_sou_x, id_sou_z):
    
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

def train(input_wosrc, sx, sz,
          tau_model, v_model, optimizer, epoch,
          batch_size, vscaler, scheduler, fast_loader, device, args):
    tau_model.train()
    v_model.train()
    loss = []
    
    # Create dataloader
    weights = torch.Tensor(torch.ones(len(input_wosrc[0]))).to(device)
    data_loader, ic = create_dataloader2d(input_wosrc, sx, sz, batch_size, shuffle='y', device=device, fast_loader=fast_loader, perm_id=None)
        
    for xz, s, taud, taud_dx, t0, t0_dx, t0_dz, index in data_loader:
        
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
        v = v_model(xzsic[:, :2]).view(-1)

        # Gradients
        gradient = torch.autograd.grad(tau, xzsic, torch.ones_like(tau), create_graph=True)[0]
        
        tau_dx = gradient[:, 0]
        tau_dz = gradient[:, 1]
    
        # Loss function based on the factored isotropic eikonal equation
        if args['exp_function']=='y':
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
        
        if args['regularization_type']=='None':
            ls_pde = torch.mean(wl2*pde**2)
        elif args['regularization_type']=='isotropic-TV':
            dv_d1 = torch.autograd.grad(v, xzsic, torch.ones_like(v), create_graph=True)[0]
            # dv_d2 = torch.autograd.grad(dv_d1, xzsic, torch.ones_like(v), create_graph=True)[0]
            L1 = nn.L1Loss(reduction='sum')
            ls_pde = torch.mean(wl2*pde**2) + args['regularization_weight'] * torch.mean(torch.abs(dv_d1[:,0]) + torch.abs(dv_d1[:,1]))
        
        ls = ls_pde
        loss.append(ls.item())
        ls.backward()
        optimizer.step()
        optimizer.zero_grad()
        weights[index] = ls

        del index, xz, xzic, sic, xzsic, taud, taud_dx, t0, t0_dx, t0_dz, ls, v, tau, tau_dx, tau_dz, gradient, num_sou, T_dx, T_dz, pde_lhs, ls_pde, pde, rec_op, rec_op_dz

    mean_loss = np.sum(loss) / len(data_loader)

    return mean_loss

def evaluate_tau(tau_model, grid_loader):
    tau_model.eval()
    
    with torch.no_grad():
        xz, s, taud, taud_dx, t0, t0_dx, t0_dz, _ = next(iter(grid_loader))
        xz.requires_grad = True
        xzs = torch.hstack((xz, s.view(-1,1)))
        T = tau_model(xzs)
        
    return T

def evaluate_velocity(v_model, grid_loader):
    v_model.eval()
    
    # Prepare input
    xz, s, taud, taud_dx, t0, t0_dx, t0_dz, _ = next(iter(grid_loader))
    xz.requires_grad = True
    xzs = torch.hstack((xz, s.view(-1,1)))

    # Compute v
    v = v_model(xzs[:,:2]).view(-1)

    return v

def training_loop(input_wosrc, sx, sz,
                  tau_model, v_model, optimizer, epochs,
                  batch_size=200**3,
                  vscaler= 1., scheduler=None, fast_loader='n', device='cuda', wandb=None, args=None):

    loss_history = []

    for epoch in range(epochs):

        # Train step
        mean_loss = train(input_wosrc, sx, sz,
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
