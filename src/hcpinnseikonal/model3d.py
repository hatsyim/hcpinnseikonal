import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io 
import time
import random
import os
import wandb

from torch.optim.lr_scheduler import ReduceLROnPlateau
from hcpinnseikonal.utils import create_dataloader3dmodeling

# Load style @hatsyim
# plt.style.use("~/science.mplstyle")
           
# Training functions
import random
import numpy as np
from zmq import device
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def numerical_traveltime3d(vel, nx, ny, nz, ns, xmin, ymin, zmin, 
                           deltax, deltay, deltaz, 
                           id_sou_x, id_sou_y, id_sou_z):
    
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

def train3d(input_wosrc, sx, sy, sz,
          tau_model, optimizer, grid_size, num_pts,
          batch_size, vscaler, scheduler, fast_loader, device, args):
    tau_model.train()
    loss = []
    
    # Whether a full or subgrid size is used
    # if grid_size==num_pts:
    #     ipermute = torch.randperm(grid_size)[:num_pts]
    # else:
    #     ipermute = None

    data_loader, ic = create_dataloader3dmodeling(
        input_wosrc, sx, sy, sz, batch_size, 
        shuffle=True, device=device, 
        fast_loader=fast_loader, perm_id=None
    )
        
    for xyz, sx, sy, sz, t0, t0_dx, t0_dy, t0_dz, v, z in data_loader:
        
        # xyz.requires_grad = True

        sxic = sx
        syic = sy
        szic = sz
        
        # Input for the data network
        xyzsic = torch.hstack((
            xyz, 
            sxic.view(-1,1), syic.view(-1,1), szic.view(-1,1), 
            z.view(-1,96)
        ))
        xyzsic.requires_grad=True
        
        # Compute T
        tau = tau_model(xyzsic).view(-1)

        # Gradients
        gradient = torch.autograd.grad(tau, xyzsic, torch.ones_like(tau), create_graph=True)[0]
        
        tau_dx = gradient[:, 0]
        tau_dy = gradient[:, 1]
        tau_dz = gradient[:, 2]
        
        T_dx = tau_dx*t0 + tau*t0_dx
        T_dy = tau_dy*t0 + tau*t0_dy
        T_dz = tau_dz*t0 + tau*t0_dz
        
        pde_lhs = (T_dx**2 + T_dy**2 + T_dz**2)
        pde = torch.mean(abs(torch.sqrt(1/pde_lhs) - v)/torch.sqrt(1/pde_lhs))   
#         pde = pde_lhs - 1 / (v ** 2)

        # # pde loss
        # pde1 = (t0 ** 2) * (tau_dx ** 2 + tau_dy ** 2 + tau_dz ** 2)
        # pde2 = (tau ** 2) * (t0_dx ** 2 + t0_dy ** 2 + t0_dz ** 2)
        # pde3 = 2 * t0 * tau * (tau_dx * t0_dx + tau_dy * t0_dy + tau_dz * t0_dz)
        # pde = (pde1 + pde2 + pde3) * 1 - 1 / (v ** 2)
        # pde = torch.mean(pde ** 2)

        # Initial condition loss
        # bc = torch.mean((tau_model(xyzs).view(-1) - 1) ** 2)
        # bc = abs((tau_model(sic).view(-1)-1)/tau_model(sic).view(-1))
        
        ls = pde #+ bc
        loss.append(ls.item())
        ls.backward()
        optimizer.step()
        optimizer.zero_grad()

        del xyz, sxic, xyzsic, t0, t0_dx, t0_dy, t0_dz, ls, v, tau, tau_dx, tau_dy, tau_dz, gradient, pde

    mean_loss = np.sum(loss) / len(data_loader)
    
    del data_loader, ic

    return mean_loss

def evaluate_tau3d(tau_model, latent, grid_loader, num_pts, batch_size, device):
    tau_model.eval()
    
    with torch.no_grad():
        T = torch.empty(num_pts, device=device)
        for i, X in enumerate(grid_loader, 0):
            
            batch_end = (i+1)*batch_size if (i+1)*batch_size<num_pts else i*batch_size + X[0].shape[0]

            # Latent
            z = X[-1]

            xyzs = torch.hstack((X[0], X[1].view(-1,1), X[2].view(-1,1), X[3].view(-1,1), z))
            # xyzs = torch.hstack((X[0], X[1].view(-1,1), X[2].view(-1,1), X[3].view(-1,1)))

            # Compute T
            T[i*batch_size:batch_end] = tau_model(xyzs).view(-1)
        
    return T

def evaluate_velocity3d(tau_model, latent, grid_loader, num_pts, batch_size, device):

    tau_model.eval()
    
    # Prepare input
    # with torch.no_grad():
    V = torch.empty(num_pts, device=device)
    for i, X in enumerate(grid_loader):

        # Compute v
        batch_end = (i+1)*batch_size if (i+1)*batch_size<num_pts else i*batch_size + X[0].shape[0]
        
        # Latent
        z = X[-1]
        
        xyzs = torch.hstack((X[0], X[1].view(-1,1), X[2].view(-1,1), X[3].view(-1,1), z))
        # xyzs = torch.hstack((X[0], X[1].view(-1,1), X[2].view(-1,1), X[3].view(-1,1)))
        
        xyzs.requires_grad=True

        # Compute T
        tau = tau_model(xyzs).view(-1)

        # Gradients
        gradient = torch.autograd.grad(tau, xyzs, torch.ones_like(tau), create_graph=True)[0]

        tau_dx = gradient[:, 0]
        tau_dy = gradient[:, 1]
        tau_dz = gradient[:, 2]

        T_dx = tau_dx*X[4] + tau*X[5]
        T_dy = tau_dy*X[4] + tau*X[6]
        T_dz = tau_dz*X[4] + tau*X[7]

        pde_lhs = (T_dx**2 + T_dy**2 + T_dz**2)
        
        print(i*batch_size, batch_end, pde_lhs.shape)
        
        V[i*batch_size:batch_end] = torch.sqrt(1/pde_lhs)
            
    return V

def training_loop3d(input_wosrc, sx, sy, sz,
                    tau_model, optimizer, grid_size, num_pts, 
                    out_epochs, in_epochs=100,
                    batch_size=200**3,
                    vscaler= 1., scheduler=None, fast_loader='n', 
                    device='cuda', wandb=None, args=None):

    loss_history = []
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, 
        patience=args['reduce_after'], verbose=True
    )

    for epoch in range(out_epochs):

        # Train step
        mean_loss = train3d(input_wosrc, sx, sy, sz,
                  tau_model, optimizer, grid_size, num_pts,
                  batch_size,
                  vscaler, scheduler, fast_loader, device, args)
        if wandb is not None:
            wandb.log({"loss": mean_loss})
        loss_history.append(mean_loss)

        if epoch % 250 == 0:
            print(f'Epoch {epoch}, Loss {mean_loss:.7f}')

        if scheduler is not None:
            scheduler.step(mean_loss)
            
        del mean_loss

    return loss_history
