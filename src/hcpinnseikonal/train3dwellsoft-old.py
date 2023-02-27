import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io 
import time
import random
import os
import wandb

from torch.optim.lr_scheduler import ReduceLROnPlateau
from hcpinnseikonal.utils import create_dataloader3dwell

# Load style @hatsyim
# plt.style.use("~/science.mplstyle")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
plt.rcParams['figure.figsize'] =  [6.4, 4.8]    
    
class WellOperator:
    """
    Class to perform well operator function that yields zero
    at exactly the well locations.
    """
    def __init__(
        self, X_well
    ):
        self.x_well = X_well[:,0]
        self.y_well = X_well[:,1]
        
    def __call__(
        self, x, y
    ):
        x_op, y_op = 1, 1
        for i in range(len(self.x_well)):
            x_op *= (x-self.x_well[i]) 
        for i in range(len(self.y_well)):
            y_op *= (y-self.y_well[i])
                    
        return nn.Tanh()(x_op + y_op)
    
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
          tau_model, v_model, tau_optimizer, v_optimizer, epoch,
          batch_size, vscaler, v_scheduler, tau_scheduler, fast_loader, device, args, well_operator):
    tau_model.train()
    v_model.train()
    loss = []
    loss_pde = []
    loss_data = []
    
    # Create dataloader
    weights = torch.Tensor(torch.ones(len(input_wosrc[0]))).to(device)
    data_loader, ic = create_dataloader3dwell(input_wosrc, sx, sy, sz, batch_size, shuffle='y', device=device, fast_loader=fast_loader, perm_id=None)
                
    # input_wsrc = [X, Y, Z, SX, SY, SZ, taud, taudx, taudy, T0, px0, py0, pz0, idx]
    sid = torch.arange(sx.size).float().to(device)
        
    for xyz, sx, sy, sz, taud, taud_dx, taud_dy, t0, t0_dx, t0_dy, t0_dz, v_well, idx in data_loader:
        
        sxic = sx
        syic = sy
        szic = sz
        sidx = idx

        # Input for the data network
        xyzsic = torch.hstack((xyz, sxic.view(-1,1), syic.view(-1,1), szic.view(-1,1)))
        xyzsic.requires_grad = True
        
        xyzr = torch.hstack((xyz[:,0].view(-1,1), xyz[:,1].view(-1,1), torch.zeros_like(xyz[:,2]).view(-1,1), 
                             sxic.view(-1,1), syic.view(-1,1), szic.view(-1,1)))
        xyzr.requires_grad = True
        
        # xyzsic = torch.hstack((xyz, sidx.view(-1,1)))

        # Compute T
        tau = tau_model(xyzsic).view(-1)
        tau_p = tau_model(xyzr).view(-1)

        # Compute v
        # v = v_well #+ well_operator(xyz[:,0].view(-1), xyz[:,1].view(-1)) * 
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
            T_dx = (tau_dx)*t0 + (tau)*t0_dx
            T_dy = (tau_dy)*t0 + (tau)*t0_dy
            T_dz = (tau_dz)*t0 + (tau)*t0_dz
            T_d = taud
            T_p = tau_p * t0
        else:
            T_dx = rec_op*tau_dx + taud_dx + t0_dx
            T_dy = rec_op*tau_dy + taud_dy + t0_dy
            T_dz = rec_op*tau_dz + rec_op_dz*tau + t0_dz
        
        pde_lhs = (T_dx**2 + T_dy**2 + T_dz**2) * vscaler
        
        # print(t0.min(), t0.max(), v.min(), v.max())
        
        # ls = torch.mean(abs(torch.sqrt(1/pde_lhs) - v)/torch.sqrt(1/pde_lhs)) + torch.mean(abs(T_p - T_d)/T_d)
        ls = torch.mean((pde_lhs - vscaler / (v ** 2))**2)
        
        # Add LVL term
        # ls_pde = ls_pde + torch.mean((v_well + v_model(torch.hstack((xyz[:, :2], torch.ones_like(xyz[:, 0]*0.863).view(-1,1)))).view(-1) - torch.tensor(4.023))**2)
        
        loss.append(ls.item())
        ls.backward()
        # loss_pde.append(torch.mean(abs(torch.sqrt(1/pde_lhs) - v)/torch.sqrt(1/pde_lhs)).item())
        # loss_data.append(torch.mean(abs(T_p - T_d)/T_d).item())
        v_optimizer.step()
        tau_optimizer.step()

        v_optimizer.zero_grad()
        tau_optimizer.zero_grad()

        del idx, xyz, sxic, xyzsic, taud, taud_dx, taud_dy, t0, t0_dx, t0_dy, t0_dz, ls, v, tau, tau_dx, tau_dy, tau_dz, gradient, T_dx, T_dz, pde_lhs, rec_op, rec_op_dz

    mean_ls = np.sum(loss) / len(data_loader)
    mean_ls_pde = 0 #np.sum(loss_pde) / len(data_loader)
    mean_ls_data = 0 #np.sum(loss_data) / len(data_loader)

    return [mean_ls, mean_ls_pde, mean_ls_data]

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

            xyzs = torch.hstack((X[0], X[1].view(-1,1), X[2].view(-1,1), X[3].view(-1,1)))
            # xyzs = torch.hstack((X[0], X[-1].view(-1,1)))
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
                    tau_model, v_model, tau_optimizer, v_optimizer, epochs,
                    batch_size=200**3,
                    vscaler= 1., v_scheduler=None, tau_scheduler=None, 
                    fast_loader='n', device='cuda', wandb=None, args=None,
                    well_operator=1):

    loss_history = []

    for epoch in range(epochs):

        # Train step
        mean_loss = train3d(input_wosrc, sx, sy, sz,
                            tau_model, v_model, tau_optimizer, v_optimizer, epoch,
                            batch_size,
                            vscaler, v_scheduler, tau_scheduler, fast_loader, 
                            device, args, well_operator)
        if wandb is not None:
            wandb.log({"loss": mean_loss[0]})
        loss_history.append(mean_loss)

        if epoch % 3 == 0:
            print(f'Epoch {epoch}, Total Loss {mean_loss[0]:.7f}, PDE Loss {mean_loss[1]:.7f}, Data Loss {mean_loss[2]:.7f}')
        
        if v_scheduler is not None:
            v_scheduler.step(mean_loss[0])
            tau_scheduler.step(mean_loss[0])

    return loss_history
