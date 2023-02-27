import random
import numpy as np
import torch
import torch.nn as nn
import skfmm

from torch.nn import Linear
from torch.utils.data import TensorDataset, DataLoader
            
def save_models(epochs, tau_model, nu_model, nu_optimizer, criterion, tau_optimizer=None):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    if tau_optimizer is not None:
        torch.save({
                    'epoch': epochs,
                    'tau_model_state_dict': tau_model.state_dict(),
                    'nu_model_state_dict': nu_model.state_dict(),
                    'nu_optimizer_state_dict': nu_optimizer.state_dict(),
                    'loss': criterion,
        }, wandb.run.dir+'/saved_model.pth')
    else:
        torch.save({
                    'epoch': epochs,
                    'tau_model_state_dict': tau_model.state_dict(),
                    'nu_model_state_dict': nu_model.state_dict(),
                    'nu_optimizer_state_dict': nu_optimizer.state_dict(),
                    'tau_optimizer_state_dict': tau_optimizer.state_dict(),
                    'loss': criterion,
        }, wandb.run.dir+'/saved_model.pth')

class SaveBestModels:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, save_dir, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        self.save_dir = save_dir
        
    def __call__(
        self, current_valid_loss, 
        epoch, tau_model, nu_model, nu_optimizer, criterion, tau_optimizer=None
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            if tau_optimizer is not None:
                torch.save({
                    'epoch': epoch+1,
                    'tau_model_state_dict': tau_model.state_dict(),
                    'nu_model_state_dict': nu_model.state_dict(),
                    'nu_optimizer_state_dict': nu_optimizer.state_dict(),
                    'tau_optimizer_state_dict': tau_optimizer.state_dict(),
                    'loss': criterion,
                }, self.save_dir+'/best_model.pth')
            else:
                torch.save({
                    'epoch': epoch+1,
                    'tau_model_state_dict': tau_model.state_dict(),
                    'nu_model_state_dict': nu_model.state_dict(),
                    'nu_optimizer_state_dict': nu_optimizer.state_dict(),
                    'loss': criterion,
                }, self.save_dir+'/best_model.pth')

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, save_dir, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        self.save_dir = save_dir
        
    def __call__(
        self, current_valid_loss, 
        epoch, tau_model, v_model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'tau_model_state_dict': tau_model.state_dict(),
                'v_model_state_dict': v_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
            }, self.save_dir+'/best_model.pth')
            
def save_model(epochs, tau_model, v_model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'tau_model_state_dict': tau_model.state_dict(),
                'v_model_state_dict': v_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
    }, wandb.run.dir+'/saved_model.pth')

import numpy

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 
        'hamming', 'bartlett', 'blackman'
        flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself 
    if an array instead of a string
    NOTE: length(output) != length(input), to correct this: 
    return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y[(window_len//2-1):-(window_len//2)]

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=200**3, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

def torch_to_numpy(x, nx=None, nz=None, ns=None):
    
    if (nx is not None) & (nz is not None) & (ns is not None):
        return x.detach().cpu().numpy().reshape(nz,nx,ns)
    else:
        return x.detach().cpu().numpy()

def create_dataloader2d(input_vec, sx, sz, batch_size=200**3, shuffle=True, 
                      device='cuda', fast_loader='n', perm_id=None):
    
    XZ = torch.from_numpy(np.vstack((input_vec[0], input_vec[1])).T).float().to(device)
    S = torch.from_numpy(input_vec[2]).float().to(device)
    
    taud = torch.from_numpy(input_vec[3]).float().to(device)
    taud_dx = torch.from_numpy(input_vec[4]).float().to(device)

    tana = torch.from_numpy(input_vec[5]).float().to(device)
    tana_dx = torch.from_numpy(input_vec[6]).float().to(device)
    tana_dz = torch.from_numpy(input_vec[7]).float().to(device)
    
    index = torch.arange(input_vec[0].size)
    
    if perm_id is not None:
        dataset = TensorDataset(XZ[perm_id], S[perm_id], taud[perm_id], 
                                taud_dx[perm_id], tana[perm_id], tana_dx[perm_id], 
                                tana_dz[perm_id], index[perm_id])
    else:
        dataset = TensorDataset(XZ, S, taud, taud_dx, tana, tana_dx, tana_dz, index)
    
    if fast_loader:
        data_loader = FastTensorDataLoader(XZ, S, taud, taud_dx, tana, 
                                           tana_dx, tana_dz, index, 
                                           batch_size=batch_size, shuffle=shuffle)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # initial condition
    ic = torch.tensor(np.array([sx, sz]), dtype=torch.float).to(device)

    return data_loader, ic.T


def create_dataloader3d(input_vec, sx, sy, sz, batch_size=200**4, shuffle=True, 
                      device='cuda', fast_loader='n', perm_id=None):
    
    # input_wsrc = [X, Y, Z, SX, SY, SZ, taud, taudx, taudy, T0, px0, py0, pz0, index]
    
    XYZ = torch.from_numpy(np.vstack((input_vec[0], input_vec[1], input_vec[2])).T).float().to(device)
    SX = torch.from_numpy(input_vec[3]).float().to(device)
    SY = torch.from_numpy(input_vec[4]).float().to(device)
    SZ = torch.from_numpy(input_vec[5]).float().to(device)
    
    taud = torch.from_numpy(input_vec[6]).float().to(device)
    taud_dx = torch.from_numpy(input_vec[7]).float().to(device)
    taud_dy = torch.from_numpy(input_vec[8]).float().to(device)

    tana = torch.from_numpy(input_vec[9]).float().to(device)
    tana_dx = torch.from_numpy(input_vec[10]).float().to(device)
    tana_dy = torch.from_numpy(input_vec[11]).float().to(device)
    tana_dz = torch.from_numpy(input_vec[12]).float().to(device)
    
    index = torch.from_numpy(input_vec[13]).float().to(device)
    
    if perm_id is not None:
        dataset = TensorDataset(XYZ[perm_id], SX[perm_id], SY[perm_id], SZ[perm_id], taud[perm_id], 
                                taud_dx[perm_id], taud_dy[perm_id], 
                                tana[perm_id], tana_dx[perm_id], 
                                tana_dy[perm_id], tana_dz[perm_id], index[perm_id])
    else:
        dataset = TensorDataset(XYZ, SX, SY, SZ, taud, taud_dx, taud_dy, tana, tana_dx, tana_dy, tana_dz, index)
    
    if fast_loader:
        data_loader = FastTensorDataLoader(XYZ, SX, SY, SZ, taud, taud_dx, 
                                           taud_dy, tana, tana_dx, tana_dy, tana_dz, index, 
                                           batch_size=batch_size, shuffle=shuffle)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # initial condition
    ic = torch.tensor(np.array([sx, sy, sz]), dtype=torch.float).to(device)

    return data_loader, ic.T

def create_dataloader3dmodelingOld(input_vec, sx, sy, sz, batch_size=200**4, shuffle=True, 
                      device='cuda', fast_loader='n', perm_id=None):
    
    # input_wsrc = [X, Y, Z, SX+len(id_sou), SY+len(id_sou), SZ+len(id_sou), T0, px0, py0, pz0, index]
    
    XYZ = torch.from_numpy(np.vstack((input_vec[0], input_vec[1], input_vec[2])).T).float().to(device)
    SX = torch.from_numpy(input_vec[3]).float().to(device)
    SY = torch.from_numpy(input_vec[4]).float().to(device)
    SZ = torch.from_numpy(input_vec[5]).float().to(device)
    
    tana = torch.from_numpy(input_vec[6]).float().to(device)
    tana_dx = torch.from_numpy(input_vec[7]).float().to(device)
    tana_dy = torch.from_numpy(input_vec[8]).float().to(device)
    tana_dz = torch.from_numpy(input_vec[9]).float().to(device)
    
    index = torch.from_numpy(input_vec[10]).float().to(device)
    
    if perm_id is not None:
        dataset = TensorDataset(XYZ[perm_id], SX[perm_id], SY[perm_id], SZ[perm_id],
                                tana[perm_id], tana_dx[perm_id], 
                                tana_dy[perm_id], tana_dz[perm_id], index[perm_id])
    else:
        dataset = TensorDataset(XYZ, SX, SY, SZ, tana, tana_dx, tana_dy, tana_dz, index)
    
    if fast_loader:
        data_loader = FastTensorDataLoader(XYZ, SX, SY, SZ, 
                                           tana, tana_dx, tana_dy, tana_dz, index, 
                                           batch_size=batch_size, shuffle=shuffle)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # initial condition
    ic = torch.tensor(np.array([sx, sy, sz]), dtype=torch.float).to(device)

    return data_loader, ic.T

def create_dataloader3dmodeling(input_vec, sx, sy, sz, batch_size=200**4, shuffle=True, 
                      device='cuda', fast_loader='n', perm_id=None):
    
    # input_wsrc = [X, Y, Z, SX+len(id_sou), SY+len(id_sou), SZ+len(id_sou), T0, px0, py0, pz0, index]
    
    XYZ = torch.from_numpy(np.vstack(list(input_vec[:3])).T).float().to(device)
    SX = torch.from_numpy(input_vec[3]).float().to(device)
    SY = torch.from_numpy(input_vec[4]).float().to(device)
    SZ = torch.from_numpy(input_vec[5]).float().to(device)
    
    tana = torch.from_numpy(input_vec[6]).float().to(device)
    tana_dx = torch.from_numpy(input_vec[7]).float().to(device)
    tana_dy = torch.from_numpy(input_vec[8]).float().to(device)
    tana_dz = torch.from_numpy(input_vec[9]).float().to(device)
    
    v = torch.from_numpy(input_vec[10]).float().to(device)
    # z = torch.from_numpy(np.vstack(list(input_vec[11:])).T).float().to(device)
    z = torch.from_numpy(input_vec[11]).float().to(device)
    
    if perm_id is not None:
        dataset = TensorDataset(XYZ[perm_id], SX[perm_id], SY[perm_id], SZ[perm_id],
                                tana[perm_id], tana_dx[perm_id], 
                                tana_dy[perm_id], tana_dz[perm_id], 
                                v[perm_id], z[perm_id])
    else:
        dataset = TensorDataset(XYZ, SX, SY, SZ, tana, tana_dx, tana_dy, tana_dz, v, z)
    
    if fast_loader:
        data_loader = FastTensorDataLoader(XYZ, SX, SY, SZ, 
                                           tana, tana_dx, tana_dy, tana_dz, v, z, 
                                           batch_size=batch_size, shuffle=shuffle)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # initial condition
    ic = torch.tensor(np.array([sx, sy, sz]), dtype=torch.float).to(device)

    return data_loader, ic.T

def create_dataloaderdd(input_vec, sx1, sz1, sx2, sz2, 
                        batch_size=200**3, shuffle=True, 
                        device='cuda', fast_loader='n', perm_id=None):
    
    XZ = torch.from_numpy(np.vstack((input_vec[0], input_vec[1])).T).float().to(device)
    SX1 = torch.from_numpy(input_vec[2]).float().to(device)
    SX2 = torch.from_numpy(input_vec[3]).float().to(device)
    
    taud1 = torch.from_numpy(input_vec[4]).float().to(device)
    taud_dx1 = torch.from_numpy(input_vec[5]).float().to(device)
    
    taud2 = torch.from_numpy(input_vec[6]).float().to(device)
    taud_dx2 = torch.from_numpy(input_vec[7]).float().to(device)

    tana1 = torch.from_numpy(input_vec[8]).float().to(device)
    tana_dx1 = torch.from_numpy(input_vec[9]).float().to(device)
    tana_dz1 = torch.from_numpy(input_vec[10]).float().to(device)
    
    tana2 = torch.from_numpy(input_vec[11]).float().to(device)
    tana_dx2 = torch.from_numpy(input_vec[12]).float().to(device)
    tana_dz2 = torch.from_numpy(input_vec[13]).float().to(device)
    
    index = torch.arange(input_vec[0].size)
    
    if perm_id is not None:
        dataset = TensorDataset(XZ[perm_id], SX1[perm_id], SX2[perm_id], 
                                taud1[perm_id], taud_dx1[perm_id], 
                                taud2[perm_id], taud_dx2[perm_id], 
                                tana1[perm_id], tana_dx1[perm_id], tana_dz1[perm_id],
                                tana2[perm_id], tana_dx2[perm_id], tana_dz2[perm_id], index[perm_id])
    else:
        dataset = TensorDataset(XZ, SX1, SX2, taud1, taud_dx1, taud2, taud_dx2, 
                                tana1, tana_dx1, tana_dz1, 
                                tana2, tana_dx2, tana_dz2, index)
    
    if fast_loader:
        data_loader = FastTensorDataLoader(XZ, SX1, SX2, taud1, taud_dx1, taud2, taud_dx2, 
                                           tana1, tana_dx1, tana_dz1, 
                                           tana2, tana_dx2, tana_dz2, index, 
                                           batch_size=batch_size, shuffle=shuffle)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # initial condition
    ic1 = torch.tensor(np.array([sx1, sz1]), dtype=torch.float).to(device)
    ic2 = torch.tensor(np.array([sx2, sz2]), dtype=torch.float).to(device)

    return data_loader, ic1.T, ic2.T

def create_dataloader2dwell(input_vec, sx, sz, batch_size=200**3, shuffle=True, 
                      device='cuda', fast_loader=False, perm_id=None):
    
    # input_wsrc = [X, Z, SX, taud, taudx, T0, px0, pz0]
    # input_wosrc = [i.ravel()[isource.reshape(-1)][perm_id] for i in input_wsrc]
    
    XZ = torch.from_numpy(np.vstack((input_vec[0], input_vec[1])).T).float().to(device)
    S = torch.from_numpy(input_vec[2]).float().to(device)
    
    taud = torch.from_numpy(input_vec[3]).float().to(device)
    taud_dx = torch.from_numpy(input_vec[4]).float().to(device)

    tana = torch.from_numpy(input_vec[5]).float().to(device)
    tana_dx = torch.from_numpy(input_vec[6]).float().to(device)
    tana_dz = torch.from_numpy(input_vec[7]).float().to(device)
    
    v0 = torch.from_numpy(input_vec[8]).float().to(device)
    nuw = torch.from_numpy(input_vec[9]).float().to(device)
    
    v0ic = torch.from_numpy(input_vec[10]).float().to(device)
    nuwic = torch.from_numpy(input_vec[11]).float().to(device)
    
    if perm_id is not None:
        dataset = TensorDataset(XZ[perm_id], S[perm_id], taud[perm_id], taud_dx[perm_id], 
                                tana[perm_id], tana_dx[perm_id], tana_dz[perm_id], 
                                v0[perm_id], nuw[perm_id])
    else:
        dataset = TensorDataset(XZ, S, taud, taud_dx, tana, tana_dx, tana_dz, v0, nuw)
    
    if fast_loader:
        data_loader = FastTensorDataLoader(XZ, S, taud, taud_dx, tana, tana_dx, tana_dz, 
                                           v0, nuw, batch_size=batch_size, shuffle=shuffle)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # Initial condition
    ic = torch.tensor(np.array([sx, sz]), dtype=torch.float).to(device)
    vic = torch.hstack([v0ic, nuwic]).to(device)

    return data_loader, ic.T, vic

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_device():

    device = 'cpu'
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        print("Cuda installed! Running on GPU!")
        device = torch.device(torch.cuda.current_device())
        print(f'Device: {device} {torch.cuda.get_device_name(device)}')
    else:
        print("No GPU available!")
    return device

def init_weights(m,init_type='xavierUniform', bias=0.1, mean=0., std=1.,):
    
    if isinstance(m, nn.Linear):
        if init_type=='xavierUniform':
            torch.nn.init.xavier_uniform_(m.weight)
        elif init_type=='xavierNormal':
            torch.nn.init.xavier_normal_(m.weight)
        elif init_type=='kaimingUniform':
            torch.nn.init.kaiming_uniform_(m.weight)
        elif init_type=='normal':
            m.weight.data.normal_(mean, std)
        elif init_type=='uniform':
            m.weight.data.uniform_(-2.0, 15.0)
        m.bias.data.fill_(bias)
        
# Multi-GPU setup
import torch, os, logging, sys
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningModule, Trainer, LightningDataModule

os.environ["MKL_THREADING_LAYER"] = "GNU" # fixes https://github.com/pytorch/pytorch/issues/37377?

divider_str="-"*40

def get_env_display_text(var_name):
    var_value = os.environ.get(var_name, "")
    return f"{var_name} = {var_value}"

def display_environment():
    """
    Print a few environment variables of note
    """
    variable_names = [
        "MASTER_ADDR",
        "MASTER_PORT",
        "NODE_RANK",
        "LOCAL_RANK",
        "GLOBAL_RANK",
        "WORLD_SIZE",
        "NCCL_SOCKET_IFNAME",
        "OMPI_COMM_WORLD_RANK",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "OMPI_COMM_WORLD_SIZE",
        "OMPI_COMM_WORLD_LOCAL_SIZE"
    ]
    var_text = "\n".join([get_env_display_text(var) for var in variable_names])
    print(f"\nEnvironmental variables:\n{divider_str}\n{var_text}\n{divider_str}\n")
    
def create_dataloader3dwell(input_vec, sx, sy, sz, batch_size=200**4, shuffle=True, 
                      device='cuda', fast_loader='n', perm_id=None):
    
    # input_wsrc = [X, Y, Z, SX, SY, SZ, taud, taudx, taudy, T0, px0, py0, pz0, index]
    
    XYZ = torch.from_numpy(np.vstack((input_vec[0], input_vec[1], input_vec[2])).T).float().to(device)
    SX = torch.from_numpy(input_vec[3]).float().to(device)
    SY = torch.from_numpy(input_vec[4]).float().to(device)
    SZ = torch.from_numpy(input_vec[5]).float().to(device)
    
    taud = torch.from_numpy(input_vec[6]).float().to(device)
    taud_dx = torch.from_numpy(input_vec[7]).float().to(device)
    taud_dy = torch.from_numpy(input_vec[8]).float().to(device)

    tana = torch.from_numpy(input_vec[9]).float().to(device)
    tana_dx = torch.from_numpy(input_vec[10]).float().to(device)
    tana_dy = torch.from_numpy(input_vec[11]).float().to(device)
    tana_dz = torch.from_numpy(input_vec[12]).float().to(device)
    
    vw = torch.from_numpy(input_vec[13]).float().to(device)
    
    index = torch.from_numpy(input_vec[14]).float().to(device)
    
    if perm_id is not None:
        dataset = TensorDataset(XYZ[perm_id], SX[perm_id], SY[perm_id], SZ[perm_id], taud[perm_id], 
                                taud_dx[perm_id], taud_dy[perm_id], 
                                tana[perm_id], tana_dx[perm_id], 
                                tana_dy[perm_id], tana_dz[perm_id], vw[perm_id], index[perm_id])
    else:
        dataset = TensorDataset(XYZ, SX, SY, SZ, taud, taud_dx, taud_dy, tana, tana_dx, tana_dy, tana_dz, vw, index)
    
    if fast_loader:
        data_loader = FastTensorDataLoader(XYZ, SX, SY, SZ, taud, taud_dx, 
                                           taud_dy, tana, tana_dx, tana_dy, tana_dz, vw, index, 
                                           batch_size=batch_size, shuffle=shuffle)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # initial condition
    ic = torch.tensor(np.array([sx, sy, sz]), dtype=torch.float).to(device)

    return data_loader, ic.T

from pykrige.ok import OrdinaryKriging

def perform_kriging(v, x, y, well_interval):

    vel_smooth = np.ones_like(v)

    for i in range(v.shape[0]):

        V_well = v[i,::well_interval,::well_interval]
        X_well = x[i,::well_interval,::well_interval]
        Y_well = y[i,::well_interval,::well_interval]

        D_well = np.vstack((Y_well.reshape(-1), X_well.reshape(-1), V_well.reshape(-1))).T

        OK = OrdinaryKriging(
            D_well[:, 0], D_well[:, 1], D_well[:, 2],
            variogram_model="linear",
            verbose=False,
            enable_plotting=False,
        )

        V, SS = OK.execute("grid", y[0,:,0], x[0,0,:])

        vel_smooth[i,:,:] = V
        
    return vel_smooth