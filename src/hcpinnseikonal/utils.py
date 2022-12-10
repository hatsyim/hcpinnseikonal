import random
import numpy as np
import torch
import skfmm
from torch.utils.data import TensorDataset, DataLoader

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
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
            }, wandb.run.dir+'/best_model.pth')
            
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
    def __init__(self, *tensors, batch_size=200**3, shuffle='n'):
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

def create_dataloader(input_vec, sx, sz, batch_size=200**3, shuffle='y', 
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