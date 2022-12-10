import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load style @hatsyim
# plt.style.use("../asset/science.mplstyle")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
plt.rcParams['figure.figsize'] =  [6.4, 4.8]

def plot_contour(pred, true, init, idx, fig_name=None, save_dir='./'):
    plt.figure()
    c_p = plt.contour(pred.reshape(nz,nx,len(id_sou_x))[:,:,idx],20, 
                      colors='k',extent=(x[0], x[-1], z[0], z[-1]))
    c_t = plt.contour(true.reshape(nz,nx,len(id_sou_x))[:,:,idx], 20, 
                      colors='y', linestyles='dashed', extent=(x[0], x[-1], z[0], z[-1]))
    c_i = plt.contour(init.reshape(nz,nx,len(id_sou_x))[:,:,idx], 20, 
                      colors='b', linestyles='dashed', extent=(x[0], x[-1], z[0], z[-1]))

    h1,_ = c_p.legend_elements()
    h2,_ = c_t.legend_elements()
    h3,_ = c_i.legend_elements()
    
    plt.legend([h1[0], h2[0], h3[0]], ['Prediction', 'True', 'Initial'])
    
    plt.scatter(sx[idx], sz[idx], s=200, marker='*', color='k')
    plt.title('Traveltime Contour')
    plt.xlabel('X (km)')
    plt.ylabel('Z (km)')
    plt.axis('tight')
    
    if fig_name is not None:
        plt.savefig(os.path.join(save_dir, fig_name), 
                    format='png', bbox_inches="tight") 

def plot_section(data, fig_name, data_type='km/s', vmin=None, vmax=None, 
                 cmap='terrain', save_dir='./', aspect='equal', 
                 xmin=0, xmax=1, zmin=0, zmax=1, 
                 sx=None, sz=None, rx=None, rz=None):
    plt.figure()
    ax = plt.gca()
    im = ax.imshow(data, extent=[xmin,xmax,zmax,zmin], cmap=cmap, 
                   aspect=aspect, vmin=vmin, vmax=vmax, interpolation='kaiser')
    
    if sx is not None:
        plt.scatter(sx, sz, 5, 'white', marker='*')
    
    if rx is not None:
        plt.scatter(rx, rz, 5, 'y', marker='v')
    
    plt.xlabel('Offset (km)', fontsize=14)
    plt.xticks(fontsize=11)
    plt.ylabel('Depth (km)', fontsize=14)
    plt.yticks(fontsize=11)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="6%", pad=0.15)
    cbar = plt.colorbar(im, cax=cax)
    
    cbar.set_label(data_type,size=10)
    
    if fig_name is not None:
        plt.savefig(os.path.join(save_dir, fig_name), 
                    format='png', bbox_inches="tight") 
        
def plot_trace(init, true, pred, trace_id, fig_name=None, save_dir='./'):
    plt.figure(figsize=(3,5))
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

    ax = plt.gca()

    plt.plot(init[:,trace_id],z,'b:')
    plt.plot(true[:,trace_id],z,'k')
    plt.plot(pred[:,trace_id],z,'r--')

    ax.set_title('Velocity (km/s)', fontsize=14)
    
    plt.xticks(fontsize=11)
    plt.ylabel('Depth (km)', fontsize=14)
    plt.xlabel('Offset '+str(x[trace_id])+' (km)', fontsize=14)
    plt.yticks(fontsize=11)
    plt.gca().invert_yaxis()
    plt.legend(['Initial','True','Inverted'], fontsize=11)
    plt.grid()
    
    if fig_name is not None:
        plt.savefig(os.path.join(save_dir, fig_name), 
                    format='png', bbox_inches="tight") 
        
def plot_horizontal(trace1, trace2, x, title, ylabel, fig_name,
                    label1, label2, save_dir='./', id_rec_x=None, id_rec_z=None):
    plt.figure(figsize=(5,3))

    ax = plt.gca()

    plt.plot(x,trace1,'b')
    plt.plot(x,trace2,'r:')
    
    if id_rec_x is not None:
        plt.scatter(x[id_rec_x], trace1[id_rec_x])

    ax.set_title(title, fontsize=14)

    plt.xticks(fontsize=11)
    plt.ylabel(ylabel, fontsize=14)
    plt.xlabel('Offset (km)', fontsize=14)
    plt.yticks(fontsize=11)
    plt.gca().invert_yaxis()
    plt.legend([label1,label2], fontsize=11)
    plt.grid()

    if fig_name!=None:
        plt.savefig(os.path.join(save_dir, fig_name), 
                    format='png', bbox_inches="tight")