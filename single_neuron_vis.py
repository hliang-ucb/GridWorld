import numpy as np
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt


def get_sdf(spikes,sigma=50,step=10):
    
    # gaussian kernel
    gx = np.arange(-3*sigma, 3*sigma)
    gaussian = np.exp(-(gx/sigma)**2/2)/(sigma*np.sqrt(2*np.pi))
    sdf = convolve1d(spikes, gaussian, axis=1)*1000
    
    return sdf


# --- Sub-function 1: Raster plot ---
def plot_raster(df, var, colors, ax=None):  
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    
    unique_vals = np.unique(df[var].values)
    
    for tt in range(len(df)):
        raster = np.where(df.iloc[tt]['spikes'])[0]
        idx = np.where(unique_vals == df.iloc[tt][var])[0][0]
        ax.vlines(raster, tt, tt + 0.8, color=colors[idx], lw=0.5)
    
    ax.set_ylabel('Trial')
    ax.set_xlabel('Time')
    ax.set_title('Raster')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return ax


# --- Sub-function 2: SDF plot (mean ± SEM) ---
def plot_sdf(data, neuronID, var, colors, ax=None):

    spikes = data['spikes']
    df = data['beh'] 
    unitNames = data['unitNames']

    idx = unitNames.unitNumber==neuronID
    df['sdf']=list(get_sdf(spikes[:,:,idx]))
    
    time = np.arange(-500,500)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    
    sdf_group_mean = (
        df.groupby(var)['sdf']
          .apply(lambda x: np.mean(np.stack(x.to_numpy()), axis=0))
    )
    
    sdf_group_sem = (
        df.groupby(var)['sdf']
          .apply(lambda x: np.std(np.stack(x.to_numpy()), axis=0) / np.sqrt(len(x)))
    )

    
    for ii, val in enumerate(sdf_group_mean.index):
        mean_trace = sdf_group_mean[val].ravel()
        sem_trace = sdf_group_sem[val].ravel()
        
        ax.plot(time, mean_trace, color=colors[ii], label=f'{var}={val}')
        ax.fill_between(time,
                        mean_trace - sem_trace,
                        mean_trace + sem_trace,
                        color=colors[ii],
                        alpha=0.3)
    
    ax.set_xlabel('Action on')
    ax.set_ylabel('Firing rate')
    ax.set_title('SDF')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.legend(title=var, loc='best', fontsize=8)
    ax.axvline(0,color='k',ls='--',lw=0.5)
    
    return ax