import matplotlib.pyplot as plt
import numpy as np

def ogram(time,freq,power,ax,cmin,cmax,colormap):
    
    levels = np.linspace(cmin, cmax, 100)
    im = ax.contourf(
        time,freq,power,
        levels=levels,
        cmap=colormap,
        extend='both'
    )

    return im
    

def plot_mean_sem(timestamps,sig,ax,color,label,ls='-'):
    mean = sig.mean(axis=0)
    sem = sig.std(axis=0)/np.sqrt(sig.shape[0])
    ax.plot(timestamps,mean,color=color,label=label,ls=ls)
    ax.fill_between(timestamps,mean-sem,mean+sem,alpha=0.2,color=color)
