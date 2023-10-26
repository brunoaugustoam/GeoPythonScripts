import numpy as np
from scipy.stats import mode
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import mplstereonet as mpl

def simple_stereogram(dipdir, dip,web=False):
    mode_dipdir = mode(dipdir).mode
    mode_dip = mode(dip).mode

    fig = plt.figure(figsize=(5,5))

    ax = fig.add_subplot(projection='stereonet')

    ax.pole(dipdir-90, dip, c='k',markersize=1.5)
    ax.plane(mode_dipdir-90, mode_dip, c='k')
    ax.density_contourf(dipdir-90, dip, measurement='poles', cmap='Reds')
    ax.grid()

    plt.text(1.2,1.3,'Pontos plotados: {}\nPlano Medio: {}/{}'.format(dipdir.shape[0],int(mode_dipdir),int(mode_dip)), 
     horizontalalignment='right',
     verticalalignment='top',wrap=True,
     transform = ax.transAxes)
    
    if web:
       return fig
    else:
       plt.show()

def simple_strcuture_hist(dipdir, dip):
    fig, axs = plt.subplots(1,2,figsize=(10,5))
    bins =  np.arange(dipdir.min() ,dipdir.max(),5)
    axs[0].hist(dipdir,bins=bins,density=True)
    
    bins =  np.arange(dip.min() ,dip.max(),2)
    axs[1].hist(dip,bins=bins,density=True)

    plt.show()

def planar_stereogram(df,dipdir_slope_mean,dip_slope_mean, web=False ): 

    friction_circle = np.arange(0,360,1)
    friction_mean= df.friction.mean()
    friction_mean_array = np.array([int(friction_mean)]* len(friction_circle))

    mode_dipdir = mode(df.dipdir).mode
    mode_dip = mode(df.dip).mode

    fig = plt.figure(figsize=(5,5))


    ax = fig.add_subplot(111, projection='stereonet')
    ax.plane(dipdir_slope_mean-(90-60),90, c='g')
    ax.plane(dipdir_slope_mean-(90+60),90, c='g')
    ax.plane(dipdir_slope_mean-270,90-dip_slope_mean, c='brown')
    ax.pole(friction_circle-90, friction_mean_array, c='brown', markersize=1)
    ax.pole(df.dipdir-90, df.dip, c='k',
            markersize=1.5)
    ax.density_contourf(df.dipdir-90, df.dip, measurement='poles', cmap='Reds', method ='schmidt')
    ax.grid()

    plt.text(1.2,1.3,'Planos instaveis: {}\nTotal: {}\nTalude - {}/{}\nAtrito médio: {}'.format(df.planar_rupture.sum(),len(df.planar_rupture),dipdir_slope_mean,dip_slope_mean,np.round(friction_mean,2)), 
    horizontalalignment='right',
    verticalalignment='top',wrap=True,
    transform = ax.transAxes)

    if web:
        return fig
    else:
        plt.show()



def multi_structure_stereogram(dataframe,web=False):
    '''Requires: column names to be: dipdir, dip'''
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(projection='stereonet')
    ax.density_contourf(dataframe.dipdir-90, dataframe.dip, measurement='poles', cmap='Reds')
    i=0
    for structure in dataframe.structure.unique():

        df = dataframe.query('structure == @structure')
        dipdir, dip = df.dipdir, df.dip
        mode_dipdir = mode(dipdir).mode
        mode_dip = mode(dip).mode

        ax.pole(dipdir-90, dip, c='k',markersize=1.5)
        ax.plane(mode_dipdir-90, mode_dip, c='k')
        
        plt.text(1.2,1.2+i,'Structure: {}\nPoints: {}\nMean Plane: {}/{}'.format(structure,dipdir.shape[0],int(mode_dipdir),int(mode_dip)), 
    horizontalalignment='right',
    verticalalignment='top',wrap=True,
    transform = ax.transAxes)
        i+=0.15
    ax.grid()


    if web:
       return fig
    else:
       plt.show()

def cumulative_hist(samples,web=False):
    q95 = np.round(samples.quantile([.95]),1)
    res = stats.cumfreq(samples,
                        numbins=25)
    
    x = res.lowerlimit + np.linspace(0, res.binsize*res.cumcount.size,
                                    res.cumcount.size)
    
    # specifying figure size
    fig = plt.figure(figsize=(10, 4))
    
    # adding sub plots
    ax1 = fig.add_subplot(1, 2, 1)
    
    # adding sub plots
    ax2 = fig.add_subplot(1, 2, 2)
    
    # getting histogram using hist function
    ax1.hist(samples, bins=25,
            alpha=0.7)
    
    # setting up the title
    ax1.set_title('Minumum Project Berm')
    
    # cumulative graph
    ax2.bar(x, res.cumcount, width=4, color="orange", alpha=1)

    # setting up the title
    ax2.set_title(f'Cumulative Minumum Project Berm')
    
    ax2.set_xlim([x.min(), x.max()])

    ax2.text(0.1,0.95,f'Minimum berm width to hold \nup to 95% of ruptures:{q95.item()}', 
    horizontalalignment='left',
    verticalalignment='top',wrap=True,
    transform = ax2.transAxes)
 
    if web:
        return fig
    else:
        plt.show()