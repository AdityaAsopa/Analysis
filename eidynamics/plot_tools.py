import sys
import os
import datetime
import importlib
import pathlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.signal import filter_design
from scipy.signal import butter, bessel, decimate, sosfiltfilt
from scipy.signal import find_peaks, peak_widths
from scipy import stats

from eidynamics import utils
from eidynamics import pattern_index



def simplify_axes(axes, remove_ticks=True, xticks=[], yticks=[], exclude_from_simplification=['bottom', 'left']):
    '''simplify axis properties to remove clutter like ticks, ticklabels, spines, etc.'''
    # check if ax is a list of axes or a numpy array
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    for ax in axes:
        for side in ['bottom', 'left', 'top', 'right']:
            if side not in exclude_from_simplification:
                ax.spines[side].set_visible(False)
            else:    
                ax.spines[side].set_linewidth(0.5)

        if 'bottom' not in exclude_from_simplification:
            ax.tick_params(axis='x', which='both', length=0)
            ax.get_xaxis().tick_bottom()
            ax.set_xticks([])
         
        # ax.tick_params(axis='both', which='both', length=0)
        if remove_ticks:
            ax.set_yticks([])
        else:
            ax.set_yticks(yticks)
            ax.set_xticks(xticks)
        
        ax.set_title('')
    return axes


def add_floating_scalebar(ax, scalebar_origin=[0,0], xlength=1.0, ylength=1.0, labelx='', labely='', unitx='', unity='', fontsize=12, color='black', linewidth=2, pad=0.1, simplify=True, exclude_axis_from_simplification=[], show_labels=False):
    """Simplifies a matplotlib axes object and adds a floating scalebar.
    Args:
        ax: matplotlib axes object
        x: x position of the scalebar in data coordinates
        y: y position of the scalebar in data coordinates
        xl: length of the x-axis of scalebar in data coordinates
        yl: length of the y-axis of scalebar in data coordinates
        labelx: label of the a-axis of scalebar
        labely: label of the y-axis of scalebar
        unitx: units of the x-axis of scalebar
        unity: units of the y-axis of scalebar
        fontsize: fontsize of the label
        color: color of the scalebar
        linewidth: linewidth of the scalebar
        pad: padding between the scalebar and the label
    Returns:
        None
    Example:
        add_floating_scalebar(ax, 1, 1, 0.1, 0.2, '0.1', '0.2', 's', 'mV')

    """
    x,y = scalebar_origin
    xl = xlength
    yl = ylength

    if simplify:
        simplify_axes(ax, exclude_from_simplification=exclude_axis_from_simplification)

    # draw a line for x axis
    ax.plot([x, x+xl], [y, y]   , color=color, linewidth=linewidth)

    
    # draw a line for y axis
    ax.plot([x, x],    [y, y+yl], color=color, linewidth=linewidth)
    
    if show_labels:    
        # write x axis label
        ax.text(x+xl/2, y-2*pad, labelx+' '+unitx, fontsize=fontsize, horizontalalignment='center', verticalalignment='top')
        # write y axis label
        ax.text(x-pad, y+yl/2, labely+' '+unity, fontsize=fontsize, horizontalalignment='right', verticalalignment='center', rotation=90) # alignment of the rotated text as a block
    

def plot_abf_data(dataDict, label=""):
    numChannels = len(dataDict[0])
    chLabels    = list(dataDict[0].keys())
    sweepLength = len(dataDict[0][chLabels[0]])

    if 'Time' in chLabels:    
        timeSignal = dataDict[0]['Time']
        chLabels.remove('Time')
    else:
        timeSignal = np.arange(0,sweepLength/2e4,1/2e4)
    
    numPlots = len(chLabels)
    fig,axs     = plt.subplots(numPlots,1,sharex=True)
    
    for sweepData in dataDict.values():
        for i,ch in enumerate(chLabels):
            if ch == 'Cmd':
                axs[i].plot(timeSignal[::5],sweepData[ch][::5],'r')
                axs[i].set_ylabel('Ch#0 Command')
            else:
                axs[i].plot(timeSignal[::5],sweepData[ch][::5],'b')
                axs[i].set_ylabel('Ch# '+str(ch))

    axs[-1].set_xlabel('Time (s)')
    axs[-1].annotate('* Data undersampled for plotting', xy=(1.0, -0.5), xycoords='axes fraction',ha='right',va="center",fontsize=6)
    fig.suptitle(label + ' - ABF Data*')
    plt.show()


def plot_data_from_df(df, data_start_column = 35 , simplify=False, combine=False, fig=None, ax=None, exclude_from_simplification=[]):
    start = data_start_column
    Fs = 2e4
    sweeps = df.shape[0]
    width = int( (df.shape[1] - start)/4 )
    T = width/Fs

    # if combine plots is false, draw all the 4 signals separately on 4 subplots
    if combine is False:
        print('Plotting all 4 signals separately')

        # check if fig and ax are supplied:
        if fig is None:
            fig = plt.figure(layout='constrained', figsize=(10, 4))
        else:
            gridspec = ax.get_subplotspec().get_gridspec()
            ax.remove()
            subfig = fig.add_subfigure(gridspec[:, 0])
        subfigs = fig.subfigures(4,1)

        subfig_axs0 = subfigs[0].subplots(1,1, sharey=True)
        subfig_axs1 = subfigs[1].subplots(1,1, sharey=True)
        subfig_axs2 = subfigs[2].subplots(1,1, sharey=True)
        subfig_axs3 = subfigs[3].subplots(1,1, sharey=True)

        axs = [subfig_axs0, subfig_axs1, subfig_axs2, subfig_axs3]
        if simplify:
            simplify_axes(axs, exclude_from_simplification=[])

        time = np.linspace(0, T, num=width, endpoint=False)
        # copy time vector as many times as there are sweeps
        Time = np.tile(time, (sweeps,1) )


        for i in range(sweeps):
            start = 35 
            trace = df.iloc[i, slice(start, start+width)]
            trace = utils.map_range(trace, 0, 5, 0,5)
            axs[0].plot(time, trace, 'black', linewidth=1, alpha=0.1)
            axs[0].set_ylabel('Cell')

            start += width
            trace = df.iloc[i, slice(start, start+width)]
            trace = utils.map_range(trace, 0, 5, 0,5)
            axs[1].plot(time, trace, 'red', linewidth=1, alpha=0.1)
            axs[1].set_ylabel('FrameTTL')

            start += width
            trace = df.iloc[i, slice(start, start+width)]
            trace = utils.map_range(trace, 0, 5, 0,5)
            axs[2].plot(time, trace, 'cyan', linewidth=1, alpha=0.1)
            axs[2].set_ylabel('PD')

            start += width
            trace = df.iloc[i, slice(start, start+width)]
            trace = utils.map_range(trace, 0, 5, 0,5)
            axs[3].plot(time, trace, 'orange', linewidth=1, alpha=0.1)
            axs[3].set_ylabel('Field')
        
        # plot average sweeps on respective axes
        axs[0].plot(time, df.iloc[:,    35:20035].mean(axis=0), color='black', linewidth=1, label='Cell')
        axs[1].plot(time, df.iloc[:, 20035:40035].mean(axis=0), color='red', linewidth=1, label='FrameTTL')
        axs[2].plot(time, df.iloc[:,40035:60035].mean(axis=0), color='cyan', linewidth=1, label='PD')
        axs[3].plot(time, df.iloc[:,60035:80035].mean(axis=0), color='orange', linewidth=1, label='Field')
  
        axs[3].set_xlabel('Time (s)')

        return fig, axs
    
    # if combine plots is true, draw all the 4 signals on a single plot
    elif combine is True:
        print('Plotting all 4 signals on a single plot')

        # check if ax is supplied
        if fig is None and ax is None:
            fig, ax = plt.subplots(1,1, figsize=(10,10))

        if simplify:
            ax = simplify_axes(ax, exclude_from_simplification=exclude_from_simplification)[0]

        time = np.linspace(0, T, num=width, endpoint=False)
        # copy time vector as many times as there are sweeps
        Time = np.tile(time, (sweeps,1) )


        for i in range(sweeps):
            start = data_start_column
            trace = df.iloc[i, slice(start, start+width)]
            trace = utils.map_range(trace, 0, 5, 2, 4)
            ax.plot(time, trace, 'black', linewidth=1, alpha=0.1)
            ax.set_ylabel('Cell')

            start += width
            trace = df.iloc[i, slice(start, start+width)]
            trace = utils.map_range(trace, 0, 5, 5, 6)
            ax.plot(time, trace, 'red', linewidth=1, alpha=0.1)
            ax.set_ylabel('FrameTTL')

            start += width
            trace = df.iloc[i, slice(start, start+width)]
            trace = utils.map_range(trace, 0, 1, 4, 5)
            ax.plot(time, trace, 'cyan', linewidth=1, alpha=0.1)
            ax.set_ylabel('PD')

            start += width
            trace = df.iloc[i, slice(start, start+width)]
            trace = utils.map_range(trace, -0.5, 0.5, 0, 2)
            ax.plot(time, trace, 'orange', linewidth=1, alpha=0.1)
            ax.set_ylabel('Field')
        
        # plot average sweeps on respective axes
        start = data_start_column
        trace_average = df.iloc[:,   start:start+width].mean(axis=0)
        trace_average = utils.map_range(trace_average, 0, 5, 2, 4)
        ax.plot(time, trace_average, color='black', linewidth=1, label='Cell')
        add_floating_scalebar(ax, scalebar_origin=[0.05, 3.0], xlength=0.1, ylength=0.5, labelx='', labely='', unitx='', unity='',
                                        fontsize=12, color='black', linewidth=2, pad=0.1, simplify=True, exclude_axis_from_simplification=[], show_labels=False)

        start += width
        trace_average = df.iloc[:,start:start+width].mean(axis=0)
        trace_average = utils.map_range(trace_average, 0, 5, 5, 6)
        ax.plot(time, trace_average, color='red', linewidth=1, label='FrameTTL')

        start += width
        trace_average = df.iloc[:,40035:60035].mean(axis=0)
        trace_average = utils.map_range(trace_average, 0, 1, 4, 5)
        ax.plot(time, trace_average, color='cyan', linewidth=1, label='PD')
        
        start += width
        trace_average = df.iloc[:,60035:80035].mean(axis=0)
        trace_average = utils.map_range(trace_average, -0.5, 0.5, 0, 2)
        ax.plot(time, trace_average, color='orange', linewidth=1, label='Field')
        add_floating_scalebar(ax, scalebar_origin=[0.05, 1.2], xlength=0.1, ylength=0.5, labelx='', labely='', unitx='', unity='',
                                        fontsize=12, color='orange', linewidth=2, pad=0.1, simplify=True, exclude_axis_from_simplification=[], show_labels=False)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage')

        return fig, ax
    

def plot_grid(spot_locs=[], spot_values=[], grid=[24,24], ax=None, simplify=True, exclude_from_simplification=[], vmin=0, vmax=1, cmap='gray', **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    if simplify:
        ax = simplify_axes(ax, exclude_from_simplification=exclude_from_simplification)[0]

    # if spot_values == []:
    #     raise ValueError('spot_locs and spot_values must be the same length')
    elif len(spot_values) == 1:
        spot_values = np.repeat(spot_values, len(spot_locs))
    elif len(spot_values) != len(spot_locs):
        raise ValueError('spot_locs and spot_values must be the same length') 


    # make a zero array of the grid size
    grid_array = np.zeros(grid)

    # fill the grid array with the spot locations
    for i in spot_locs:
        locx = i % grid[0]
        locy = i // grid[1]
        grid_array[locy, locx] = spot_values[i]

    ax.imshow(grid_array, cmap=cmap, vmin=vmin, vmax=vmax)
    # have the axis scaled
    # ax.axis('scaled')

    ax.set_xlim(0, grid[0])
    ax.set_ylim(0, grid[1])

    # invert the y axis
    ax.invert_yaxis()

    # add the colorbar
    cbar = plt.colorbar(ax.imshow(grid_array, cmap=cmap, vmin=vmin, vmax=vmax), ax=ax, label='Depolarization (pA)',**kwargs)
    # add colorbar label
    # cbar.set_ylabel('Depolorization (pA)')
    
    ax.set_aspect(1/pattern_index.polygon_frame_properties['aspect_ratio'])

    return locx, locy, ax