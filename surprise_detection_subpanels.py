'''
The script should be imported into the main Fig8 script and the function plot_surprise_expt_subpanels(ax1,ax2,ax3) should be called from there.
The function will plot the three subpanels of the surprise detection experiment.

This script is self-sufficient and will load its own data from the storage.
For the last plot, the raw data will be loaded and processed using Upi's surprise_w2 script.
The functions will return the axes object after plotting.

If run from the terminal, the script will make a new figure and save both svg and png versions.
'''

import numpy as np
import pandas as pd
import matplotlib           as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import surprise_w2 as surprise
from scipy.stats import wilcoxon

sns.set_context('paper')
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 16
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['lines.linewidth'] = 2

global datapath


def plot_transition(chosen_transitions=[1,2,3], chosen_freq=20, chosen_numsq=15, chosen_cells = [3101, 2681, 2682, 2822], ax=None):
    datapath  = Path( r"c:\\users\aditya\\onedrive\\ncbs\\Lab\\Projects\\EI_Dynamics\\Analysis\\parsed_data\\Surprise\\")
    datapath = Path(datapath) / "all_cells_surprise_responses_only_transitions.h5"
    datadf_long = pd.read_hdf(datapath, key='data')
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    sample = datadf_long[
                        (datadf_long['freq']==chosen_freq) &
                        (datadf_long['numSq']==chosen_numsq) &
                        (datadf_long['transition_count'].isin(chosen_transitions)) &
                        (datadf_long['cell'].isin(chosen_cells))
                        ]
    # subtract the mean of two pre and two post values from all pre and post values
    sample['response'] = sample['response'] - sample.groupby(['cell', 'expt', 'freq', 'repeat', 'transition_count'])['response'].transform('mean')

    sns.pointplot(x='pulse', y='response', hue='cell', data=sample, palette='tab10', dodge=True, ax=ax)
    ax.axvspan(1.5,3, color='red', alpha=0.2)
    ax.axvspan(0, 1.5, color='green', alpha=0.2)
    ax.text(0.5, 0.45, 'pre', fontsize=16, color='k')
    ax.text(2.5, 0.45, 'post', fontsize=16, color='k')

    return ax


def plot_transition_response_histogram(chosen_transitions=[1,2,3], chosen_freq=20, chosen_numsq=15, ax=None):
    datapath  = Path( r"c:\\users\aditya\\onedrive\\ncbs\\Lab\\Projects\\EI_Dynamics\\Analysis\\parsed_data\\Surprise\\")
    datapath = Path(datapath) / "all_cells_surprise_responses_transition_wide.h5"
    datadf_wide = pd.read_hdf(datapath, key='data')

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    sample = datadf_wide[
                        (datadf_wide['freq']==chosen_freq) &
                        (datadf_wide['numSq']==chosen_numsq) &
                        (datadf_wide['transition_count'].isin(chosen_transitions))
                        ]
    # subtract the mean of two pre and two post values from all pre and post values
    sns.histplot(data=sample, x='transition_response', binwidth=0.05, kde=True, element='step', ax=ax)
    ax.set_xlim(-1,1)

    return ax


def plot_transition_heatmap(ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    datapath  = Path( r"c:\\users\aditya\\onedrive\\ncbs\\Lab\\Projects\\EI_Dynamics\\Analysis\\parsed_data\\Surprise\\")
    rawdatapath = datapath / "all_cells_surprise_CC_long.h5"
    rawdatadf = pd.read_hdf(rawdatapath, key='data')

    pvals = []
    for n in [5,15]:
        sample = rawdatadf[(rawdatadf['numSq']==n)]
        for cc in sample['cellID'].unique():
            cell_sample = sample[sample['cellID']==cc]
            pk, freq = surprise.scanData(cell_sample) # get responses from Upi's script
            for ff in (sorted(freq)):
                for t in [1,2,3]:
                    preidx, postidx = 8*(t-1)+7, 8*(t-1)+9
                    pre  = np.array(freq[ff])[:, preidx : preidx+2 ].flatten()
                    post = np.array(freq[ff])[:, postidx: postidx+2].flatten()
                    pval = wilcoxon(pre, post, alternative='less').pvalue
                    pvals.append({
                                'cell': cc,
                                'freq': ff,
                                'numSq':n,
                                'transition':t,
                                'pval':pval
                                })
    pvals_df = pd.DataFrame(pvals)
    pvals_df['significant'] = pvals_df['pval'] < 0.05
    pvals_df['significant'] = pvals_df['significant'].astype(int)

    fraction_significant_transitions = pvals_df.groupby(['numSq', 'freq'])['significant'].mean().reset_index()
    # pivot the fraction_significant_transitions
    pivotdf = fraction_significant_transitions.pivot_table(index='numSq', columns='freq', values='significant')
    # heatmap
    sns.heatmap(pivotdf, annot=True, ax=ax)

    return ax


def plot_surprise_expt_subpanels(ax1,ax2,ax3):
    ''' any tinkering with the axes can be done here after the plots are returned from their respective functions'''
    # plot 1
    plot_transition(chosen_transitions=[1,2,3], chosen_freq=20, chosen_numsq=15, chosen_cells = [3101, 2681, 2682, 2822], ax=ax1)
    ax1.set_title('Transition responses')

    # plot 2
    plot_transition_response_histogram(chosen_transitions=[1,2,3], chosen_freq=20, chosen_numsq=15, ax=ax2)
    ax2.set_title('Transition response histogram')

    # plot 3
    plot_transition_heatmap(ax=ax3)
    ax3.set_title('Transition responses heatmap')

    return ax1, ax2, ax3

if __name__ == "__main__":
    fig, axs = plt.subplots(3, 1, figsize=(10, 30))
    plot_surprise_expt_subpanels(axs[0], axs[1], axs[2])
    plt.tight_layout()
    plt.show()

    # save fig
    fig.savefig('surprise_detection_subpanels.svg', format='svg')
    fig.savefig('surprise_detection_subpanels.png', format='png')