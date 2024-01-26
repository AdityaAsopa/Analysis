''' Sanity Checks

1. Is IR stable?
2. Is Ra/Cm/Tau stable?
3. Is ChR2 desensitizing?
4. Is baseline stable?
5. Are there spurious spiks?

'''

import numpy as np
import matplotlib
#matplotlib.use("Agg") # to suppress default matplotlib bitmap backend engine (QtAgg), prevents resource overload
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper')
import pandas as pd
from scipy import signal
from pathlib import Path

from eidynamics import ephys_classes, utils

# cellDirectory = Path("..\\AnalysisFiles\\all_cells_qc\\")

def run_qc(cellObject, cellDirectory, mode='cell'):
    '''
    mode: ['cell', 'batch']
    '''
    global cell_location
    global cellID
    numParams = len(cellObject.metadata_columns)
    cell_location = cellDirectory
    for protocol, protocol_data in cellObject.data.items():
        if protocol_data is not None:
            dataDF = protocol_data.copy()
            dataDF = dataDF.iloc[:,:numParams]
            dataDF = dataDF[dataDF['exptID'] != 0]
            cellID = str(cellObject.cellID) #

            
            is_baseline_stable(dataDF, protocol)
            is_IR_stable(      dataDF, protocol)
            # is_response_present(dataDF, protocol) #KS test for noisy data (for test mentioned in Aanchal-Sahil's paper)
            # is_ChR2_stable(    dataDF, protocol)
            # is_spiking_stable( dataDF, cellID, exptID_range)
            
            # if cellObject.properties.clamp == 'VC':
            # is_Ra_stable(  dataDF, cellID, exptID_range)
            # elif cellObject.properties.clamp == 'CC':
            is_tau_stable(dataDF, protocol)

            print(f'Plots saved in {cell_location}')
        

def is_baseline_stable(dataDf, protocol_name):
    df = dataDf.copy()
    df.sort_values(by=['exptID','sweep'])

    expt_seq = (np.min(np.unique(df["exptSeq"])) , np.max(np.unique(df["exptSeq"])))

    plt.figure()
    graph = sns.catplot(data=df, x='exptID', y='sweepBaseline', kind='box', dodge=False, height=6, aspect=1.33)
    graph.fig.suptitle(cellID)
    plt.savefig(cell_location / (str(cellID) + '_' + protocol_name + '_baseline_trend_expt.png') )

    plt.figure()
    sns.set(rc={"figure.figsize":(12,5)})
    graph = sns.lineplot(data=df, x='sweep', y='sweepBaseline', palette='flare', hue='exptID', hue_norm=expt_seq)
    graph.set_title(cellID)
    
    figpath = cell_location / (cellID + '_' + protocol_name + '_baseline_trend_stacked_sweeps.png')
    plt.savefig(figpath)
    print("saved figs in: ", figpath)
    

    plt.close('all')
    

def is_IR_stable(dataDf, protocol_name):
    df = dataDf.copy()
    df.sort_values(by=['exptID','sweep'])

    expt_seq = (np.min(np.unique(df["exptSeq"])) , np.max(np.unique(df["exptSeq"])))

    plt.figure()
    graph = sns.catplot(data=df, hue='exptID', y='IR', x='exptID', kind='box', dodge=False, height=6, aspect=1.33)
    graph.fig.suptitle(cellID)
    plt.savefig(cell_location / (cellID + '_' + protocol_name + '_IR_trend_expt.png') )

    plt.figure()
    sns.set(rc={"figure.figsize":(12,5)})
    graph = sns.lineplot(data=df, x='sweep', y='IR', palette='flare', hue='exptID', hue_norm=expt_seq)
    graph.set_title(cellID)
    
    figpath = cell_location / (cellID + '_' + protocol_name + '_IR_trend_stacked_sweeps.png')
    
    plt.savefig(figpath)
    print("saved figs in: ", figpath)

    plt.close('all')


def is_response_present(dataDF, protocol):
    from scipy.stats import ks_2samp

    # Assume `data` is your array of 10000 datapoints
    baseline = data[:2000]
    period = data[2000:]

    # Compute the CDF of the baseline period and the period of interest
    cdf_baseline = np.cumsum(np.histogram(baseline, bins=1000, density=True)[0])
    cdf_period = np.cumsum(np.histogram(period, bins=1000, density=True)[0])

    # Calculate the maximum difference between the two CDFs using the KS test
    ks_statistic, p_value = ks_2samp(cdf_baseline, cdf_period)

    # Compare the maximum difference with the critical value at a significance level of 0.05
    critical_value = 1.36 / np.sqrt(len(data))
    if ks_statistic > critical_value:
        print("The two distributions are significantly different.")
    else:
        print("The two distributions are not significantly different.")



def is_ChR2_stable(dataDf, protocol_name):
    df = dataDf.loc[dataDf['numSq']>0]

    expt_seq = (np.min(np.unique(df["exptSeq"])) , np.max(np.unique(df["exptSeq"])))

    df2 = df[['exptID', 'sweep', 'numSq', 'clampPotential', 'firstpulsetime', 'firstpeakres', 'firstpulse_peaktime']]
    df2 = df2.sort_values(by=['exptID', 'sweep'])

    plt.figure()
    graph = sns.catplot(data=df2, x='firstpeakres', y='exptID', hue='numSq', col='clampPotential', kind='swarm', dodge=False, orient="h", palette='mako', height=5, aspect=2.4)
    graph.fig.suptitle(cellID)
    
    figpath = cell_location / (cellID + protocol_name + '_firstpulse_response_trend_vs_exptID.png')
    print(figpath)
    plt.savefig(figpath )

    plt.close('all')
    

def is_spiking_stable(dataDF, protocol_name):
    pass


def is_tau_stable(dataDf, protocol_name):
    df = dataDf.copy()
    df.sort_values(by=['exptID','sweep'])

    expt_seq = (np.min(np.unique(df["exptSeq"])) , np.max(np.unique(df["exptSeq"])))

    plt.figure()
    graph = sns.catplot(data=df, hue='exptID', y='tau', x='exptID', kind='box', dodge=False, height=6, aspect=1.33)
    graph.fig.suptitle(cellID)
    
    plt.savefig(cell_location / (cellID + '_' + protocol_name + '_Tau_trend_expt.png') )

    plt.figure()
    sns.set(rc={"figure.figsize":(12,5)})
    graph = sns.lineplot(data=df, x='sweep', y='tau', palette='flare', hue='exptID', hue_norm=expt_seq)
    graph.set_title(cellID)
    
    figpath = cell_location / (cellID + '_' + protocol_name + '_Tau_trend_stacked_sweeps.png')
    plt.savefig(figpath )
    print("saved figs in: ", figpath)

    plt.close('all')


def is_Ra_stable(dataDF, protocol_name, cellID, exptID_range):
    '''
    find if the mean IR value changes by 20% during the course of expts.
    '''


def _signal_sign_cf(clampingPot, clamp):
    '''
    conversion function to convert CC/VC clamping potential values
    to inverting factors for signal. For VC recordings, -70mV clamp means EPSCs
    that are recorded as negative deflections. To get peaks, we need to invert 
    the signal and take max. 
    But for CC recordings, EPSPs are positive deflections and therefore, no inversion
    is needed.
    In data DF, clamping potential for VC and CC is stored as -70/0 mV and clamp is stored
    as 0 for CC and 1 for VC.

    VC                  CC
    -70 -> E -> -1      -70 -> E -> +1
    0   -> I -> +1
    '''    
    return (1+(clampingPot/35))**clamp