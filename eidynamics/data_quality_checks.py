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
    
    cell_location = cellDirectory
    for protocol, protocol_data in cellObject.data.items():
        if protocol_data is not None:
            dataDF = protocol_data.copy()
            dataDF = dataDF.iloc[:,:40]
            dataDF = dataDF[dataDF['exptID'] != 0]
            cellID = str(cellObject.cellID)

            
            is_baseline_stable(dataDF)
            is_IR_stable(      dataDF)
            # is_ChR2_stable(    dataDF) # commented out due to patternID being a list now, not a single number
            # is_spiking_stable( dataDF, cellID, exptID_range)
            
            # if cellObject.properties.clamp == 'VC':
            # is_Ra_stable(  dataDF, cellID, exptID_range)
            # elif cellObject.properties.clamp == 'CC':
            is_tau_stable(dataDF)

            print('Plots saved in {}'.format( (cell_location  ) ) ) #/ str(cellID)
        

def is_baseline_stable(dataDf):
    df = dataDf.copy()
    df.sort_values(by=['exptID','sweep'])

    expt_seq = (np.min(np.unique(df["exptSeq"])) , np.max(np.unique(df["exptSeq"])))

    plt.figure()
    graph = sns.catplot(data=df, x='exptID', y='sweepBaseline', kind='box', dodge=False, height=6, aspect=1.33)
    graph.fig.suptitle(cellID)
    plt.savefig(cell_location / (str(cellID) + '_baseline_trend_expt.png') )

    plt.figure()
    sns.set(rc={"figure.figsize":(12,5)})
    graph = sns.lineplot(data=df, x='sweep', y='sweepBaseline', palette='flare', hue='exptID', hue_norm=expt_seq)
    graph.set_title(cellID)
    
    figpath = cell_location / (cellID + '_baseline_trend_stacked_sweeps.png')
    print(figpath)
    plt.savefig(figpath)
    

    plt.close('all')
    

def is_IR_stable(dataDf):
    df = dataDf.copy()
    df.sort_values(by=['exptID','sweep'])

    expt_seq = (np.min(np.unique(df["exptSeq"])) , np.max(np.unique(df["exptSeq"])))

    plt.figure()
    graph = sns.catplot(data=df, hue='exptID', y='IR', x='exptID', kind='box', dodge=False, height=6, aspect=1.33)
    graph.fig.suptitle(cellID)
    plt.savefig(cell_location / (cellID + '_IR_trend_expt.png') )

    plt.figure()
    sns.set(rc={"figure.figsize":(12,5)})
    graph = sns.lineplot(data=df, x='sweep', y='IR', palette='flare', hue='exptID', hue_norm=expt_seq)
    graph.set_title(cellID)
    
    figpath = cell_location / (cellID + '_IR_trend_stacked_sweeps.png')
    print(figpath)
    plt.savefig(figpath)

    plt.close('all')


def is_ChR2_stable(dataDf):
    df = dataDf.loc[dataDf['numSq']>0]

    expt_seq = (np.min(np.unique(df["exptSeq"])) , np.max(np.unique(df["exptSeq"])))

    df2 = df[['exptID', 'sweep', 'numSq', 'clampPotential', 'patternID', 'firstpulsetime', 'firstpeakres', 'firstpulse_peaktime']]
    df2 = df2.sort_values(by=['exptID', 'sweep'])

    plt.figure()
    graph = sns.catplot(data=df2, x='firstpeakres', y='exptID', hue='numSq', col='clampPotential', kind='swarm', dodge=False, orient="h", palette='mako', height=5, aspect=2.4)
    graph.fig.suptitle(cellID)
    
    figpath = cell_location / (cellID + '_firstpulse_response_trend_vs_exptID.png')
    print(figpath)
    plt.savefig(figpath )

    plt.close('all')
    

def is_spiking_stable(dataDF):
    pass


def is_tau_stable(dataDf):
    df = dataDf.copy()
    df.sort_values(by=['exptID','sweep'])

    expt_seq = (np.min(np.unique(df["exptSeq"])) , np.max(np.unique(df["exptSeq"])))

    plt.figure()
    graph = sns.catplot(data=df, hue='exptID', y='tau', x='exptID', kind='box', dodge=False, height=6, aspect=1.33)
    graph.fig.suptitle(cellID)
    
    plt.savefig(cell_location / (cellID + '_Tau_trend_expt.png') )

    plt.figure()
    sns.set(rc={"figure.figsize":(12,5)})
    graph = sns.lineplot(data=df, x='sweep', y='tau', palette='flare', hue='exptID', hue_norm=expt_seq)
    graph.set_title(cellID)
    
    figpath = cell_location / (cellID + '_Tau_trend_stacked_sweeps.png')
    print(figpath)
    plt.savefig(figpath )

    plt.close('all')


def is_Ra_stable(dataDF, cellID, exptID_range):
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