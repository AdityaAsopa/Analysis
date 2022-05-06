''' Sanity Checks

1. Is IR stable?
2. Is Ra/Cm/Tau stable?
3. Is ChR2 desensitizing?
4. Is baseline stable?
5. Are there spurious spiks?

'''

import numpy as np
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
    cell_location = cellDirectory

    dataDF = cellObject.data.copy()
    dataDF = dataDF.loc[dataDF['exptID']!=0]
    cellID = cellObject.cellID
    exptID_range = ( np.min(np.unique(dataDF['exptID'])), np.max(np.unique(dataDF['exptID'])) )
    
    is_baseline_stable(dataDF, cellID, exptID_range)
    is_IR_stable(      dataDF, cellID, exptID_range)
    is_ChR2_stable(    dataDF, cellID, exptID_range)
    # is_spiking_stable( dataDF, cellID, exptID_range)
    
    # if cellObject.properties.clamp == 'VC':
    # is_Ra_stable(  dataDF, cellID, exptID_range)
    # elif cellObject.properties.clamp == 'CC':
    is_tau_stable( dataDF, cellID, exptID_range)
        

def is_baseline_stable(datadf, cellID, exptID_range):
    df = datadf.iloc[:,[0,1,6]]
    df.sort_values(by=['exptID','sweep'])
    
    plt.figure()
    sns.catplot(data=df, x='exptID', y='MeanBaseline', kind='box', dodge=False)
    plt.savefig(cell_location / (str(cellID) + '_baseline_trend_expt.png') )

    plt.figure()
    sns.lineplot(data=df, x='sweep', y='MeanBaseline', palette='flare', hue='exptID', hue_norm=exptID_range)
    plt.savefig(cell_location / (str(cellID) + '_baseline_trend_stacked_sweeps.png') )

    plt.close('all')
    

def is_IR_stable(dataDf, cellID, exptID_range):
    df = dataDf.iloc[:,[0,1,11]]
    df.sort_values(by=['exptID','sweep'])

    plt.figure()
    sns.catplot(data=df, hue='exptID', y='InputRes', x='exptID', kind='box', dodge=False)
    plt.savefig(cell_location / (str(cellID) + '_IR_trend_expt.png') )

    plt.figure()
    sns.lineplot(data=df, x='sweep', y='InputRes', palette='flare', hue='exptID', hue_norm=exptID_range)
    plt.savefig(cell_location / (str(cellID) + '_IR_trend_stacked_sweeps.png') )

    plt.close('all')


def is_ChR2_stable(dataDF, cellID, exptID_range):
    df = dataDF.loc[dataDF['numSq']>0]

    led = df.iloc[1,29:20029]
    led = np.where(led>0.9*np.max(led), np.max(led), 0)
    _, peak_props = signal.find_peaks(led, height=np.max(led), width=38)
    first_pulse_start = int(peak_props['left_ips'][0]) + 20029

    df['firstpulsestart'] = first_pulse_start
    _ss = _signal_sign_cf(df.iloc[:,7], df.iloc[:,8])
    res_traces = (df.iloc[:, first_pulse_start: first_pulse_start+1000]).multiply(_ss, axis=0)

    df['peakres'] = np.max( res_traces , axis=1)

    df2 = df.loc[:,('exptID', 'sweep', 'numSq', 'ClampingPotl', 'patternID', 'firstpulsestart', 'peakres')]
    df2 = df2.sort_values(by=['exptID', 'sweep'])

    plt.figure()
    sns.catplot(data=df2, x='peakres', y='exptID', hue='numSq', col='ClampingPotl', kind='swarm', dodge=False, orient="h", palette='mako')
    plt.savefig(cell_location / (str(cellID) + '_firstpulse_response_trend_vs_exptID.png') )

    plt.close('all')
    

def is_spiking_stable(dataDF):
    pass


def is_tau_stable(dataDF, cellID, exptID_range):
    df = dataDF.iloc[:,[0,1,12]]
    df.sort_values(by=['exptID','sweep'])

    plt.figure()
    sns.catplot(data=df, hue='exptID', y='Tau', x='exptID', kind='box', dodge=False)
    plt.savefig(cell_location / (str(cellID) + '_Tau_trend_expt.png') )

    plt.figure()
    sns.lineplot(data=df, x='sweep', y='Tau', palette='flare', hue='exptID', hue_norm=exptID_range)
    plt.savefig(cell_location / (str(cellID) + '_Tau_trend_stacked_sweeps.png') )

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