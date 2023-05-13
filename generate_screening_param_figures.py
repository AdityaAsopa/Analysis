import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper')
import pandas as pd
from scipy import signal
from pathlib import Path

from eidynamics import ephys_classes, utils
import all_cells
import collate_dataset

def run_qc(cellObject, cellDirectory):
    global cell_location
    cell_location = cellDirectory

    for protocol, protocol_data in cellObject.data.items():
        if protocol_data is not None:

            dataDF = protocol_data.iloc[:,:29]
            dataDF = dataDF.loc[dataDF['numSq']!=0]
            cellID = cellObject.cellID

            global exptID_range
            exptID_range = ( np.min(np.unique(dataDF['exptID'])), np.max(np.unique(dataDF['exptID'])) )

            fig_gen(dataDF, cellID, cellDirectory )

def fig_gen(dataDF, cellID, cellDirectory, mode='cell'):

    global cell_location
    cell_location = cellDirectory
    
    expt_seq = (np.min(np.unique(dataDF["exptID"])) , np.max(np.unique(dataDF["exptID"])))
    
    metafig = plt.figure(figsize=(16, 8))
    metafig.suptitle(str(cellID))
    
    ax1 = plt.subplot2grid( (3,4), (0,0), rowspan=1, colspan=1)
    ax2 = plt.subplot2grid( (3,4), (1,0), rowspan=1, colspan=1)
    ax3 = plt.subplot2grid( (3,4), (2,0), rowspan=1, colspan=1)
    ax4 = plt.subplot2grid( (3,4), (0,1), rowspan=1, colspan=3)
    ax5 = plt.subplot2grid( (3,4), (1,1), rowspan=1, colspan=3)
    ax6 = plt.subplot2grid( (3,4), (2,1), rowspan=1, colspan=3)
      
    is_baseline_stable(dataDF, cellID, expt_seq, ax1, ax4)
    is_IR_stable(      dataDF, cellID, expt_seq, ax2, ax5)
    # is_ChR2_stable(    dataDF, cellID, expt_seq, ax6) # commented out because patternID, firstpeakres etc are not present in dataframe anymore
    is_tau_stable(     dataDF, cellID, expt_seq, ax3)
    
    plt.tight_layout()
    # plt.show()
    metafig.savefig(cell_location / (str(cellID) + '_parameters.png') )

    plt.close('all')

def is_tau_stable(dataDF, cellID, exptID_range, plot_axis):
    df = dataDF.copy()
    df.sort_values(by=['exptID','sweep'])

    sns.boxplot(data=df, hue='exptID', y='tau', x='exptID', palette='viridis', dodge=False, ax=plot_axis)
    plot_axis.get_legend().remove()

def is_ChR2_stable(dataDF, cellID, exptID_range, plot_axis):
    df = dataDF.copy()
    df = df.sort_values(by=['exptID', 'sweep'])

    sns.stripplot(data=df, x='firstpeakres', y='exptID', hue='clampPotential', orient="h", palette='mako', ax=plot_axis)

def is_IR_stable(dataDf, cellID, exptID_range, plot1_axis, plot2_axis):
    df = dataDf.copy()
    df.sort_values(by=['exptID','sweep'])

    sns.boxplot(data=df, hue='exptID', y='IR', x='exptID', palette='viridis', dodge=False, ax=plot1_axis)
    plot1_axis.get_legend().remove()
    # norm = plt.Normalize(0,10)
    # print(type(norm))

    sns.lineplot(data=df, x='sweep', y='IR', palette='viridis', hue='exptID', hue_norm=exptID_range, ax=plot2_axis)
 
def is_baseline_stable(datadf, cellID, exptID_range, plot1_axis, plot2_axis):
    df = datadf.copy()
    df.sort_values(by=['exptID','sweep'])
    
    sns.boxplot(data=df, x='exptID', y='sweepBaseline', dodge=False, palette='viridis', ax=plot1_axis)
    # norm = plt.Normalize(0,10)
    # print(type(norm))
    sns.lineplot(data=df, x='sweep', y='sweepBaseline', palette='viridis', hue='exptID', hue_norm=exptID_range, ax=plot2_axis)
  
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

def collated_plots():
    collate_dataset.main()

def main():
    for cell in all_cells.all_cells:
        cellpath = all_cells.project_path_root / cell
        cellID = cellpath.stem
        cellpickle = cellpath / (str(cellID) + ".pkl")

        cell = ephys_classes.Neuron.loadCell(cellpickle)
        print(cellpath)
        run_qc(cell, cellpath)

def test():
    for cell in all_cells.test_cells:
        cellpath = Path(cell)
        cellID = cellpath.stem
        cellpickle = cellpath / (str(cellID) + ".pkl")

        cell = ephys_classes.Neuron.loadCell(cellpickle)
        print(cellpath)
        run_qc(cell, cellpath)

if __name__ == "__main__": 
    main()
