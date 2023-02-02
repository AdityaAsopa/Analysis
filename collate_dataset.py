# %%

from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal

sns.set_context('talk')

from eidynamics import utils, data_quality_checks, ephys_classes
import parse_data
import all_cells


# Make a table of all experiments on all cells
def main():
    all_cell_data = []
    sweeps = []

    cell_set = all_cells.all_cells
    print(cell_set)
    for cell in cell_set:
        cellpath = all_cells.project_path_root / cell
        cellID = cellpath.stem
        cellpickle = cellpath / (str(cellID) + ".pkl")
        print(cellpickle)

        neuron = ephys_classes.Neuron.loadCell(cellpickle)
        neuron_data = neuron.data
        print(neuron_data.shape)
        all_cell_data.append(neuron_data)
        sweeps.append(neuron_data.shape[0])
        

    all_expt = pd.concat(all_cell_data, ignore_index=True, axis=0)
    print(all_expt.shape) 
    # del cell_set, cellID, cellpath, cellpickle, neuron_data, _ss, all_cell_data, sweeps, neuron, led, peak_props, first_pulse_start, res_traces, expt_ids, expt_idxs, i, j,    
    all_expt.to_hdf('all_expt_data.h5', format='fixed', key='data', mode='w')

    all_cell_plots(all_expt)

    # %%
def all_cell_plots(all_expt_df):
    df = all_expt_df.iloc[:,:29].copy()

    # %% [markdown]
    # # CC Cells

    # %% [markdown]
    # ### Fig 1: CC | [expt_seq vs first-pulse-response] X cell_ID
    # Does the first pulse response decrease during an experiment session on a cell?
    # 
    # An experiment session consists of an array of experiments done on the cell with every experiment corresponding to one protocol (one stim frequency)
    # 
    # Observation: Across all the patterns, as the session progresses, the EPSP response of the cell to the optical stimulation decreases.

    # %%
    plt.close('all')
    plt.figure()
    df_subset = df.loc[ (df["Clamp"] == "CC")  & (df["AP"]== 0.0) & (df["intensity"]== 100.0) & (df["pulseWidth"]== 2.0) & (df["AP"]== 0.0) & (df["patternID"] < 56) & (df["Condition"]== "CTRL") & (df["numSq"]>= 1.0)].copy() 
    

    print(df_subset)

    cat = sns.catplot(data=df_subset, y='firstpeakres', x='expt_seq', hue='numSq', col='cellID', col_wrap=5, palette='viridis', kind='point', markers=["^", "o", "s"] ) #  
    cat.fig.suptitle('CC | [expt_seq vs first-pulse-response] X cell_ID | Control, Intensity=100%, Pulse Width = 2ms')
    cat.fig.subplots_adjust(top=.9)
    figpath = all_cells.project_path_root / "Lab\\Projects\\EI_Dynamics\\AnalysisFiles\\all_cell_collate_qc\\"
    plt.savefig(figpath / 'CC_expt_seq_vs_first-pulse-response_X_cell_ID.png')

    # %% [markdown]
    # ### Fig 2: CC | [Expt sequence vs IR ] x Cell_ID
    # Does the input resistance of the cell change during an experiment session on a cell?
    # 
    # An experiment session consists of an array of experiments done on the cell with every experiment corresponding to one protocol (one stim frequency)
    # 

    # %%
    plt.close('all')
    plt.figure()
    df_subset = df.loc[ (df["Clamp"] == "CC") & (df["Condition"]== "CTRL") & (df["intensity"]== 100.0) & (df["pulseWidth"]== 2.0)   & (df["patternID"] < 56) ].copy()
    cat = sns.catplot(data=df_subset, y='InputRes', x='expt_seq', hue='numSq', col='cellID', palette='viridis', col_wrap=5, kind='point', markers=["^", "o", "s"] )
    cat.fig.suptitle('CC | [expt_seq vs Input Resistance (MOhm)] X cell_ID | Control, Intensity=100%, Pulse Width = 2ms')
    cat.fig.subplots_adjust(top=.9)
    figpath = all_cells.project_path_root / "Lab\\Projects\\EI_Dynamics\\AnalysisFiles\\all_cell_collate_qc\\"
    plt.savefig(figpath / 'CC_expt_seq_vs_IR_X_cell_ID.png')

    # %% [markdown]
    # ### Fig 3: CC | [Expt sequence vs Tau ] x Cell_ID
    # Does the Tau of the cell change during an experiment session on a cell?
    # 
    # An experiment session consists of an array of experiments done on the cell with every experiment corresponding to one protocol (one stim frequency)
    # 

    # %%
    plt.close('all')
    plt.figure()
    df_subset = df.loc[ (df["Clamp"] == "CC") & (df["Condition"]== "CTRL") & (df["intensity"]== 100.0) & (df["pulseWidth"]== 2.0)   & (df["patternID"] < 56) ].copy() 
    df_subset.loc[:,"Tau"] *= 1000
    cat = sns.catplot(data=df_subset, y='Tau', x='expt_seq', hue='numSq', col='cellID', palette='viridis', col_wrap=5, kind='point', markers=["^", "o", "s"] )
    cat.fig.suptitle('CC | [expt_seq vs Tau (ms)] X cell_ID | Control, Intensity=100%, Pulse Width = 2ms')
    cat.fig.subplots_adjust(top=.9)
    figpath = all_cells.project_path_root / "Lab\\Projects\\EI_Dynamics\\AnalysisFiles\\all_cell_collate_qc\\"
    plt.savefig(figpath / 'CC_expt_seq_vs_Tau_X_cell_ID.png')

    # %% [markdown]
    # ### Fig 4: CC | [Expt sequence vs Baseline (mV) ] x Cell_ID
    # Does the Vm of the cell change during an experiment session on a cell?
    # 
    # An experiment session consists of an array of experiments done on the cell with every experiment corresponding to one protocol (one stim frequency)
    # 

    # %%
    plt.close('all')
    plt.figure()
    df_subset = df.loc[ (df["Clamp"] == "CC") & (df["Condition"]== "CTRL") & (df["intensity"]== 100.0) & (df["pulseWidth"]== 2.0)   & (df["patternID"] < 56) ].copy() 
    df_subset.loc[:,"Tau"] *= 1000
    cat = sns.catplot(data=df_subset, y='MeanBaseline', x='expt_seq', hue='numSq', col='cellID', palette='viridis', col_wrap=5, kind='point', markers=["^", "o", "s"] )
    cat.set(ylim=(-80, -50))
    cat.fig.suptitle('CC | [expt_seq vs Baseline (mV)] X cell_ID | Control, Intensity=100%, Pulse Width = 2ms')
    cat.fig.subplots_adjust(top=.9)
    figpath = all_cells.project_path_root / "Lab\\Projects\\EI_Dynamics\\AnalysisFiles\\all_cell_collate_qc\\"
    plt.savefig(figpath / 'CC_expt_seq_vs_Baseline_X_cell_ID.png')




    # =========================================================================================================
    # =========================================================================================================


    # %% [markdown]
    # # VC Cells

    # %% [markdown]
    # ### Fig 5: VC | [Expt sequence vs first-pulse-response x EI ] x numSq x_cellID
    # Does the first pulse response decrease during an experiment session on a cell?
    # 
    # An experiment session consists of an array of experiments done on the cell with every experiment corresponding to one protocol (one stim frequency)
    # 
    # Observation: Across all the patterns, as the session progresses, the EPSP response of the cell to the optical stimulation decreases.

    # %%
    plt.close('all')
    plt.figure()
    df_subset = df.loc[ (df["Clamp"] == "VC")  & (df["intensity"]== 100.0) & (df["pulseWidth"]== 2.0) & (df["AP"]== 0.0) & (df["patternID"] < 56) & (df["Condition"]== "CTRL") & (df["numSq"]>= 1.0)].copy() 
    cat = sns.catplot(data=df_subset, y='firstpeakres', x='expt_seq', hue='ClampingPotl', col='cellID', row='numSq', palette='viridis', kind='strip')#, markers=["^", "o", "s"] ) #  
    cat.fig.suptitle('VC | [expt_seq vs first-pulse-response x E/I ] X numSq x cell_ID | Control, Intensity=100%, Pulse Width=2ms')
    cat.fig.subplots_adjust(top=.9)
    figpath = all_cells.project_path_root / "Lab\\Projects\\EI_Dynamics\\AnalysisFiles\\all_cell_collate_qc\\"
    plt.savefig(figpath / 'VC_expt_seq_vs_first-pulse-response_X_EI_X_numSq_X_cell_ID.png')

    # %% [markdown]
    # ### Fig 6: VC | [Expt sequence vs series resistance x EI ] x numSq x_cellID
    # Does the input resistance of the cell change during an experiment session on a cell?
    # 
    # An experiment session consists of an array of experiments done on the cell with every experiment corresponding to one protocol (one stim frequency)
    # 

    # %%
    plt.close('all')
    plt.figure()
    df_subset = df.loc[ (df["Clamp"] == "VC") & (df["Condition"]== "CTRL") & (df["intensity"]== 100.0) & (df["pulseWidth"]== 2.0)   & (df["patternID"] < 56) & (df["Tau"] < 100) & (df["Tau"] > 0.0)].copy()
    cat = sns.catplot(data=df_subset, y='Tau', x='expt_seq', hue='numSq', col='cellID', palette='viridis', col_wrap=5, kind='point', markers=["^", "o", "s"])
    cat.fig.suptitle('VC | [expt_seq vs Series Resistance (MOhm)] X cell_ID | Control, Intensity=100%, Pulse Width = 2ms')
    cat.fig.subplots_adjust(top=.9)
    figpath = all_cells.project_path_root / "Lab\\Projects\\EI_Dynamics\\AnalysisFiles\\all_cell_collate_qc\\"
    plt.savefig(figpath / 'VC_expt_seq_vs_Rs_X_EI_X_numSq_X_cell_ID.png')

    # %% [markdown]
    # ### Fig 7: VC | [Expt sequence vs Baseline (mV) ] x Cell_ID
    # Does the Vm of the cell change during an experiment session on a cell?
    # 
    # An experiment session consists of an array of experiments done on the cell with every experiment corresponding to one protocol (one stim frequency)
    # 

    # %%
    plt.close('all')
    plt.figure()
    df_subset = df.loc[ (df["Clamp"] == "VC") & (df["Condition"]== "CTRL") & (df["intensity"]== 100.0) & (df["pulseWidth"]== 2.0)   & (df["patternID"] < 56) ].copy() 
    df_subset.loc[:,"Tau"] *= 1000
    cat = sns.catplot(data=df_subset, y='MeanBaseline', x='expt_seq', hue='numSq', col='cellID', palette='viridis', col_wrap=5, kind='point', markers=["^", "o", "s"] )
    cat.set(ylim=(-80, -50))
    cat.fig.suptitle('VC | [expt_seq vs Baseline (mV)] X cell_ID | Control, Intensity=100%, Pulse Width = 2ms')
    cat.fig.subplots_adjust(top=.9)
    figpath = all_cells.project_path_root / "Lab\\Projects\\EI_Dynamics\\AnalysisFiles\\all_cell_collate_qc\\"
    plt.savefig(figpath / 'VC_expt_seq_vs_Baseline_X_cell_ID.png')


if __name__ == "__main__": 
    main()
