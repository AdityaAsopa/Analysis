import sys
from pathlib import Path
import traceback
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal

sns.set_context('talk')

from eidynamics import utils, data_quality_checks, ephys_classes
from eidynamics import expt_to_dataframe
import parse_data
import all_cells
from typing import Optional
import argparse


def main(cell_set: list = None, protocols: list = ['FreqSweep'], clampMode: str = 'all') -> None:
    if cell_set is None:
        cell_set = all_cells.all_cells

    for protocol in protocols:
        all_expt_ = pd.DataFrame()
        x = 0
        for cell in cell_set:
            print(cell)
            cellpath = all_cells.project_path_root / cell
            cellID = cellpath.stem
            cellpickle = cellpath / (str(cellID) + ".pkl")

            try:
                neuron = ephys_classes.Neuron.loadCell(cellpickle)
                neuron_data = neuron.data[protocol]
                if neuron_data is None:
                    continue
                # convert all columns from 37 and onwards to 'float32' type
                neuron_data.iloc[:, 37:] = neuron_data.iloc[:, 37:].astype(np.float32)
            except Exception as err:
                print(err)
                # traceback of the error
                traceback.print_exc()
                print("Error loading cell: ", cellpickle)
                continue
            
            all_expt_ = pd.concat([all_expt_, neuron_data], ignore_index=True, axis=0)
            x += neuron_data.shape[0]
            del neuron_data
        
        # if clampMode is all, then do nothing and move on, but if clampMode is 'CC' or 'VC', then filter the df
        if clampMode == 'CC':
            print('processing only CC and loose patch data')
            all_expt_ = all_expt_[all_expt_['clampMode'] != 'VC']
        elif clampMode == 'VC':
            print('processing only VC data')
            all_expt_ = all_expt_[all_expt_['clampMode'] == 'VC']
        else:
            pass
            

        if not all_expt_.empty:
            print(f'{protocol} has {all_expt_["cellID"].nunique()} unique cells with total {all_expt_.shape[0]} sweeps')

            # add analysis params to the df
            print("adding analysis params to the df")
            df_short, df_long = expt_to_dataframe.add_analysed_params2(all_expt_)
            del all_expt_

            # save  dfs
            print("return from expt_to_dataframe. Saving DFs")
            save_df_to_h5(df_short, filename_suffix='short', protocol=protocol, clampMode=clampMode, save_combined_also=True)
            del df_short
            print("Saving the large DF now")
            save_df_to_h5(df_long, filename_suffix='long', protocol=protocol, clampMode=clampMode, save_combined_also=True)

def all_cell_plots(all_expt_df):
    df = all_expt_df.iloc[:,:29].copy()
 
    # # CC Cells

    # ### Fig 1: CC | [expt_seq vs first-pulse-response] X cell_ID
    # Does the first pulse response decrease during an experiment session on a cell?
    # An experiment session consists of an array of experiments done on the cell with every experiment corresponding to one protocol (one stim frequency)
    # Observation: Across all the patterns, as the session progresses, the EPSP response of the cell to the optical stimulation decreases.

    plt.close('all')
    plt.figure()
    df_subset = df.loc[ (df["Clamp"] == "CC")  & (df["AP"]== 0.0) & (df["intensity"]== 100.0) & (df["pulseWidth"]== 2.0) & (df["AP"]== 0.0) & (df["patternID"] < 56) & (df["Condition"]== "CTRL") & (df["numSq"]>= 1.0)].copy() 
    

    print(df_subset)

    cat = sns.catplot(data=df_subset, y='firstpeakres', x='expt_seq', hue='numSq', col='cellID', col_wrap=5, palette='viridis', kind='point', markers=["^", "o", "s"] ) #  
    cat.fig.suptitle('CC | [expt_seq vs first-pulse-response] X cell_ID | Control, Intensity=100%, Pulse Width = 2ms')
    cat.fig.subplots_adjust(top=.9)
    figpath = all_cells.project_path_root / "Lab\\Projects\\EI_Dynamics\\AnalysisFiles\\all_cell_collate_qc\\"
    plt.savefig(figpath / 'CC_expt_seq_vs_first-pulse-response_X_cell_ID.png')

    # [markdown]
    # ### Fig 2: CC | [Expt sequence vs IR ] x Cell_ID
    # Does the input resistance of the cell change during an experiment session on a cell?
    # 
    # An experiment session consists of an array of experiments done on the cell with every experiment corresponding to one protocol (one stim frequency)
    # 

    #
    plt.close('all')
    plt.figure()
    df_subset = df.loc[ (df["Clamp"] == "CC") & (df["Condition"]== "CTRL") & (df["intensity"]== 100.0) & (df["pulseWidth"]== 2.0)   & (df["patternID"] < 56) ].copy()
    cat = sns.catplot(data=df_subset, y='InputRes', x='expt_seq', hue='numSq', col='cellID', palette='viridis', col_wrap=5, kind='point', markers=["^", "o", "s"] )
    cat.fig.suptitle('CC | [expt_seq vs Input Resistance (MOhm)] X cell_ID | Control, Intensity=100%, Pulse Width = 2ms')
    cat.fig.subplots_adjust(top=.9)
    figpath = all_cells.project_path_root / "Lab\\Projects\\EI_Dynamics\\AnalysisFiles\\all_cell_collate_qc\\"
    plt.savefig(figpath / 'CC_expt_seq_vs_IR_X_cell_ID.png')

    # [markdown]
    # ### Fig 3: CC | [Expt sequence vs Tau ] x Cell_ID
    # Does the Tau of the cell change during an experiment session on a cell?
    # 
    # An experiment session consists of an array of experiments done on the cell with every experiment corresponding to one protocol (one stim frequency)
    # 

    #
    plt.close('all')
    plt.figure()
    df_subset = df.loc[ (df["Clamp"] == "CC") & (df["Condition"]== "CTRL") & (df["intensity"]== 100.0) & (df["pulseWidth"]== 2.0)   & (df["patternID"] < 56) ].copy() 
    df_subset.loc[:,"Tau"] *= 1000
    cat = sns.catplot(data=df_subset, y='Tau', x='expt_seq', hue='numSq', col='cellID', palette='viridis', col_wrap=5, kind='point', markers=["^", "o", "s"] )
    cat.fig.suptitle('CC | [expt_seq vs Tau (ms)] X cell_ID | Control, Intensity=100%, Pulse Width = 2ms')
    cat.fig.subplots_adjust(top=.9)
    figpath = all_cells.project_path_root / "Lab\\Projects\\EI_Dynamics\\AnalysisFiles\\all_cell_collate_qc\\"
    plt.savefig(figpath / 'CC_expt_seq_vs_Tau_X_cell_ID.png')

    # [markdown]
    # ### Fig 4: CC | [Expt sequence vs Baseline (mV) ] x Cell_ID
    # Does the Vm of the cell change during an experiment session on a cell?
    # 
    # An experiment session consists of an array of experiments done on the cell with every experiment corresponding to one protocol (one stim frequency)
    # 

    #
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


    # [markdown]
    # # VC Cells

    # [markdown]
    # ### Fig 5: VC | [Expt sequence vs first-pulse-response x EI ] x numSq x_cellID
    # Does the first pulse response decrease during an experiment session on a cell?
    # 
    # An experiment session consists of an array of experiments done on the cell with every experiment corresponding to one protocol (one stim frequency)
    # 
    # Observation: Across all the patterns, as the session progresses, the EPSP response of the cell to the optical stimulation decreases.

    #
    plt.close('all')
    plt.figure()
    df_subset = df.loc[ (df["Clamp"] == "VC")  & (df["intensity"]== 100.0) & (df["pulseWidth"]== 2.0) & (df["AP"]== 0.0) & (df["patternID"] < 56) & (df["Condition"]== "CTRL") & (df["numSq"]>= 1.0)].copy() 
    cat = sns.catplot(data=df_subset, y='firstpeakres', x='expt_seq', hue='ClampingPotl', col='cellID', row='numSq', palette='viridis', kind='strip')#, markers=["^", "o", "s"] ) #  
    cat.fig.suptitle('VC | [expt_seq vs first-pulse-response x E/I ] X numSq x cell_ID | Control, Intensity=100%, Pulse Width=2ms')
    cat.fig.subplots_adjust(top=.9)
    figpath = all_cells.project_path_root / "Lab\\Projects\\EI_Dynamics\\AnalysisFiles\\all_cell_collate_qc\\"
    plt.savefig(figpath / 'VC_expt_seq_vs_first-pulse-response_X_EI_X_numSq_X_cell_ID.png')

    # [markdown]
    # ### Fig 6: VC | [Expt sequence vs series resistance x EI ] x numSq x_cellID
    # Does the input resistance of the cell change during an experiment session on a cell?
    # 
    # An experiment session consists of an array of experiments done on the cell with every experiment corresponding to one protocol (one stim frequency)
    # 

    #
    plt.close('all')
    plt.figure()
    df_subset = df.loc[ (df["Clamp"] == "VC") & (df["Condition"]== "CTRL") & (df["intensity"]== 100.0) & (df["pulseWidth"]== 2.0)   & (df["patternID"] < 56) & (df["Tau"] < 100) & (df["Tau"] > 0.0)].copy()
    cat = sns.catplot(data=df_subset, y='Tau', x='expt_seq', hue='numSq', col='cellID', palette='viridis', col_wrap=5, kind='point', markers=["^", "o", "s"])
    cat.fig.suptitle('VC | [expt_seq vs Series Resistance (MOhm)] X cell_ID | Control, Intensity=100%, Pulse Width = 2ms')
    cat.fig.subplots_adjust(top=.9)
    figpath = all_cells.project_path_root / "Lab\\Projects\\EI_Dynamics\\AnalysisFiles\\all_cell_collate_qc\\"
    plt.savefig(figpath / 'VC_expt_seq_vs_Rs_X_EI_X_numSq_X_cell_ID.png')

    # [markdown]
    # ### Fig 7: VC | [Expt sequence vs Baseline (mV) ] x Cell_ID
    # Does the Vm of the cell change during an experiment session on a cell?
    # 
    # An experiment session consists of an array of experiments done on the cell with every experiment corresponding to one protocol (one stim frequency)
    # 

    #
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

def save_df_to_h5(df: pd.DataFrame, filename_suffix: str, protocol: str = 'FreqSweep', clampMode: str = 'CC', save_combined_also: bool = False) -> None:
    if save_combined_also:
        combineddata_filename = f"parsed_data/all_cells_{protocol}_combined_{filename_suffix}.h5"
        df.to_hdf(combineddata_filename, format='fixed', key='data', mode='w')
    
    if clampMode == 'CC':
        df_save = df[df['clampMode'] != 'VC']
        filename = f"parsed_data/all_cells_{protocol}_CC_{filename_suffix}.h5"
        df_save.to_hdf(filename, format='fixed', key='data', mode='w')
    elif clampMode == 'VC':
        df_save = df[df['clampMode'] == 'VC']
        filename = f"parsed_data/all_cells_{protocol}_VC_{filename_suffix}.h5"
        df_save.to_hdf(filename, format='fixed', key='data', mode='w')
    elif clampMode == 'all':
        df_save = df[df['clampMode'] != 'VC']
        filename = f"parsed_data/all_cells_{protocol}_CC_{filename_suffix}.h5"
        df_save.to_hdf(filename, format='fixed', key='data', mode='w')

        df_save = df[df['clampMode'] == 'VC']
        filename = f"parsed_data/all_cells_{protocol}_VC_{filename_suffix}.h5"
        df_save.to_hdf(filename, format='fixed', key='data', mode='w')


   
if __name__ == "__main__":
    protocols = ['FreqSweep','grid', 'LTMRand', 'surprise', 'convergence', 'SpikeTrain']#,'FreqSweep']
    
    parser = argparse.ArgumentParser(description='Process all cells')
    parser.add_argument('--protocol', type=str, default='FreqSweep', help='protocol name')
    parser.add_argument('--clampMode', type=str, default='all', help="'all', 'CC', 'VC'")
    
    args = parser.parse_args()

    # if protocol arg is 'all' then cycle through all the protocols from the list
    if args.protocol == 'all':
        for protocol in protocols:
            main(protocols=[protocol], clampMode=args.clampMode)
    else:
        main(protocols=[args.protocol], clampMode=args.clampMode)


