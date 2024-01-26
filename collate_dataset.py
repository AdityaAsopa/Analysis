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


# def main(cell_set: list = None, protocols: list = ['FreqSweep']) -> None:
#     all_cell_data = []

#     if cell_set is None:
#         cell_set = all_cells.all_cells

#     for protocol in protocols:
#         all_cell_data_ = []
#         x = 0
#         for cell in cell_set:
#             print(cell)
#             cellpath = all_cells.project_path_root / cell
#             cellID = cellpath.stem
#             cellpickle = cellpath / (str(cellID) + ".pkl")
#             print(cellpickle)

#             try:
#                 neuron = ephys_classes.Neuron.loadCell(cellpickle)
#                 neuron_data = neuron.data[protocol]
#                 if neuron_data is None:
#                     continue
#             except Exception as err:
#                 print(err)
#                 print("Error loading cell: ", cellpickle)
#                 continue
            
#             all_cell_data_.append(neuron_data)
#             x += neuron_data.shape[0]
            
#         if len(all_cell_data_) != 0:
#             print(f'{protocol} has {len(all_cell_data_)} cells with total {x} sweeps')
#             all_expt_ = pd.concat(all_cell_data_, ignore_index=True, axis=0)

#             # add analysis params to the df
#             print("adding analysis params to the df")
#             df_short, df_long = expt_to_dataframe.add_analysed_params2(all_expt_)
#             del all_expt_

#             # save  dfs
#             print("return from expt_to_dataframe. Saving DFs")
#             save_df_to_h5(df_short, filename_suffix='short', protocol=protocol, save_combined_also=True)
#             del df_short
#             print("Saving the large DF now")
#             save_df_to_h5(df_long, filename_suffix='long', protocol=protocol, save_combined_also=False)

### copilot version
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
    protocols = ['grid', 'LTMRand', 'surprise', 'convergence', 'SpikeTrain']#'FreqSweep'
    
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


'''
Code for combining all the protocols into one dataframe
# Frequency Sweep Protocol - Current Clamp
freq_sweep_datapath =  r"C:\Users\adity\OneDrive\NCBS\Lab\Projects\EI_Dynamics\Analysis\parsed_data\all_cells_FreqSweep_CC_short.h5"
# other protocol data paths: spiketrain, grid, LTMRand, surprise
spiketrain_datapath = r"C:\Users\adity\OneDrive\NCBS\Lab\Projects\EI_Dynamics\Analysis\parsed_data\all_cells_Spiketrain_CC_short.h5"
grid_datapath       = r"C:\Users\adity\OneDrive\NCBS\Lab\Projects\EI_Dynamics\Analysis\parsed_data\all_cells_Grid_CC_short.h5"
LTMRand_datapath    = r"C:\Users\adity\OneDrive\NCBS\Lab\Projects\EI_Dynamics\Analysis\parsed_data\all_cells_LTMRand_CC_short.h5"
surprise_datapath   = r"C:\Users\adity\OneDrive\NCBS\Lab\Projects\EI_Dynamics\Analysis\parsed_data\all_cells_Surprise_CC_short.h5"

# all all the h5 files into a list of dataframes and concatenate them
df1 = pd.read_hdf(freq_sweep_datapath, key='data')
df2 = pd.read_hdf(spiketrain_datapath, key='data')
df3 = pd.read_hdf(grid_datapath, key='data')
df4 = pd.read_hdf(LTMRand_datapath, key='data')
df5 = pd.read_hdf(surprise_datapath, key='data')
df_short_all_CC = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
# del the individual dataframes as they are not needed anymore
del df1, df2, df3, df4, df5
print('df_short_all_CC has {} sweeps'.format(len(df_short_all_CC)))
# make df2 an alias of df_short_all_CC
df2 = df_short_all_CC
# df2 = df2[ pd.notnull(df2['peaks_cell'])] # remove all sweeps that had NaNs in analysed params (mostly due to bad pulse detection)
# df2 = df2.reset_index(drop=True)

allprotocol_600ms_datapath =  r"parsed_data\all_cells_allprotocols_4channels_combined_600ms.h5"
dfcut = pd.read_hdf(allprotocol_600ms_datapath, key='data')
#print size of df
print("Size of dataframe: ", dfcut.shape)
# dfcut = dfcut[ pd.notnull(dfcut['peaks_cell'])] # remove all sweeps that had NaNs in analysed params (mostly due to bad pulse detection)
# reset index
# dfcut = dfcut.reset_index(drop=True)
# print new size of df
print("Size of dataframe: ", dfcut.shape)


----------------
# in the dataframe df_cut, each row has 36049 columns (for 600ms). first 49 columns are metadata, next 12000 columns are the 600ms data for each channel: cell, stim, and field.
# for each row, there is a metadata column value called 'probePulseStart'. Multiplying that value with sampling frequency Fs=2e4 will give the data point at which the pulse starts.
# from that datapoint, in each row, for each channel, i want to find the max, min, area-under-the-curve, and the time-to-peak. I want to then store these values in new columns in the dataframe.
# del df
#let's start
# new_cols = ['cell_fpr_max', 'cell_fpr_min', 'cell_fpr_auc', 'cell_fpr_ttp', 'cell_fpr_p2p', 'field_fpr_max', 'field_fpr_min', 'field_fpr_auc', 'field_fpr_ttp','field_fpr_p2p' ]
# # set type to be float32 and fill with zeros
# for new_col in new_cols:
#     dfcut[new_col] = 0.0
#     dfcut[new_col] = dfcut[new_col].astype('float32')
Fs = 2e4
# for each row, for each channel, find the max, min, area-under-the-curve, and the time-to-peak. I want to then store these values in new columns in the dataframe.

for i in range(dfcut.shape[0]):
    row = dfcut.iloc[i,:]
    #print dataframe index of the row being processed
    ind = row.name
    # print(ind, row['cellID'], row['exptID'], row['sweep'])
    pps = int(row['probePulseStart']*Fs)
    clamp = row['clampMode']
    clampPot = row['clampPotential']
    if row['probePulseStart'] == row['pulseTrainStart']:
        ipi = int(Fs/row['stimFreq'])
    else:
        ipi = 2000
    [cell, stim, field] = np.reshape(row.iloc[49:36049], (3, -1))
    # baseline subtract
    cell = cell - np.mean(cell[:pps])
    field = field - np.mean(field[:pps])
    stim = stim - np.mean(stim[:pps])
    # flip cell if it is VC and clamp potential is -70mV
    if (clamp == 'VC') & (clampPot == -70):
        cell *= -1
    # calculate parameters for each channel and store in the new column
    dfcut.loc[ind,'cell_fpr_max'] = (np.max(cell[pps:pps+ipi]) ).astype('float32')
    dfcut.loc[ind,'cell_fpr_min'] = (np.min(cell[pps:pps+ipi])).astype('float32')
    dfcut.loc[ind,'cell_fpr_auc'] = (np.trapz(cell[pps:pps+ipi])).astype('float32')
    dfcut.loc[ind,'cell_fpr_ttp'] = (np.argmax(cell[pps:pps+ipi])).astype('float32')
    dfcut.loc[ind,'cell_fpr_p2p'] = (np.max(cell[pps:pps+ipi]) - np.min(cell[pps:pps+ipi])).astype('float32')
    dfcut.loc[ind,'field_fpr_max'] = (np.max(field[pps:pps+ipi])).astype('float32')
    dfcut.loc[ind,'field_fpr_min'] = (np.min(field[pps:pps+ipi])).astype('float32')
    dfcut.loc[ind,'field_fpr_auc'] = (np.trapz(field[pps:pps+ipi])).astype('float32')
    # if field min is bigger in amplitude than field max, then get the time to peak as the time to the field min
    if np.abs(np.min(field[pps:pps+ipi])) > np.abs(np.max(field[pps:pps+ipi])):
        dfcut.loc[ind,'field_fpr_ttp'] = (np.argmax(-1*field[pps:pps+ipi])).astype('float32')
    else:
        dfcut.loc[ind,'field_fpr_ttp'] = (np.argmax(field[pps:pps+ipi])).astype('float32')
    dfcut.loc[ind,'field_fpr_p2p'] = (np.max(field[pps:pps+ipi]) - np.min(field[pps:pps+ipi])).astype('float32')
    # print(dfcut.loc[ind, ['cellID', 'exptID', 'sweep']])

#save dfcut as a new file
# dfcutshort = pd.concat([dfcut.iloc[:,:49], dfcut.iloc[:,36049:]], axis=1)
# print(dfcutshort.columns)
dfcutshort.to_hdf(r"C:\Users\adity\OneDrive\NCBS\Lab\Projects\EI_Dynamics\Analysis\parsed_data\all_cells_allprotocols_with_fpr_values.h5", key='data', mode='w')



# i want to go through all the files in the folder and subfolders under cell_data_path that match the filename "_rec.abf" and open using abf_to_data

unitdf = pd.DataFrame(columns=['cellID', 'exptID','numChannels','cellunit', 'fieldunit'])
x = list(cell_data_path.glob('**/*_rec.abf'))
print(len(x))
for i,abffile in enumerate(x):
    print(i/367, abffile)
    filename        = pathlib.Path(abffile).name[:15]
    try:
        abf             = pyabf.ABF(abffile)
        numChannels     = abf.channelCount
        channelList     = abf.channelList
        if numChannels==4:
            fieldunit       = abf.adcUnits[3]
        elif numChannels==3:
            fieldunit = ""
        cellunit = abf.adcUnits[0]
        
    except:
        print("Error in file: ", abffile)
        cellunit, fieldunit = "", ""

    unitdf.loc[i]   = {'cellID':int(abffile.parent.stem),  'exptID':int(abffile.name[-12:-8]), 'numChannels':numChannels,'cellunit':cellunit, 'fieldunit':fieldunit}

dfshortpath = r"C:\Users\adity\OneDrive\NCBS\Lab\Projects\EI_Dynamics\Analysis\parsed_data\all_cells_allprotocols_with_fpr_values.h5"
dfmerge = pd.merge(df2, unitdf, on=['cellID', 'exptID'], how='left')
# save at dfshortpath
dfmerge.to_hdf(dfshortpath, key='data', mode='w')

'''