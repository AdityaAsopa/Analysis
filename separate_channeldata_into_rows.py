
import numpy as np
import pandas as pd
# suppress warnings
import warnings
warnings.filterwarnings("ignore")

def separate_channel_save_hdf(df2, cellname):
    print(f'processing {cellname}')
    
    df2 = df2[ df2['cellID']==cellname ]
    print(df2.shape)
    
    melt_by = np.linspace(0, 79999,80000).astype(int)
    ids = ['cellID', 'exptID','stimFreq', 'clampPotential', 'condition', 'numSq', 'patternList','sweep']
    dff = pd.melt(df2, id_vars=ids, value_vars=melt_by, var_name='datapoint', value_name='current')

    # create a new column called 'channel' in dff and set it to 'cell' if datapoint is between 0 and 19999, 'frameTTL' if datapoint between 20000 and 39999, 'stim' if datapoint between 40000 and 59999, 'field' if datapoint between 60000 and 79999
    dff['channel'] = np.where(dff['datapoint']<20000, 'cell', np.where(dff['datapoint']<40000, 'frameTTL', np.where(dff['datapoint']<60000, 'stim', 'field')))
    # create a new column called 'time' in dff and set it to (datapoint % 20000)/20000
    dff['datapoint'] = dff['datapoint'] % 20000
    dff['patternList'] = dff['patternList'].astype(int)
    # pivot the dataframe back to original shape
    ids.append('channel')
    df_pivot = dff.pivot_table(index=ids,columns='datapoint', values='current').reset_index()

    print(df_pivot.shape)
    # save dataframe as hdf
    filename = str(cellname) + '_FreqSweep_channel_expanded.h5'
    df_pivot.to_hdf(filename, key='data')

def main(freq_sweep_datapath, clamp='VC'):
    df = pd.read_hdf(freq_sweep_datapath, key='data')
    
    # list of cc cells
    if clamp== 'VC':
        cell_list = np.array([7492, 7491, 6301, 6201, 1931, 1621, 1541, 1531, 1524, 1523, 1522, 1491, 111])
        df = df[ df['cellID'].isin(cell_list) ]
    elif clamp == 'CC':
        cell_list = df['cellID'].unique()
        df = df[ pd.notnull(df['peaks_cell'])]

    df = df.reset_index(drop=True)

    for cell in cell_list:
        separate_channel_save_hdf(df, cell)

    dfs = []
    size = 0
    # combine all the dataframes
    for cell in cell_list:
        print(f'processing {cell}')
        filename = str(cell) + '_FreqSweep_channel_expanded.h5'
        dfs.append(pd.read_hdf(filename, key='data'))
        size += dfs[-1].shape[0]
        print(size)

    dffs = pd.concat(dfs, ignore_index=True)
    # save
    filename = 'screened_cells_FreqSweep_' + clamp + '_long_channel_split_into_rows.h5'
    dffs.to_hdf(filename, key='data')
    print(dffs.shape)

if __name__ == "__main__":
    freq_sweep_datapath =  r"C:\Users\adity\OneDrive\NCBS\Lab\Projects\EI_Dynamics\Analysis\parsed_data\all_cells_FreqSweep_CC_long.h5" 
    main(freq_sweep_datapath, clamp='CC')
    freq_sweep_datapath =  r"C:\Users\adity\OneDrive\NCBS\Lab\Projects\EI_Dynamics\Analysis\parsed_data\all_cells_FreqSweep_VC_long.h5"
    main(freq_sweep_datapath, clamp='VC')