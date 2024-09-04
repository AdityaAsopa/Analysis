from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import psutil

print('starting', psutil.virtual_memory())
freq_sweep_datapath =  r"C:\Users\adity\OneDrive\NCBS\Lab\Projects\EI_Dynamics\Analysis\parsed_data\all_cells_FreqSweep_VC_long.h5" 
df = pd.read_hdf(freq_sweep_datapath, key='data')
df = df[ pd.notnull(df['peaks_cell'])] # remove all sweeps that had NaNs in analysed params (mostly due to bad pulse detection)
# reset index
df = df.reset_index(drop=True)

vc_cells = np.unique( df [ (df['clampMode']=='VC') ]['cellID'])
cc_cells = np.unique(df['cellID'])
vc_cells_screened = np.array([7492, 7491, 6301, 6201, 1931, 1621, ])# part 1
#vc_cells_screened = np.array([1541, 1531, 1524, 1523, 1522, 1491, 111])
print(vc_cells_screened)
df = df[ df['cellID'].isin(vc_cells_screened)]

print('melting now', '\n', psutil.virtual_memory())
melt_by = np.linspace(0, 79999,80000).astype(int)
dff = pd.melt(df, id_vars=['cellID', 'stimFreq', 'clampPotential', 'numSq', 'patternList','sweep'], value_vars=melt_by, var_name='datapoint', value_name='current')

print('melting done, deleting df', '\n', psutil.virtual_memory())
# create a new column called 'channel' in dff and set it to 'cell' if datapoint is between 0 and 19999, 'frameTTL' if datapoint between 20000 and 39999, 'stim' if datapoint between 40000 and 59999, 'field' if datapoint between 60000 and 79999
dff['channel'] = np.where(dff['datapoint']<20000, 'cell', np.where(dff['datapoint']<40000, 'frameTTL', np.where(dff['datapoint']<60000, 'stim', 'field')))
# create a new column called 'time' in dff and set it to (datapoint % 20000)/20000
dff['datapoint'] = dff['datapoint'] % 20000
dff['patternList'] = dff['patternList'].astype(int)
# pivot the dataframe back to original shape

print('pivoting now', '\n', psutil.virtual_memory())
df_pivot = dff.pivot_table(index=['cellID','stimFreq','clampPotential','numSq','sweep','patternList','channel'],columns='datapoint', values='current').reset_index()


print(df_pivot.shape)
print('saving', '\n', psutil.virtual_memory())
# save dataframe as hdf
filename='klvuvuivi_screened_cells_FreqSweep_VC_channelr_long_separated_part1gaugah.h5'
df_pivot.to_hdf(filename, key='data')

print('done', filename, '\n', psutil.virtual_memory())