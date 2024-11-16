import sys
import os
import importlib
from   pathlib      import Path
import traceback

import numpy                as np
import matplotlib           as mpl
import matplotlib.pyplot    as plt
import seaborn              as sns
import pandas               as pd

from scipy.stats    import kruskal, wilcoxon, mannwhitneyu, ranksums
# from scipy.signal   import filter_design
from scipy.optimize import curve_fit
from lmfit import Model, Parameters

from eidynamics     import utils, plot_tools
from eidynamics     import pattern_index
import all_cells
rollvar_baseline = utils.mean_at_least_rolling_variance

sns.set_context('paper')
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 16
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['lines.linewidth'] = 2

# make a colour map viridis
viridis = mpl.colormaps["viridis"]
flare   = mpl.colormaps["flare"]
crest   = mpl.colormaps["crest"]
magma   = mpl.colormaps["magma"]
edge    = mpl.colormaps['edge']

color_E = flare
color_I = crest
color_freq = {1:magma(0.05), 5:magma(0.1), 10:magma(0.2), 20:magma(.4), 30:magma(.5), 40:magma(.6), 50:magma(.7), 100:magma(.9)}
color_squares = color_squares = {1:viridis(0.2), 5:viridis(.4), 7:viridis(.6), 15:viridis(.8), 20:viridis(1.0)}
color_EI = {-70:flare(0), 0:crest(0)}
color_cells = mpl.colormaps['tab10']
Fs = 2e4

freq_sweep_pulses = np.arange(9)

# Load the data
figure_raw_material_location = Path(r"paper_figure_matter\\")
paper_figure_export_location = Path(r"paper_figures\\")
data_path_FS                 = Path(r"parsed_data\\FreqSweep\\")
data_path_LTM                 = Path(r"parsed_data\\LTMRand\\")
data_path_grid               = Path(r"parsed_data\\Grid\\")
data_path_analysed           = Path(r"parsed_data\\second_order\\")
raw_data_path_cellwise       = Path(r"C:\Users\adity\OneDrive\NCBS\Lab\Projects\EI_Dynamics\Data\Screened_cells\\")

def sdnfunc(expected, gamma):
    return gamma * expected / (gamma + expected)

def nosdn(expected, m):
    return m * expected

figure_raw_material_location = Path(r"paper_figure_matter\\")
paper_figure_export_location = Path(r"paper_figures\\Figure3v4\\")
data_path_FS                 = Path(r"parsed_data\\FreqSweep\\")
data_path_LTM                 = Path(r"parsed_data\\LTMRand\\")
data_path_grid               = Path(r"parsed_data\\Grid\\")
data_path_analysed           = Path(r"parsed_data\\second_order\\")
raw_data_path_cellwise       = Path(r"..\Data\Screened_cells\\")

### Load the CC data and screen
# Update September 2024
# September 2024
# short data path that contains the kernel fit data for FreqSweep protocol, also contains the field p2p data. latest and checked. Use this for all freqsweep measurements.
# Contains screening parameters also.
# 18Sep24
CC_FS_shortdf_withkernelfit_datapath = data_path_FS / "all_cells_FreqSweep_CC_kernelfit_response_measurements.h5"
cc_FS_shortdf = pd.read_hdf(CC_FS_shortdf_withkernelfit_datapath, key='data')
print(cc_FS_shortdf.shape)

CC_LTM_shortdf_withkernelfit_datapath = data_path_LTM / "all_cells_LTM_CC_kernelfit_response_measurements_noNANs.h5"
cc_LTM_shortdf = pd.read_hdf(CC_LTM_shortdf_withkernelfit_datapath, key='data')
print(cc_LTM_shortdf.shape)

cc_FS_LTM_shortdf = pd.concat([cc_FS_shortdf, cc_LTM_shortdf], axis=0, ignore_index=True)
# reset index
cc_FS_LTM_shortdf.reset_index(drop=True, inplace=True)

# Data screening
# CC data screening based on dataflag_fields: protocol freqsweep
cc_FS_LTM_shortdf_slice = cc_FS_LTM_shortdf[
                                            (cc_FS_LTM_shortdf['location'] == 'CA1') &
                                            (cc_FS_LTM_shortdf['numSq'].isin([1,5,7,15])) &
                                            (cc_FS_LTM_shortdf['stimFreq'].isin([20,30,40,50])) &
                                            (cc_FS_LTM_shortdf['condition'] == 'Control') &
                                            (cc_FS_LTM_shortdf['ch0_response']==1) &
                                            (cc_FS_LTM_shortdf['IR'] >50) & (cc_FS_LTM_shortdf['IR'] < 400) &
                                            (cc_FS_LTM_shortdf['tau'] < 40) & 
                                            # (cc_FS_LTM_shortdf['probePulseStart']==0.2) &
                                            # (cc_FS_LTM_shortdf['intensity']==100) &
                                            # (cc_FS_LTM_shortdf['pulseWidth']==2) &
                                            (cc_FS_LTM_shortdf['spike_in_baseline_period'] == 0) &
                                            (cc_FS_LTM_shortdf['ac_noise_power_in_ch0'] < 40)
                                            ]
print(cc_FS_LTM_shortdf.shape, '--screened-->', cc_FS_LTM_shortdf_slice.shape)
screened_cc_trialIDs = cc_FS_LTM_shortdf_slice['trialID'].unique()

# save trial IDs as a numpy array text file, all trialID are strings
np.savetxt(paper_figure_export_location / "Figure3_screened_trialIDs_CC_FS_LTM.txt", screened_cc_trialIDs, fmt='%s')

cc_FS_LTM_shortdf_slice['patternList'] = cc_FS_LTM_shortdf_slice['patternList'].astype('int32')
patternIDs = np.sort( cc_FS_LTM_shortdf_slice[cc_FS_LTM_shortdf_slice['numSq'] != 1]['patternList'].unique() )

print(f"Unique cells in screened data: { cc_FS_LTM_shortdf_slice['cellID'].nunique()}")
print(f"Unique sweeps in screened data: {cc_FS_LTM_shortdf_slice['trialID'].nunique()}")

# # take list stored in "peaks_field_norm" column and make it into new columns
cc_FS_LTM_shortdf_slice = utils.expand_list_column(cc_FS_LTM_shortdf_slice, 'peaks_field_norm', 'pfn_')

VC_FS_shortdf_withkernelfit_datapath = data_path_FS / "all_cells_FreqSweep_VC_kernelfit_response_measurements.h5"
vc_FS_shortdf = pd.read_hdf(VC_FS_shortdf_withkernelfit_datapath, key='data')
print(vc_FS_shortdf.shape)

# save df
vc_FS_shortdf.to_hdf(VC_FS_shortdf_withkernelfit_datapath, key='data', mode='w')
# VC data screening
# VC data screening based on dataflag_fields
vc_FS_shortdf_slice = vc_FS_shortdf[
            (vc_FS_shortdf['location'] == 'CA1') &
            (vc_FS_shortdf['numSq'].isin([1,5,15])) &
            (vc_FS_shortdf['stimFreq'].isin([20,30,40,50])) &
            (vc_FS_shortdf['condition'] == 'Control') &
            (vc_FS_shortdf['ch0_response']==1) &
            # (vc_FS_shortdf['intensity'] == 100) &
            # (vc_FS_shortdf['pulseWidth'] == 2) &
            # (vc_FS_shortdf['probePulseStart']==0.2) &
            (vc_FS_shortdf['IR'] >40) & (vc_FS_shortdf['IR'] < 400) &
            (vc_FS_shortdf['tau'] < 40) & 
            (vc_FS_shortdf['ac_noise_power_in_ch0'] < 40)&
            (vc_FS_shortdf['valley_0'].notnull())
        ]
print(vc_FS_shortdf.shape, '--screened-->', vc_FS_shortdf_slice.shape)
screened_vc_trialIDs = vc_FS_shortdf_slice['trialID'].unique()
np.savetxt(paper_figure_export_location / "Figure3_screened_trialIDs_VC_FS.txt", screened_vc_trialIDs, fmt='%s')

print(f"Unique cells in screened data: { vc_FS_shortdf_slice['cellID'].nunique()}")
print(f"Unique sweeps in screened data: {vc_FS_shortdf_slice['trialID'].nunique()}")

# take list stored in "peaks_field_norm" column and make it into new columns
# vc_FS_shortdf_slice = utils.expand_list_column(vc_FS_shortdf_slice, 'pulseTimes', 'stimOnset_')

print('## Loading SDN and fit data')
sdn_df = pd.read_hdf(paper_figure_export_location / "Figure3_sdn_data_FS_LTM.h5", key='data')
print(sdn_df.shape)

fitdf = pd.read_hdf(paper_figure_export_location / "Figure3_gamma_and_slope_fits_FS_LTM.h5", key='data')
print(fitdf.shape)

cc_delay_df = pd.read_hdf(paper_figure_export_location / "Figure3_delay_df_CC_FS.h5", key='data')
vc_delay_df = pd.read_hdf(paper_figure_export_location / "Figure3_delay_df_VC_FS.h5", key='data')
ebyi_df     = pd.read_hdf(    paper_figure_export_location / "Figure3_ebyi_df_VC_FS.h5" , key='data')


def calculate_expected_response(celldf, pulse_index, freq, patternID,):
    """
    Calculate the expected response of a pattern based on the response to individual spots in the pattern
    """
    from eidynamics import pattern_index
    # constants
    Fs      = 2e4
    cellID  = celldf['cellID'].iloc[0]
    
    # checks
    field_data=True if celldf['numChannels'].iloc[0] == 4 else False

    # check if the given cell has 1sq data
    if not 1 in celldf['numSq'].unique():
        # print('No 1Sq data for this cell', celldf['numSq'].unique())
        # generate dataerror to be caught by the calling function
        raise ValueError(f'Cell: {cellID} - No 1Sq data for this cell. {pulse_index}, {freq}, {patternID}')
    # data
    pattern_response_df             = celldf[(celldf['patternList'] == patternID) & (celldf['stimFreq'] == freq)  ]
    if pattern_response_df.shape[0] == 0:
        raise ValueError(f'Cell: {cellID} - No data for this pattern {patternID} and freq {freq} Hz')
    
    # get the pattern
    constituent_spots_of_pattern    = pattern_index.get_patternIDlist_for_nSq_pattern(patternID) #1sq spots that make the pattern in the patternID
    numSq                           = len(constituent_spots_of_pattern)

    obs_col = 'PSC_' + str(pulse_index)
    obs_col_field = 'pfn_' + str(pulse_index)

    # # slice the dataframe to get the response to the given pattern
    celldf                          = celldf.loc[:, ~celldf.columns.isin(celldf.columns[28:49])]
    celldf.loc[:, 'patternList']    = celldf['patternList'].astype('int32')
    
    # step 0: get the observed response from the pattern_response_df
    observed_response_cell      = pattern_response_df.loc[:, obs_col].values
    if field_data:
        observed_response_field     = pattern_response_df.loc[:, obs_col_field].values
        observed_response_scaled    = observed_response_cell / observed_response_field
    else:
        observed_response_scaled    = observed_response_cell * np.nan
    
    # expected response calculation
    # step 1: slice the dataframe to get only those rows where 'patternList' is in the list 'constituent_spots_of_pattern'
    df1sq = celldf.loc[celldf['patternList'].isin(constituent_spots_of_pattern), :].copy()
    
    # step 2: get the peaks for each row between columns probePulseStart and probePulseStart+ipi
    # here i am taking the mean of all the trials of the constituent patterns and then summing those means
    expected_response = df1sq.loc[:,('patternList','PSC_0')].groupby(by='patternList').mean().sum()['PSC_0']
    
    return numSq, freq, patternID, pulse_index, field_data, observed_response_cell, observed_response_scaled, expected_response

def sdn_fits(xdata, ydata, f,s,t):
    # # if xdata and ydata lenghts are not same, 
    if len(xdata) != len(ydata):
        raise ValueError('Length of xdata and ydata are not same')
        return np.nan, np.nan, np.nan, np.nan
        
    # if xdata or yadata is empty or have length 0, 
    if len(xdata) <3 or len(ydata) <3:
        return np.nan, np.nan, np.nan, np.nan
        
    # Create an lmfit model for the sdnfunc
    model = Model(sdnfunc)
    # Create a set of parameters
    params = Parameters()
    params.add('gamma', value=5)

    # Create an lmfit model for the data
    model_linear = Model(nosdn)
    # Create a set of parameters
    params_linear = Parameters()
    params_linear.add('m', value=1)

    # Fit the sdnfunc to  data using lmfit and method = cobyla
    result = model.fit(ydata, params, observed=xdata, method='cobyla')

    # also try fitting xdata and ydata to a linear model
    result_linear = model_linear.fit(ydata, params_linear, observed=xdata, method='cobyla')

    # Extract the fitted parameters
    fitted_gamma = np.round( result.best_values['gamma'], 3)
    fitted_slope = np.round( result_linear.best_values['m'], 3)

    return fitted_gamma, fitted_slope, result.rsquared, result_linear.rsquared

def gamma_distribution(df_sdn, fitdf=None, x='expected_response', y='observed_response', first='cellID', second='pulse_index', third='freq'):
    gamma_dist = []

    # get gamma distribution for the entire dataset
    dfslice = df_sdn.dropna(subset=[x,y])
    g,m,r2g,r2lin = sdn_fits(dfslice[x], dfslice[y], 'all','all','all')
    gamma_dist.append({'expected': x, 'observed':y, first:1000, second:1000, third:1000, 'sample_size':dfslice.shape[0], 'gamma':g, 'slope':m, 'r2_sdn':r2g, 'r2_lin':r2lin})
    f,s,t = np.nan, np.nan, np.nan

    for f in np.sort(df_sdn[first].unique()):
        dfslice = df_sdn[(df_sdn[first] == f)]
        # remove nan and inf from data
        dfslice = dfslice[(np.abs(dfslice[x]) != np.inf) & (np.abs(dfslice[y]) != np.inf)].dropna(subset=[x,y])
        # if most of x and y data is 0, the model will not converge, so we need to remove all zero entries
        dfslice = dfslice[(dfslice[x] != 0) & (dfslice[y] != 0)]
        
        g,m,r2g,r2lin = sdn_fits(dfslice[x], dfslice[y], f,'all','all')
        gamma_dist.append({'expected': x, 'observed':y, first:f, second:1000, third:1000, 'sample_size':dfslice.shape[0], 'gamma':g, 'slope':m, 'r2_sdn':r2g, 'r2_lin':r2lin})
        
        for s in np.sort(df_sdn[second].unique()):
            dfslice = df_sdn[(df_sdn[first] == f) & (df_sdn[second] == s)].dropna(subset=[x,y])
            # remove np.inf from data
            dfslice = dfslice[(np.abs(dfslice[x]) != np.inf) & (np.abs(dfslice[y]) != np.inf)].dropna(subset=[x,y])
            # if most of x and y data is 0, the model will not converge, so we need to remove all zero entries
            dfslice = dfslice[(dfslice[x] != 0) & (dfslice[y] != 0)]
            
            g,m,r2g,r2lin = sdn_fits(dfslice[x], dfslice[y], f,s,'all')
            gamma_dist.append({'expected': x, 'observed':y, first:f, second:s, third:1000, 'sample_size':dfslice.shape[0], 'gamma':g, 'slope':m, 'r2_sdn':r2g, 'r2_lin':r2lin})
            
            for t in np.sort(df_sdn[third].unique()):
                dfslice = df_sdn[(df_sdn[first] == f) & (df_sdn[second] == s) & (df_sdn[third] == t)].dropna(subset=[x,y])
                # remove nan
                # remove np.inf from data
                dfslice = dfslice[(np.abs(dfslice[x]) != np.inf) & (np.abs(dfslice[y]) != np.inf)].dropna(subset=[x,y])
                # if most of x and y data is 0, the model will not converge, so we need to remove all zero entries
                dfslice = dfslice[(dfslice[x] != 0) & (dfslice[y] != 0)]
                g,m,r2g,r2lin = sdn_fits(dfslice[x], dfslice[y], f,s,t)
                gamma_dist.append({'expected': x, 'observed':y, first:f, second:s, third:t, 'sample_size':dfslice.shape[0], 'gamma':g, 'slope':m, 'r2_sdn':r2g, 'r2_lin':r2lin})


    # create a dataframe from the list of dicts
    df_gamma_dist = pd.DataFrame(gamma_dist)

    if fitdf is not None:
        fitdf2 = pd.concat([fitdf, df_gamma_dist], axis=0)
    else:
        fitdf2 = df_gamma_dist

    return fitdf2

def generate_cc_delay_df(cc_short_df):
    idvars = ['cellID','stimFreq','numSq','patternList','pulseWidth','intensity','trialID']

    valvars2a = [f'peakdelay_{i}' for i in freq_sweep_pulses]
    valvars2b = [f'onsetdelay_{i}' for i in freq_sweep_pulses]

    df2a = cc_short_df.melt(id_vars=idvars, 
                value_vars=valvars2a, 
                var_name='peak', value_name='peak_delay')

    df2b = cc_short_df.melt(id_vars=idvars, 
                value_vars=valvars2b, 
                var_name='onset', value_name='onset_delay')

    df2a['pulse'] = df2a['peak' ].apply(lambda x: int(x.split('_')[-1]))
    df2b['pulse'] = df2b['onset'].apply(lambda x: int(x.split('_')[-1]))

    # # concat df1 and df2 on axis1
    cc_delay_df  = pd.merge(df2a,  df2b, on=['cellID','stimFreq','numSq','patternList','pulseWidth','intensity','trialID','pulse'], )
    # print('after merger: ', cc_delay_df.shape)

    # # remove those trials for which peak_delay and onset_delay are negative
    cc_delay_df.drop(columns=['peak','onset'], inplace=True)
    cc_delay_df = cc_delay_df[(cc_delay_df['peak_delay']>0) &(cc_delay_df['peak_delay']<0.05) ]
    cc_delay_df = cc_delay_df[(cc_delay_df['onset_delay']>0)&(cc_delay_df['onset_delay']<0.05) ]
    cc_delay_df = cc_delay_df[cc_delay_df['peak_delay'] > cc_delay_df['onset_delay']]
    # print('after removing negative onset to peak times: ', cc_delay_df.shape)

    # cc_delay_df.drop(columns=['trialID'], inplace=True)
    # # drop NaNs in time_to_peak
    cc_delay_df = cc_delay_df.dropna(subset=['peak_delay','onset_delay'])
    # print('merged and peak and onset calculated df: ', cc_delay_df.shape)
    columnscc = ['peak_delay','onset_delay']
    cc_delay_df[columnscc] = cc_delay_df[columnscc] * 1000

    print(cc_delay_df.shape)
    return cc_delay_df

def generate_vc_delay_df(vc_shortdf):    
    idvars = ['cellID','clampPotential','stimFreq','numSq','patternList','pulseWidth','intensity','trialID']

    valvars2a = [f'peakdelay_{i}' for i in freq_sweep_pulses]
    valvars2b = [f'onsetdelay_{i}' for i in freq_sweep_pulses]

    df2a = vc_shortdf.melt(id_vars=idvars, 
                value_vars=valvars2a, 
                var_name='peak', value_name='peak_delay')

    df2b = vc_shortdf.melt(id_vars=idvars, 
                value_vars=valvars2b, 
                var_name='onset', value_name='onset_delay')

    df2a['pulse'] = df2a['peak' ].apply(lambda x: int(x.split('_')[-1]))
    df2b['pulse'] = df2b['onset'].apply(lambda x: int(x.split('_')[-1]))

    # concat df1 and df2 on axis1
    df3  = pd.merge(df2a,  df2b, on=['cellID','clampPotential','stimFreq','numSq','patternList','pulseWidth','intensity','trialID','pulse'], )

    # remove those trials from df3 where time to peak is smaller than time to valley
    # remove those trials for which peak_onset and valley_onset are negative
    df3.drop(columns=['trialID'], inplace=True)
    df3.drop(columns=['onset','peak'], inplace=True)
    df3 = df3[(df3['peak_delay']>0) &(df3['peak_delay']<0.05)]
    df3 = df3[(df3['onset_delay']>0)&(df3['onset_delay']<0.05)]
    df3 = df3[ df3['peak_delay']    > df3['onset_delay']]
    # # drop NaNs in time_to_peak
    df3 = df3.dropna(subset=['peak_delay','onset_delay'])

    df4 = df3.groupby(['cellID','clampPotential','stimFreq','numSq','patternList','pulseWidth','intensity','pulse']).median().reset_index()
    # # pivot w.r.t clampPotential
    peak_delay_df  = df4.pivot(index=['cellID','stimFreq','numSq','patternList','pulseWidth','intensity','pulse'], columns='clampPotential', values='peak_delay').reset_index()
    onset_delay_df = df4.pivot(index=['cellID','stimFreq','numSq','patternList','pulseWidth','intensity','pulse'], columns='clampPotential', values='onset_delay').reset_index()

    # drop NaNs from df4pivot from columns -70 and 0
    peak_delay_df  = peak_delay_df.dropna(subset=[-70,0])
    onset_delay_df = onset_delay_df.dropna(subset=[-70,0])

    # # subtract -70 from 0
    peak_delay_df['peak_delayEI']   = (peak_delay_df[0]  - peak_delay_df[-70] )
    onset_delay_df['onset_delayEI'] = (onset_delay_df[0] - onset_delay_df[-70])

    # # rename -70 and 0 columns to exc_onset and inh_onset
    peak_delay_df.rename(columns={-70:'exc_peak', 0:'inh_peak'}, inplace=True)
    onset_delay_df.rename(columns={-70:'exc_onset', 0:'inh_onset'}, inplace=True)

    # # merge the two
    vc_delay_df = pd.merge(peak_delay_df, onset_delay_df, on=['cellID','stimFreq','numSq','patternList','pulseWidth','intensity','pulse'])
    # remove those rows where the onset delay is more than 20 or less than -20
    vc_delay_df = vc_delay_df[(vc_delay_df['onset_delayEI'] < 20) & (vc_delay_df['onset_delayEI'] > -20)]
    # multiply the delay by 1000 to convert to ms: following columns: 'exc_peak','inh_peak','peak_delayEI','exc_onset','inh_onset','onset_delayEI'
    columnsvc = ['exc_peak','inh_peak','peak_delayEI','exc_onset','inh_onset','onset_delayEI']
    vc_delay_df[columnsvc] = vc_delay_df[columnsvc] * 1000
    print(vc_delay_df.shape)
    return vc_delay_df

def generate_ebyi_df(vc_shortdf):
    idvars = ['cellID','clampPotential','stimFreq','numSq','patternList','pulseWidth','intensity','trialID']
    valvars = [f'PSC_{i}' for i in freq_sweep_pulses]
    df2 = vc_shortdf.melt(id_vars=idvars, 
                value_vars=valvars, 
                var_name='pulse', value_name='PSC')

    # if clampPotential=-70, remove rows with positive PSC values
    df2 = df2[((df2['clampPotential']==-70) & (df2['PSC']<0)) | ((df2['clampPotential']==0) & (df2['PSC']>0))]

    df2['pulse'] = df2['pulse'].apply(lambda x: int(x.split('_')[-1]))
    df2 = df2.dropna(subset=['PSC'])
    df2.drop(columns=['trialID'], inplace=True)
    df2['numSq'] = df2['numSq'].astype('int')
    df4 = df2.groupby(['cellID','clampPotential','stimFreq','numSq','patternList','pulseWidth','intensity','pulse']).mean().reset_index()
    ebyi_df = df4.pivot(index=['cellID','stimFreq','numSq','patternList','pulseWidth','intensity','pulse'], columns='clampPotential', values='PSC').reset_index()
    ebyi_df = ebyi_df.dropna(subset=[-70,0])
    # ratio of -70 and 0
    ebyi_df['EbyI'] = ( - ebyi_df[-70] / ebyi_df[0])
    ebyi_df = ebyi_df[(ebyi_df['EbyI'] < 20) & (ebyi_df['EbyI'] > 0)]

    print(ebyi_df.shape)
    return ebyi_df

# make dataset instead of loading
def make_dataset():
    # Update September 2024
    # September 2024
    # short data path that contains the kernel fit data for FreqSweep protocol, also contains the field p2p data. latest and checked. Use this for all freqsweep measurements.
    # Contains screening parameters also.
    # 18Sep24
    CC_FS_shortdf_withkernelfit_datapath = data_path_FS / "all_cells_FreqSweep_CC_kernelfit_response_measurements.h5"
    cc_FS_shortdf = pd.read_hdf(CC_FS_shortdf_withkernelfit_datapath, key='data')
    print(cc_FS_shortdf.shape)

    CC_LTM_shortdf_withkernelfit_datapath = data_path_LTM / "all_cells_LTM_CC_kernelfit_response_measurements_noNANs.h5"
    cc_LTM_shortdf = pd.read_hdf(CC_LTM_shortdf_withkernelfit_datapath, key='data')
    print(cc_LTM_shortdf.shape)

    cc_FS_LTM_shortdf = pd.concat([cc_FS_shortdf, cc_LTM_shortdf], axis=0, ignore_index=True)
    # reset index
    cc_FS_LTM_shortdf.reset_index(drop=True, inplace=True)

    # Data screening
    # CC data screening based on dataflag_fields: protocol freqsweep
    cc_FS_LTM_shortdf_slice = cc_FS_LTM_shortdf[
                                                (cc_FS_LTM_shortdf['location'] == 'CA1') &
                                                (cc_FS_LTM_shortdf['numSq'].isin([1,5,7,15])) &
                                                (cc_FS_LTM_shortdf['stimFreq'].isin([20,30,40,50])) &
                                                (cc_FS_LTM_shortdf['condition'] == 'Control') &
                                                (cc_FS_LTM_shortdf['ch0_response']==1) &
                                                (cc_FS_LTM_shortdf['IR'] >50) & (cc_FS_LTM_shortdf['IR'] < 400) &
                                                (cc_FS_LTM_shortdf['tau'] < 40) & 
                                                # (cc_FS_LTM_shortdf['probePulseStart']==0.2) &
                                                # (cc_FS_LTM_shortdf['intensity']==100) &
                                                # (cc_FS_LTM_shortdf['pulseWidth']==2) &
                                                (cc_FS_LTM_shortdf['spike_in_baseline_period'] == 0) &
                                                (cc_FS_LTM_shortdf['ac_noise_power_in_ch0'] < 40)
                                                ]
    print(cc_FS_LTM_shortdf.shape, '--screened-->', cc_FS_LTM_shortdf_slice.shape)
    screened_cc_trialIDs = cc_FS_LTM_shortdf_slice['trialID'].unique()

    # save trial IDs as a numpy array text file, all trialID are strings
    np.savetxt(paper_figure_export_location / "Figure3_screened_trialIDs_CC_FS_LTM.txt", screened_cc_trialIDs, fmt='%s')

    cc_FS_LTM_shortdf_slice['patternList'] = cc_FS_LTM_shortdf_slice['patternList'].astype('int32')
    patternIDs = np.sort( cc_FS_LTM_shortdf_slice[cc_FS_LTM_shortdf_slice['numSq'] != 1]['patternList'].unique() )

    print(f"Unique cells in screened data: { cc_FS_LTM_shortdf_slice['cellID'].nunique()}")
    print(f"Unique sweeps in screened data: {cc_FS_LTM_shortdf_slice['trialID'].nunique()}")

    # # take list stored in "peaks_field_norm" column and make it into new columns
    cc_FS_LTM_shortdf_slice = utils.expand_list_column(cc_FS_LTM_shortdf_slice, 'peaks_field_norm', 'pfn_')

    # VC data

    VC_FS_shortdf_withkernelfit_datapath = data_path_FS / "all_cells_FreqSweep_VC_kernelfit_response_measurements.h5"
    vc_FS_shortdf = pd.read_hdf(VC_FS_shortdf_withkernelfit_datapath, key='data')
    print(vc_FS_shortdf.shape)

    # save df
    vc_FS_shortdf.to_hdf(VC_FS_shortdf_withkernelfit_datapath, key='data', mode='w')
    # VC data screening
    # VC data screening based on dataflag_fields
    vc_FS_shortdf_slice = vc_FS_shortdf[
                (vc_FS_shortdf['location'] == 'CA1') &
                (vc_FS_shortdf['numSq'].isin([1,5,15])) &
                (vc_FS_shortdf['stimFreq'].isin([20,30,40,50])) &
                (vc_FS_shortdf['condition'] == 'Control') &
                (vc_FS_shortdf['ch0_response']==1) &
                # (vc_FS_shortdf['intensity'] == 100) &
                # (vc_FS_shortdf['pulseWidth'] == 2) &
                # (vc_FS_shortdf['probePulseStart']==0.2) &
                (vc_FS_shortdf['IR'] >40) & (vc_FS_shortdf['IR'] < 400) &
                (vc_FS_shortdf['tau'] < 40) & 
                (vc_FS_shortdf['ac_noise_power_in_ch0'] < 40)&
                (vc_FS_shortdf['valley_0'].notnull())
            ]
    print(vc_FS_shortdf.shape, '--screened-->', vc_FS_shortdf_slice.shape)
    screened_vc_trialIDs = vc_FS_shortdf_slice['trialID'].unique()
    np.savetxt(paper_figure_export_location / "Figure3_screened_trialIDs_VC_FS.txt", screened_vc_trialIDs, fmt='%s')

    print(f"Unique cells in screened data: { vc_FS_shortdf_slice['cellID'].nunique()}")
    print(f"Unique sweeps in screened data: {vc_FS_shortdf_slice['trialID'].nunique()}")

    # take list stored in "peaks_field_norm" column and make it into new columns
    # vc_FS_shortdf_slice = utils.expand_list_column(vc_FS_shortdf_slice, 'pulseTimes', 'stimOnset_')

    # generate sdn data
    sdn_data = []

    for cell in np.sort(cc_FS_LTM_shortdf_slice['cellID'].unique()):
        for patternID in patternIDs:
            for freq in [20, 30, 40, 50]:
                celldf = cc_FS_LTM_shortdf_slice[(cc_FS_LTM_shortdf_slice['cellID'] == cell)]
                for pulse_index in freq_sweep_pulses:
                    try:
                        x = calculate_expected_response(celldf, pulse_index, freq, patternID)
                        numSq, freq, patternID, pulse_index, field_data, observed_response, observed_response_scaled, expected_response = x
                        for obs, obs_sc in zip(observed_response, observed_response_scaled):
                            AP = 1 if obs > 20 else 0
                            # print(f'AP detected in {cell}, {numSq}, {patternID}, {freq}, {pulse_index}') if obs > 20 else ''
                            sdn_data.append({
                                            'cellID':   cell,
                                            'numSq':    numSq,
                                            'stimFreq': freq,
                                            'patternID':patternID,
                                            'pulse':    pulse_index,
                                            'obs':      obs,
                                            'obs_scaled':obs_sc,
                                            'exp':      expected_response,
                                            'AP':       AP,
                                            })
                    except ValueError as e:
                        print(e)
                        continue

    # convert the list of dicts into a dataframe
    sdn_df = pd.DataFrame(sdn_data)
    sdn_df.to_hdf(paper_figure_export_location / "Figure3_sdn_data_FS_LTM.h5", key='data')
    print(sdn_df.shape)

    fitdf_temp = gamma_distribution(sdn_df[(sdn_df['AP']==0)], sdndf=None,        x='exp', y='obs',        first='cellID', second='pulse', third='stimFreq')   
    fitdf      = gamma_distribution(sdn_df[(sdn_df['AP']==0)], sdndf=fitdf_temp,  x='exp', y='obs_scaled', first='cellID', second='pulse', third='stimFreq') 
    print(fitdf.shape)
    fitdf.to_hdf(paper_figure_export_location / "Figure3_gamma_and_slope_fits_FS_LTM.h5", key='data')

    cc_delay_df = generate_cc_delay_df(cc_FS_LTM_shortdf_slice)
    vc_delay_df = generate_vc_delay_df(vc_FS_shortdf_slice)
    ebyi_df     = generate_ebyi_df(vc_FS_shortdf_slice)

    # save the dfs
    cc_delay_df.to_hdf(paper_figure_export_location / "Figure3_delay_df_CC_FS.h5", key='data')
    vc_delay_df.to_hdf(paper_figure_export_location / "Figure3_delay_df_VC_FS.h5", key='data')
    ebyi_df.to_hdf(    paper_figure_export_location / "Figure3_ebyi_df_VC_FS.h5" , key='data')

    return sdn_df, fitdf, cc_delay_df, vc_delay_df, ebyi_df

def main():
    plt.close('all')
    Fig3, ax3 = plt.subplot_mosaic([['A','B','C'],['D','E','F'],['G','H','I'],['J','Li','Lii'],['Ki','Mi','Mi'],['Kii','Mii','Mii']], figsize=(21,25),)
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    color_pulses_lin   = mpl.colormaps['Greens']
    color_pulses_gamma = mpl.colormaps['Purples']

    # drop all rows where gamma is nan
    fitdf_slice = fitdf[~fitdf['gamma'].isna()]
    selected_cell = 3402

    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ############ Voltage clamp plots #################
    # E by I ratio vs pulse index across frequencies
    ax3['A'].text(-0.1, 1.05, 'A', fontweight='bold', fontsize=16, ha='center', transform=ax3['A'].transAxes)
    sns.pointplot(data=cc_delay_df, x='pulse', y='peak_PSP', ax=ax3['A'], color=rocket_r(0.5), errorbar='ci',)
    # run a kruskal wallis test across the pulsewise responses
    pulsewise_responses = cc_delay_df.pivot_table(columns='pulse', index='trialID', values='peak_PSP', )
    # now run KW test across columns
    kw_res = kruskal(*[pulsewise_responses[col] for col in pulsewise_responses.columns])
    ax3['A'].set_ylim([0, 1.5])
    ax3['A'].set_yticks([0,0.5,1.0,1.5])
    ax3['A'].legend([],[], frameon=False)
    ax3['A'].set_ylabel('PSP (mV)', fontsize=12)
    ax3['A'].set_xlabel('Pulse Index', fontsize=12)
    sns.despine(ax=ax3['A'], top=True, right=True, offset=10, trim=True)

    ax3['D'].text(-0.1, 1.05, 'D', fontweight='bold', fontsize=16, ha='center', transform=ax3['D'].transAxes)
    sns.pointplot(data=ebyi_df, x='pulse', y=-70, ax=ax3['D'], color=flare(0.5), errorbar='ci', label='Exc')
    sns.pointplot(data=ebyi_df, x='pulse', y = 0, ax=ax3['D'], color=crest(0.5), errorbar='ci', label='Inh' )
    ax3['D'].set_ylim([0,1.5])
    ax3['D'].set_yticks([0,0.5,1.0,1.5])
    ax3['D'].legend(loc='lower left', ncols=4, fontsize='small', bbox_to_anchor=(0.0, 1.0))
    ax3['D'].set_ylabel('PSC Amplitude (norm.)', fontsize=12)
    ax3['D'].set_xlabel('Pulse Index', fontsize=12)
    sns.despine(ax=ax3['D'], top=True, right=True, offset=10, trim=True)

    ax3['G'].text(-0.1, 1.05, 'G', fontweight='bold', fontsize=16, ha='center', transform=ax3['G'].transAxes)
    sns.pointplot(data=ebyi_df, x='pulse', y='EbyI', ax=ax3['G'], color=rocket_r(0.5), errorbar='ci',)
    ax3['G'].set_ylim([0, 3])
    ax3['G'].set_yticks(np.arange(0,3.1,1))
    ax3['G'].legend([],[], frameon=False)
    ax3['G'].set_ylabel('E / I', fontsize=12)
    ax3['G'].set_xlabel('Pulse Index', fontsize=12)
    sns.despine(ax=ax3['G'], top=True, right=True, offset=10, trim=True)

    ### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    ax3['B'].text(-0.1, 1.05, 'B', fontweight='bold', fontsize=16, ha='center', transform=ax3['B'].transAxes)
    sns.pointplot(data=cc_delay_df, x='pulse', y='onset_delay', ax=ax3['B'], color=rocket_r(0.5), errorbar='ci',)
    ax3['B'].set_ylim([0, 10])
    ax3['B'].set_yticks([0,5,10])
    # ax3['B'].legend(loc='upper left', ncols=4, fontsize='small', bbox_to_anchor=(0.0, 1.0))
    ax3['B'].set_ylabel('PSP Onset Delay (ms)', fontsize=12)
    ax3['B'].set_xlabel('Pulse Index', fontsize=12)
    sns.despine(ax=ax3['B'], top=True, right=True, offset=10, trim=True)

    ax3['E'].text(-0.1, 1.05, 'E', fontweight='bold', fontsize=16, ha='center', transform=ax3['E'].transAxes)
    sns.pointplot(data=vc_delay_df, x='pulse', y='exc_onset', ax=ax3['E'], color=flare(0.5), errorbar='ci',)
    sns.pointplot(data=vc_delay_df, x='pulse', y='inh_onset', ax=ax3['E'], color=crest(0.5), errorbar='ci',)
    ax3['E'].set_xticks( np.arange(0,9))
    ax3['E'].set_yticks( np.arange(0,11,5))
    ax3['E'].set_xlabel('Pulse Index', fontsize=12)
    ax3['E'].set_ylabel('PSC Onset Delay (ms)')
    ax3['E'].legend([],[], frameon=False)
    # ax3['E'].legend( loc='lower left', ncols=4, fontsize='small', bbox_to_anchor=(0.0, 0.0))
    [ax3['E'].spines[place].set_visible(False) for place in ['top', 'right', ] ]
    sns.despine(ax=ax3['E'], offset=10, trim=True)

    ax3['H'].text(-0.1, 1.05, 'H', fontweight='bold', fontsize=16, ha='center', transform=ax3['H'].transAxes)
    sns.pointplot(data=vc_delay_df, x='pulse', y='onset_delayEI', ax=ax3['H'], color=rocket_r(0.5), errorbar='ci',)
    ax3['H'].set_ylim([-0.5,5])
    ax3['H'].set_yticks([0,2,4])
    ax3['H'].legend([],[], frameon=False)
    ax3['H'].set_ylabel('Onset Delay (E-I) (ms)', fontsize=12)
    ax3['H'].set_xlabel('Pulse Index', fontsize=12)
    sns.despine(ax=ax3['H'], top=True, right=True, offset=10, trim=True)


    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Voltage clamp plots
    # onset delay vs pulse index across frequencies
    # E by I ratio vs pulse index across frequencies
    ax3['C'].text(-0.1, 1.05, 'C', fontweight='bold', fontsize=16, ha='center', transform=ax3['C'].transAxes)
    sns.pointplot(data=cc_delay_df, x='pulse', y='peak_delay', ax=ax3['C'], color=rocket_r(0.5), errorbar='ci',)
    ax3['C'].set_ylim([0, 25])
    ax3['C'].set_yticks(np.arange(0,26,5))
    ax3['C'].legend([],[], frameon=False)
    ax3['C'].set_ylabel('PSP Peak Delay (ms)', fontsize=12)
    ax3['C'].set_xlabel('Pulse Index', fontsize=12)
    sns.despine(ax=ax3['C'], top=True, right=True, offset=10, trim=True)

    sns.pointplot(data=vc_delay_df, x='pulse', y='exc_peak', ax=ax3['F'], color=flare(0.5), errorbar='ci',)
    sns.pointplot(data=vc_delay_df, x='pulse', y='inh_peak', ax=ax3['F'], color=crest(0.5), errorbar='ci',)

    ax3['F'].text(-0.1, 1.05, 'F', fontweight='bold', fontsize=16, ha='center', transform=ax3['F'].transAxes)
    # ax3['F'].set_ylim([0, 20])
    ax3['F'].set_xticks( np.arange(0,9))
    ax3['F'].set_yticks( np.arange(0,21,5))
    ax3['F'].set_xlabel('Pulse Index', fontsize=12)
    ax3['F'].set_ylabel('PSC Peak Delay (ms)')
    ax3['F'].legend([],[], frameon=False)
    [ax3['F'].spines[place].set_visible(False) for place in ['top', 'right', ] ]
    sns.despine(ax=ax3['F'], offset=10, trim=True)

    ax3['I'].text(-0.1, 1.05, 'I', fontweight='bold', fontsize=16, ha='center', transform=ax3['I'].transAxes)
    sns.pointplot(data=vc_delay_df, x='pulse', y='peak_delayEI', ax=ax3['I'], color=rocket_r(0.5), errorbar='ci',)
    ax3['I'].set_ylim([-0.5,5])
    ax3['I'].set_yticks(np.arange(0,6,2))
    ax3['I'].set_ylabel('Peak Delay (E-I) (ms)', fontsize=12)
    ax3['I'].set_xlabel('Pulse Index', fontsize=12)
    ax3['I'].legend([],[], frameon=False)
    sns.despine(ax=ax3['I'], top=True, right=True, offset=10, trim=True)

    ### ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ### ------------SDN Plots--------------------------------------------------------------------------------------------------------------------------------------------
    # Plot 3A: Scatterplot and SDN for cell = selected_cell, pulse = 0
    ax3['J'].text(-0.1, 1.05, 'J', fontweight='bold', fontsize=16, ha='center', transform=ax3['J'].transAxes)
    dftemp = sdn_df[(sdn_df['cellID']==selected_cell) & (sdn_df['pulse']==0) & (sdn_df['AP']==0)]
    ax3['J'] = sns.scatterplot(data=dftemp, x='exp', y='obs', hue='numSq', size='numSq',sizes=[150], ax=ax3['J'], palette=color_squares)
    # add gamma fit for 0th pule and all frequencies
    gammatemp       = fitdf_slice[(fitdf_slice['cellID']==selected_cell)&(fitdf_slice['observed']=='obs')&(fitdf_slice['pulse']==0)&(fitdf_slice['stimFreq']==1000)]['gamma'].values
    slopetemp       = fitdf_slice[(fitdf_slice['cellID']==selected_cell)&(fitdf_slice['observed']=='obs')&(fitdf_slice['pulse']==0)&(fitdf_slice['stimFreq']==1000)]['slope'].values
    r2_gammatemp    = fitdf_slice[(fitdf_slice['cellID']==selected_cell)&(fitdf_slice['observed']=='obs')&(fitdf_slice['pulse']==0)&(fitdf_slice['stimFreq']==1000)]['r2_sdn'].values
    r2_slopetemp    = fitdf_slice[(fitdf_slice['cellID']==selected_cell)&(fitdf_slice['observed']=='obs')&(fitdf_slice['pulse']==0)&(fitdf_slice['stimFreq']==1000)]['r2_lin'].values
    print(gammatemp, slopetemp, r2_gammatemp, r2_slopetemp)
    ax3['J'].plot(np.linspace(0,20,20), sdnfunc(np.linspace(0,20,20),gammatemp), color='purple', linewidth=3, label=f'γ = {gammatemp[0]:.2f}')
    ax3['J'].plot(np.linspace(0,20,20), nosdn(np.linspace(0,20,20),slopetemp),   color='green', linewidth=3,    label=f'm = {slopetemp[0]:.2f}')
    ax3['J'].plot([0,15],[0,15], color='grey', linestyle='--')

    ax3['J'].set_xlabel('Expected response (mV)')
    ax3['J'].set_ylabel('Observed response (mV)')
    ax3['J'].legend(loc='upper left')

    ax3['J'].set_xlim([0,15])
    ax3['J'].set_ylim([0,10])
    sns.despine(bottom=False, left=False, trim=True, offset=10, ax=ax3['J'])

    # ### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # Plot 3B: lineplot of gamma across all cells, where x-axis: pulse, y-axis: gamma, hue: stim_freq
    ax3['Ki' ].text(-0.1, 1.05, 'Ki',  fontweight='bold', fontsize=16, ha='center', transform=ax3['Ki'].transAxes)
    ax3['Kii'].text(-0.1, 1.05, 'Kii', fontweight='bold', fontsize=16, ha='center', transform=ax3['Kii'].transAxes)


    for i,f in enumerate([20,30,40,50]):
        gammas = []
        slopes = []
        for p in range(9):
            dftemp = sdn_df[(sdn_df['cellID']==selected_cell) & (sdn_df['pulse']==p)& (sdn_df['stimFreq']==f)]
            if dftemp.shape[0] == 0:
                continue
            gammatemp = fitdf_slice[(fitdf_slice['cellID']==selected_cell)&(fitdf_slice['observed']=='obs')&(fitdf_slice['pulse']==p)&(fitdf_slice['stimFreq']==f)]['gamma'].values
            slopetemp = fitdf_slice[(fitdf_slice['cellID']==selected_cell)&(fitdf_slice['observed']=='obs')&(fitdf_slice['pulse']==p)&(fitdf_slice['stimFreq']==f)]['slope'].values
            gammas.append(gammatemp)
            slopes.append(slopetemp)

        ax3['Ki' ].plot(np.arange(9), np.array(gammas), color='purple', linewidth=3, label=f'γ ({f} Hz)', alpha=0.2+i*0.2)
        ax3['Kii'].plot(np.arange(9), np.array(slopes), color='green', linewidth=3, label=f'm ({f} Hz)', alpha=0.2+i*0.2)

        # set ylim
        ax3['Ki'].set_ylim( [0, 10])
        ax3['Kii'].set_ylim([0,0.6])

        sns.despine(bottom=False, left=False, ax=ax3['Ki'],  trim=True, offset=10)
        sns.despine(bottom=False, left=False, ax=ax3['Kii'], trim=True, offset=10)

        # legend outside
        ax3['Ki'].legend( bbox_to_anchor=(0.0, 0.8),  loc='lower left')
        ax3['Kii'].legend(bbox_to_anchor=(0.0, 0.8), loc='lower left')

        ax3['Ki'].set_xlabel('Pulse index', fontdict={'fontsize':12})
        ax3['Ki'].set_ylabel('Gamma (γ)', fontdict={'fontsize':12})
        ax3['Kii'].set_xlabel('Pulse index', fontdict={'fontsize':12})
        ax3['Kii'].set_ylabel('Slope (m)', fontdict={'fontsize':12})


    ### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Histogram of gamma values for all the cells in the control condition
    ax3['Li'].text( -0.1, 1.05, 'L i', fontweight='bold', fontsize=16, ha='center', transform=ax3['Li'].transAxes)
    ax3['Lii'].text(-0.1, 1.05, 'L ii', fontweight='bold', fontsize=16, ha='center', transform=ax3['Lii'].transAxes)

    gammadist0 = fitdf[(fitdf['cellID']!=1000) &(fitdf['pulse']==0) & (fitdf['stimFreq']!=1000)& (fitdf['observed']=='obs')& (fitdf['sample_size']!=0)].dropna(subset=['gamma','slope'])
    gammadist8 = fitdf[(fitdf['cellID']!=1000) &(fitdf['pulse']==8) & (fitdf['stimFreq']!=1000)& (fitdf['observed']=='obs')& (fitdf['sample_size']!=0)].dropna(subset=['gamma','slope'])
    # any gamma value above 100 can be capped at 100
    cap = 50
    gammadist0['gamma'] = gammadist0['gamma'].apply(lambda x: cap if x>cap else x)
    gammadist8['gamma'] = gammadist8['gamma'].apply(lambda x: cap if x>cap else x)

    sns.histplot(data=gammadist0, x='gamma', color='#8b1489', kde=True, ax=ax3['Li'], alpha=1.0, edgecolor='None', binwidth=5, label='Pulse 0', line_kws={'lw': 2,})
    sns.histplot(data=gammadist8, x='gamma', color='#a61900', kde=True, ax=ax3['Li'], alpha=0.5, edgecolor='None', binwidth=5, label='Pulse 8', line_kws={'lw': 2,})

    sns.histplot(data=gammadist0, x='slope', color='#148a14', kde=True, ax=ax3['Lii'], alpha=1.0, edgecolor='None', binwidth=0.1, label='Pulse 0', line_kws={'lw': 2,})
    sns.histplot(data=gammadist8, x='slope', color='#0088a5', kde=True, ax=ax3['Lii'], alpha=0.5, edgecolor='None', binwidth=0.1, label='Pulse 8', line_kws={'lw': 2,})

    # add a vertical line at gammma = 10 and slope = 1
    ax3['Li'].axvline(10, color='black', linestyle='--')
    ax3['Lii'].axvline(1, color='black', linestyle='--')

    ax3['Li'].set_xlabel('Gamma (γ)', fontsize=12)
    ax3['Li'].set_ylabel('Count', fontsize=12)
    ax3['Lii'].set_xlabel('Slope (m)', fontsize=12)
    ax3['Lii'].set_ylabel('Count', fontsize=12)

    sns.despine(bottom=False, left=False, ax=ax3['Li'])
    sns.despine(bottom=False, left=False, ax=ax3['Lii'])
    ax3['Li'].tick_params(axis='both', which='major', labelsize=12)
    ax3['Lii'].tick_params(axis='both', which='major', labelsize=12)

    ax3['Li'].legend(loc='upper right')
    ax3['Lii'].legend(loc='upper right')

    # statistics on gamma and slope
    # rank-order test to check if pulse 0 and pulse 8 distributions are different
    _, pval_gamma = mannwhitneyu(gammadist0['slope'], gammadist8['slope'])
    _, pval_slope = mannwhitneyu(gammadist0['slope'], gammadist8['slope'])

    # stat annotate on the plot
    ax3['Li'].text(0.7, 0.5, f'p = {pval_gamma:.3f}', transform=ax3['Li'].transAxes, fontsize=12, color='grey')
    ax3['Lii'].text(0.7, 0.5, f'p = {pval_slope:.3f}', transform=ax3['Lii'].transAxes, fontsize=12, color='grey')

    # ### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Plot 3D: Heatmap of slope
    importlib.reload(plot_tools)

    fitdf_slice = fitdf[(fitdf['cellID']!=1000) & (fitdf['pulse']!=1000) & (fitdf['stimFreq']!=1000)& (fitdf['observed']=='obs')& (fitdf['sample_size']!=0)].dropna(subset=['gamma','slope'])

    fitdf_slice.drop(columns=['expected','observed','cellID'], inplace=True)
    x = fitdf_slice.groupby(['pulse', 'stimFreq']).median().reset_index()
    n = fitdf_slice.groupby(['pulse', 'stimFreq']).count().reset_index()
    gammapivot = x.pivot(index='stimFreq', columns='pulse', values='gamma')
    gammapivot_n = n.pivot(index='stimFreq', columns='pulse', values='gamma')
    slopepivot = x.pivot(index='stimFreq', columns='pulse', values='slope')
    slopepivot_n = n.pivot(index='stimFreq', columns='pulse', values='slope')
    ax3['Mi'], _, _, _, _ = plot_tools.ax_to_partial_dist_heatmap_ax(gammapivot, gammapivot_n, Fig3, ax3['Mi'], barw=0.03, pad=0.01, shrink=0.8, palette='Purples', force_vmin_to_zero=True, annotate=False)
    ax3['Mii'], _, _, _, _ = plot_tools.ax_to_partial_dist_heatmap_ax(slopepivot, slopepivot_n, Fig3, ax3['Mii'], barw=0.03, pad=0.01, shrink=0.8, palette='Greens', force_vmin_to_zero=True, annotate=False)
    ax3['Mi'].text( -0.1, 1.05, 'Mi', fontweight='bold', fontsize=16, ha='center', transform=ax3['Mi'].transAxes)
    ax3['Mii'].text(-0.1, 1.05, 'Mii', fontweight='bold', fontsize=16, ha='center', transform=ax3['Mii'].transAxes)


    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    for a in ax3.keys():
        ax3[a].tick_params(axis='both', which='major', labelsize=12)
        # axis label fontsize
        ax3[a].set_xlabel(ax3[a].get_xlabel(), fontsize=12)
        ax3[a].set_ylabel(ax3[a].get_ylabel(), fontsize=12)
        # spine width
        ax3[a].spines['left'].set_linewidth(1)
        ax3[a].spines['bottom'].set_linewidth(1)

    # Fig3.tight_layout()
    ## save fig 3
    # Fig3.savefig(paper_figure_export_location / 'Figure3v6.png', dpi=300, bbox_inches='tight')
    # Fig3.savefig(paper_figure_export_location / 'Figure3v6.svg', dpi=300, bbox_inches='tight')

# make dataset
# sdn_df, fitdf, cc_delay_df, vc_delay_df, ebyi_df = make_dataset()

main()