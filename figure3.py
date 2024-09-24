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
paper_figure_export_location = Path(r"paper_figures\\Figure3v4\\")
data_path_FS                 = Path(r"parsed_data\\FreqSweep\\")
data_path_LTM                 = Path(r"parsed_data\\LTMRand\\")
data_path_grid               = Path(r"parsed_data\\Grid\\")
data_path_analysed           = Path(r"parsed_data\\second_order\\")
raw_data_path_cellwise       = Path(r"C:\Users\adity\OneDrive\NCBS\Lab\Projects\EI_Dynamics\Data\Screened_cells\\")

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

VC_FS_shortdf_withkernelfit_datapath = data_path_FS / "all_cells_FreqSweep_VC_kernelfit_response_measurements.h5"
vc_FS_shortdf = pd.read_hdf(VC_FS_shortdf_withkernelfit_datapath, key='data')
print(vc_FS_shortdf.shape)

cc_FS_LTM_shortdf = pd.concat([cc_FS_shortdf, cc_LTM_shortdf], axis=0, ignore_index=True)
# reset index
cc_FS_LTM_shortdf.reset_index(drop=True, inplace=True)

# short data path for all protocols.
# Does not contain kernel fit measurements and does not contain screening parameters. Only use for other protocols.
# 18Sep24
dfshortpath     = data_path_analysed / "all_cells_allprotocols_with_fpr_values.h5"
xc_all_shortdf  = pd.read_hdf(dfshortpath, key='data')
print(xc_all_shortdf.shape)

# Load the long dataset
cc_FS_datapath =  data_path_FS / "all_cells_FreqSweep_CC_long.h5" 
vc_FS_datapath =  data_path_FS / "all_cells_FreqSweep_VC_long.h5"

# Data screening
# CC data screening based on dataflag_fields: protocol freqsweep
cc_FS_LTM_shortdf_slice = cc_FS_LTM_shortdf[
                                            (cc_FS_LTM_shortdf['location'] == 'CA1') &
                                            (cc_FS_LTM_shortdf['numSq'].isin([1,5,7,15])) &
                                            (cc_FS_LTM_shortdf['stimFreq'].isin([20,30,40,50])) &
                                            (cc_FS_LTM_shortdf['condition'] == 'Control') &
                                            (cc_FS_LTM_shortdf['ch0_response']==1) &
                                            (cc_FS_LTM_shortdf['IR'] >50) & (cc_FS_shortdf['IR'] < 300) &
                                            (cc_FS_LTM_shortdf['tau'] < 40) & 
                                            (cc_FS_LTM_shortdf['spike_in_baseline_period'] == 0) &
                                            (cc_FS_LTM_shortdf['ac_noise_power_in_ch0'] < 40) 
                                            ]
print(cc_FS_LTM_shortdf.shape, '--screened-->', cc_FS_LTM_shortdf_slice.shape)
screened_cc_trialIDs = cc_FS_LTM_shortdf_slice['trialID'].unique()

# save trial IDs as a numpy array text file, all trialID are strings
np.savetxt(data_path_FS / "Figure3_screened_trialIDs_CC_FS_LTM.txt", screened_cc_trialIDs, fmt='%s')

cc_FS_LTM_shortdf_slice['patternList'] = cc_FS_LTM_shortdf_slice['patternList'].astype('int32')
patternIDs = np.sort( cc_FS_LTM_shortdf_slice[cc_FS_LTM_shortdf_slice['numSq'] != 1]['patternList'].unique() )

# take list stored in "peaks_field_norm" column and make it into new columns
cc_FS_LTM_shortdf_slice = utils.expand_list_column(cc_FS_LTM_shortdf_slice, 'peaks_field_norm', 'pfn_')

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


def sdnfunc(observed, gamma):
    return gamma * observed / (gamma + observed)

def nosdn(observed, m):
    return m * observed

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
    fitted_gamma = result.best_values['gamma']
    fitted_slope = result_linear.best_values['m']

    return fitted_gamma, fitted_slope, result.rsquared, result_linear.rsquared

def gamma_distribution(df_sdn, sdndf=None, x='expected_response', y='observed_response', first='cellID', second='pulse_index', third='freq'):
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

    if sdndf is not None:
        sdndf2 = pd.concat([sdndf, df_gamma_dist], axis=0)
    else:
        sdndf2 = df_gamma_dist

    return sdndf2


# create expected vs observed data from the raw data
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
sdn_df.to_hdf(data_path_FS / "Figure3_sdn_data_FS_LTM.h5", key='data')
print(sdn_df.shape)

# generate gamma and slope fits for the sdn data
fitdf_temp = gamma_distribution(sdn_df[(sdn_df['AP']==0)], sdndf=None,        x='exp', y='obs',        first='cellID', second='pulse', third='stimFreq')   
fitdf      = gamma_distribution(sdn_df[(sdn_df['AP']==0)], sdndf=fitdf_temp,  x='exp', y='obs_scaled', first='cellID', second='pulse', third='stimFreq') 


# Main figure 3
plt.close('all')

Fig3, ax3 = plt.subplot_mosaic([['A','Ci','Cii'],['Bi','Di','Di'],['Bii','Dii','Dii']], figsize=(15,15), )
plt.subplots_adjust(wspace=0.6, hspace=0.6)

color_pulses_lin   = mpl.colormaps['Greens']
color_pulses_gamma = mpl.colormaps['Purples']

# drop all rows where gamma is nan
fitdf_slice = fitdf[~fitdf['gamma'].isna()]
selected_cell = 3402

### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Plot 3A: Scatterplot and SDN for cell = selected_cell, pulse = 0
ax3['A'].text(-0.1, 1.1, 'A', fontsize=16, ha='center', transform=ax3['A'].transAxes)
dftemp = sdn_df[(sdn_df['cellID']==selected_cell) & (sdn_df['pulse']==0) & (sdn_df['AP']==0)]
ax3['A'] = sns.scatterplot(data=dftemp, x='exp', y='obs', hue='numSq', size='numSq',sizes=[150], ax=ax3['A'], palette=color_squares)
# add gamma fit for 0th pule and all frequencies
gammatemp       = fitdf_slice[(fitdf_slice['cellID']==selected_cell)&(fitdf_slice['observed']=='obs')&(fitdf_slice['pulse']==0)&(fitdf_slice['stimFreq']==1000)]['gamma'].values
slopetemp       = fitdf_slice[(fitdf_slice['cellID']==selected_cell)&(fitdf_slice['observed']=='obs')&(fitdf_slice['pulse']==0)&(fitdf_slice['stimFreq']==1000)]['slope'].values
r2_gammatemp    = fitdf_slice[(fitdf_slice['cellID']==selected_cell)&(fitdf_slice['observed']=='obs')&(fitdf_slice['pulse']==0)&(fitdf_slice['stimFreq']==1000)]['r2_sdn'].values
r2_slopetemp    = fitdf_slice[(fitdf_slice['cellID']==selected_cell)&(fitdf_slice['observed']=='obs')&(fitdf_slice['pulse']==0)&(fitdf_slice['stimFreq']==1000)]['r2_lin'].values
print(gammatemp, slopetemp, r2_gammatemp, r2_slopetemp)
ax3['A'].plot(np.linspace(0,20,20), sdnfunc(np.linspace(0,20,20),gammatemp), color='purple', linewidth=3, label=f'γ = {gammatemp[0]:.2f} ($R^2$= {r2_gammatemp[0]:.2f})')
ax3['A'].plot(np.linspace(0,20,20), nosdn(np.linspace(0,20,20),slopetemp),   color='green', linewidth=3,    label=f'm = {slopetemp[0]:.2f} ($R^2$= {r2_slopetemp[0]:.2f})')
ax3['A'].plot([0,15],[0,15], color='grey', linestyle='--')

ax3['A'].set_xlabel('Expected response (mV)')
ax3['A'].set_ylabel('Observed response (mV)')
ax3['A'].legend(loc='upper left')

ax3['A'].set_xlim([0,20])
ax3['A'].set_ylim([0,10])
sns.despine(bottom=False, left=False, trim=True, offset=10, ax=ax3['A'])

# ### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # Plot 3B: lineplot of gamma across all cells, where x-axis: pulse, y-axis: gamma, hue: stim_freq
ax3['Bi'].text(-0.1, 1.1, 'Bi', fontsize=16, ha='center', transform=ax3['Bi'].transAxes)
ax3['Bii'].text(-0.1, 1.1, 'Bii', fontsize=16, ha='center', transform=ax3['Bii'].transAxes)


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

    ax3['Bi'].plot(np.arange(9), np.array(gammas), color='purple', linewidth=3, label=f'γ ({f} Hz)', alpha=0.2+i*0.2)
    ax3['Bii'].plot(np.arange(9), np.array(slopes), color='green', linewidth=3, label=f'm ({f} Hz)', alpha=0.2+i*0.2)

    # set ylim
    ax3['Bi'].set_ylim([0,10])
    ax3['Bii'].set_ylim([0,0.4])

    sns.despine(bottom=False, left=False, ax=ax3['Bi'],   trim=True, offset=10)
    sns.despine(bottom=False, left=False, ax=ax3['Bii'], trim=True, offset=10)

    # legend outside
    ax3['Bi'].legend( bbox_to_anchor=(0.0, 0.8),  loc='lower left')
    ax3['Bii'].legend(bbox_to_anchor=(0.0, 0.8), loc='lower left')

    ax3['Bi'].set_xlabel('Pulse index', fontdict={'fontsize':12})
    ax3['Bi'].set_ylabel('Gamma (γ)', fontdict={'fontsize':12})
    ax3['Bii'].set_xlabel('Pulse index', fontdict={'fontsize':12})
    ax3['Bii'].set_ylabel('Slope (m)', fontdict={'fontsize':12})


### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Histogram of gamma values for all the cells in the control condition
ax3['Ci'].text( -0.1, 1.1, 'C i', fontsize=16, ha='center', transform=ax3['Ci'].transAxes)
ax3['Cii'].text(-0.1, 1.1, 'C ii', fontsize=16, ha='center', transform=ax3['Cii'].transAxes)

gammadist0 = fitdf[(fitdf['cellID']!=1000) &(fitdf['pulse']==0) & (fitdf['stimFreq']!=1000)& (fitdf['observed']=='obs')& (fitdf['sample_size']!=0)].dropna(subset=['gamma','slope'])
gammadist8 = fitdf[(fitdf['cellID']!=1000) &(fitdf['pulse']==8) & (fitdf['stimFreq']!=1000)& (fitdf['observed']=='obs')& (fitdf['sample_size']!=0)].dropna(subset=['gamma','slope'])
# any gamma value above 100 can be capped at 100
cap = 50
gammadist0['gamma'] = gammadist0['gamma'].apply(lambda x: cap if x>cap else x)
gammadist8['gamma'] = gammadist8['gamma'].apply(lambda x: cap if x>cap else x)

sns.histplot(data=gammadist0, x='gamma', color='#8b1489', kde=True, ax=ax3['Ci'], alpha=1.0, edgecolor='None', binwidth=5, label='Pulse 0', line_kws={'lw': 2,})
sns.histplot(data=gammadist8, x='gamma', color='#a61900', kde=True, ax=ax3['Ci'], alpha=0.5, edgecolor='None', binwidth=5, label='Pulse 8', line_kws={'lw': 2,})

sns.histplot(data=gammadist0, x='slope', color='#148a14', kde=True, ax=ax3['Cii'], alpha=1.0, edgecolor='None', binwidth=0.1, label='Pulse 0', line_kws={'lw': 2,})
sns.histplot(data=gammadist8, x='slope', color='#0088a5', kde=True, ax=ax3['Cii'], alpha=0.5, edgecolor='None', binwidth=0.1, label='Pulse 8', line_kws={'lw': 2,})

# add a vertical line at gammma = 10 and slope = 1
ax3['Ci'].axvline(10, color='black', linestyle='--')
ax3['Cii'].axvline(1, color='black', linestyle='--')

ax3['Ci'].set_xlabel('Gamma (γ)', fontsize=12)
ax3['Ci'].set_ylabel('Count', fontsize=12)
ax3['Cii'].set_xlabel('Slope (m)', fontsize=12)
ax3['Cii'].set_ylabel('Count', fontsize=12)

sns.despine(bottom=False, left=False, ax=ax3['Ci'])
sns.despine(bottom=False, left=False, ax=ax3['Cii'])
ax3['Ci'].tick_params(axis='both', which='major', labelsize=12)
ax3['Cii'].tick_params(axis='both', which='major', labelsize=12)

ax3['Ci'].legend(loc='upper right')
ax3['Cii'].legend(loc='upper right')

# statistics on gamma and slope
# rank-order test to check if pulse 0 and pulse 8 distributions are different
_, pval_gamma = mannwhitneyu(gammadist0['slope'], gammadist8['slope'])
_, pval_slope = mannwhitneyu(gammadist0['slope'], gammadist8['slope'])

# stat annotate on the plot
ax3['Ci'].text(0.7, 0.5, f'p = {pval_gamma:.3f}', transform=ax3['Ci'].transAxes, fontsize=12, color='grey')
ax3['Cii'].text(0.7, 0.5, f'p = {pval_slope:.3f}', transform=ax3['Cii'].transAxes, fontsize=12, color='grey')

# ### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Plot 3D: Heatmap of slope
importlib.reload(plot_tools)
ax3['Di'].text( -0.1, 1.1, 'Di', fontsize=16, ha='center', transform=ax3['Di'].transAxes)
ax3['Dii'].text(-0.1, 1.1, 'Dii', fontsize=16, ha='center', transform=ax3['Dii'].transAxes)

fitdf_slice = fitdf[(fitdf['cellID']!=1000) & (fitdf['pulse']!=1000) & (fitdf['stimFreq']!=1000)& (fitdf['observed']=='obs')& (fitdf['sample_size']!=0)].dropna(subset=['gamma','slope'])

fitdf_slice.drop(columns=['expected','observed','cellID'], inplace=True)
x = fitdf_slice.groupby(['pulse', 'stimFreq']).median().reset_index()
n = fitdf_slice.groupby(['pulse', 'stimFreq']).count().reset_index()
gammapivot = x.pivot(index='stimFreq', columns='pulse', values='gamma')
gammapivot_n = n.pivot(index='stimFreq', columns='pulse', values='gamma')
slopepivot = x.pivot(index='stimFreq', columns='pulse', values='slope')
slopepivot_n = n.pivot(index='stimFreq', columns='pulse', values='slope')
plot_tools.ax_to_partial_dist_heatmap_ax(gammapivot, gammapivot_n, Fig3, ax3['Di'], barw=0.03, pad=0.01, shrink=0.8, palette='Purples', force_vmin_to_zero=True, annotate=False)
plot_tools.ax_to_partial_dist_heatmap_ax(slopepivot, slopepivot_n, Fig3, ax3['Dii'], barw=0.03, pad=0.01, shrink=0.8, palette='Greens', force_vmin_to_zero=True, annotate=False)

# save fig3
Fig3.savefig(paper_figure_export_location / 'Figure3.png', dpi=300, bbox_inches='tight')
Fig3.savefig(paper_figure_export_location / 'Figure3.svg', dpi=300, bbox_inches='tight')
