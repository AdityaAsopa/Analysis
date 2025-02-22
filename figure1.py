# Short-term dynamics of Excitation-Inhibition Balance in Hippocampal CA3-CA1 circuit
# Aditya Asopa, Upinder Singh Bhalla, NCBS
# Publication: https://www.biorxiv.org/content/10.1101/2024.10.30.621034v1

# Imports -----------------------------------------------------------------------------------------------
from   pathlib      import Path
import importlib

import numpy                as np
import matplotlib           as mpl
import matplotlib.pyplot    as plt
import matplotlib.patches   as mpatches
import seaborn              as sns
import pandas               as pd

from scipy.stats   import kruskal, wilcoxon, mannwhitneyu, ranksums
from scipy.optimize import curve_fit
from scipy.stats import linregress

from PIL            import Image

from eidynamics.fit_PSC     import find_sweep_expected
from eidynamics     import utils, plot_tools
from eidynamics     import pattern_index
import eidynamics.plotFig2 as plotFig2
import eidynamics.stat_annotate as stat_annotate

# sns.set_context('paper')
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 16
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['lines.linewidth'] = 2

# make a colour map viridis
viridis = mpl.colormaps["viridis"]
cividis = mpl.colormaps["cividis"]
flare   = mpl.colormaps["flare"]
crest   = mpl.colormaps["crest"]
magma   = mpl.colormaps["magma"]
edge    = mpl.colormaps['edge']

color_E             = "flare"
color_I             = "crest"
color_freq          = {1:magma(0.05), 5:magma(0.1), 8:magma(0.15), 10:magma(0.2), 20:magma(.4), 30:magma(.5), 40:magma(.6), 50:magma(.7), 100:magma(.9)}
color_squares       = {1:viridis(0.2), 5:viridis(.4), 7:viridis(.6), 15:viridis(.8), 20:viridis(1.0)}
color_squares_EI    = {-70: {1:flare(0.2), 5:flare(.4), 7:flare(.6), 15:flare(.8), 20:flare(1.0)}, 
                         0: {1:crest(0.2), 5:crest(.4), 7:crest(.6), 15:crest(.8), 20:crest(1.0)}}
color_EI            = {-70:flare(0.5), 0:crest(0.5)}
Fs = 2e4

freq_sweep_pulses = np.arange(9)

# Load Data -----------------------------------------------------------------------------------------------
figure_raw_material_location = Path(r"\\storage.ncbs.res.in\adityaa\Lab\Projects\EI_Dynamics\Analysis\paper_figures\submission")
paper_figure_export_location = Path(r"\\storage.ncbs.res.in\adityaa\Lab\Projects\EI_Dynamics\Analysis\paper_figures\submission")
data_path_FS                 = Path(r"\\storage.ncbs.res.in\adityaa\Lab\Projects\EI_Dynamics\Analysis\parsed_data\Jan_2025\FreqSweep")
data_path_LTM                = Path(r"\\storage.ncbs.res.in\adityaa\Lab\Projects\EI_Dynamics\Analysis\parsed_data\Jan_2025\LTMRand")
data_path_grid               = Path(r"\\storage.ncbs.res.in\adityaa\Lab\Projects\EI_Dynamics\Analysis\parsed_data\Jan_2025\Grid")
data_path_analysed           = Path(r"\\storage.ncbs.res.in\adityaa\Lab\Projects\EI_Dynamics\Analysis\parsed_data\Jan_2025\second_order")

# short data path that contains the kernel fit data for FreqSweep protocol, also contains the field p2p data. latest and checked. Use this for all freqsweep measurements.
# Contains screening parameters also.
# 18Sep24
CC_FS_shortdf_withkernelfit_datapath = data_path_FS / "all_cells_FreqSweep_CC_kernelfit_response_measurements.h5"
cc_FS_shortdf = pd.read_hdf(CC_FS_shortdf_withkernelfit_datapath, key='data')
print(cc_FS_shortdf.shape)

VC_FS_shortdf_withkernelfit_datapath = data_path_FS / "all_cells_FreqSweep_VC_kernelfit_response_measurements.h5"
vc_FS_shortdf = pd.read_hdf(VC_FS_shortdf_withkernelfit_datapath, key='data')
print(vc_FS_shortdf.shape)

# short data path for all protocols.
# Does not contain kernel fit measurements and does not contain screening parameters. Only use for other protocols.
# 18Sep24
dfshortpath     = data_path_analysed / "all_cells_allprotocols_with_fpr_values.h5"
xc_all_shortdf  = pd.read_hdf(dfshortpath, key='data')
print(xc_all_shortdf.shape)

# Long Datadframes containing raw data and some analysed params
# Load the long dataset
cc_FS_datapath =  data_path_FS / "all_cells_FreqSweep_CC_long.h5" 
vc_FS_datapath =  data_path_FS / "all_cells_FreqSweep_VC_long.h5"

# Load it later as needed
# cc_FS_longdf = pd.read_hdf(cc_FS_datapath, key='data')
# vc_FS_longdf = pd.read_hdf(vc_FS_datapath, key='data')

# Screening -----------------------------------------------------------------------------------------------
# CC data screening based on dataflag_fields
cc_FS_shortdf_slice = cc_FS_shortdf[
            (cc_FS_shortdf['location'] == 'CA1') &
            (cc_FS_shortdf['numSq'].isin([1,5,15])) &
            (cc_FS_shortdf['condition'] == 'Control') &
            (cc_FS_shortdf['ch0_response']==1) &
            (cc_FS_shortdf['intensity'] == 100) &
            (cc_FS_shortdf['pulseWidth'] == 2) &
            (cc_FS_shortdf['IR'] >50) & (cc_FS_shortdf['IR'] < 300) &
            (cc_FS_shortdf['tau'] < 40) & 
            (cc_FS_shortdf['ac_noise_power_in_ch0'] < 40) &
            (cc_FS_shortdf['spike_in_baseline_period'] == 0) &
            (cc_FS_shortdf['sweepBaseline'] > -100) & (cc_FS_shortdf['sweepBaseline'] < 100)
        ]
print(cc_FS_shortdf.shape, '--screened-->', cc_FS_shortdf_slice.shape)
screened_cc_trialIDs = cc_FS_shortdf_slice['trialID'].unique()

print(f"Unique cells in screened data: { cc_FS_shortdf_slice['cellID'].nunique()}")
print(f"Unique sweeps in screened data: {cc_FS_shortdf_slice['trialID'].nunique()}")

# save the screened trialIDs
# save trial IDs as a numpy array text file, all trialID are strings
np.savetxt(paper_figure_export_location / "Figure1_screened_trialIDs_CC_FS.txt", screened_cc_trialIDs, fmt='%s')

# VC data screening based on dataflag_fields
vc_FS_shortdf_slice = vc_FS_shortdf[
            (vc_FS_shortdf['location'] == 'CA1') &
            (vc_FS_shortdf['numSq'].isin([1,5,15])) &
            (vc_FS_shortdf['stimFreq'].isin([20,30,40,50])) &
            (vc_FS_shortdf['condition'] == 'Control') &
            (vc_FS_shortdf['ch0_response']==1) &
            (vc_FS_shortdf['IR'] >50) & (vc_FS_shortdf['IR'] < 300) &
            (vc_FS_shortdf['tau'] < 40) & 
            (vc_FS_shortdf['ac_noise_power_in_ch0'] < 40)
        ]
print(vc_FS_shortdf.shape, '--screened-->', vc_FS_shortdf_slice.shape)
screened_vc_trialIDs = vc_FS_shortdf_slice['trialID'].unique()

print(f"Unique cells in screened data: { vc_FS_shortdf_slice['cellID'].nunique()}")
print(f"Unique sweeps in screened data: {vc_FS_shortdf_slice['trialID'].nunique()}")

# save the screened trialIDs
# save trial IDs as a numpy array text file, all trialID are strings
np.savetxt(paper_figure_export_location / "Figure1_screened_trialIDs_VC_FS.txt", screened_vc_trialIDs, fmt='%s')

# combine short dataframes slice
xc_FS_shortdf_slice = pd.concat([cc_FS_shortdf_slice, vc_FS_shortdf_slice], axis=0)

### Load the Longform data and keep the screened trials only to save space
# load the data
cc_FS_datapath =  data_path_FS / "all_cells_FreqSweep_CC_long.h5"
cc_FS_longdf = pd.read_hdf(cc_FS_datapath, key='data')

cc_FS_longdf_slice = cc_FS_longdf[ cc_FS_longdf['trialID'].isin(screened_cc_trialIDs) ]
print(cc_FS_longdf.shape, '--screened-->', cc_FS_longdf_slice.shape)
del cc_FS_longdf

# load the data
vc_FS_datapath =  data_path_FS / "all_cells_FreqSweep_VC_long.h5"
vc_FS_longdf = pd.read_hdf(vc_FS_datapath, key='data')

vc_FS_longdf_slice = vc_FS_longdf[ vc_FS_longdf['trialID'].isin(screened_vc_trialIDs) ]
print(vc_FS_longdf.shape, '--screened-->', vc_FS_longdf_slice.shape)
del vc_FS_longdf

xc_FS_longdf_slice = pd.concat([cc_FS_longdf_slice, vc_FS_longdf_slice])
del cc_FS_longdf_slice, vc_FS_longdf_slice

# Plotting -----------------------------------------------------------------------------------------------
def main():
    # Figure 1 initialization
    plt.close('all')
    # aspect ratio of the figure = 1
    w, h = [19,27]
    mosaic = '''
                AAABBBCCCDDD
                EEEEFFFFGGGG
                HHHHIIIIJJJJ
                KKKKLLLLMMMM
                '''
    fig1, ax1 = plt.subplot_mosaic(mosaic, figsize=(w, h), )
    ax1a, ax1b, ax1c, ax1d, ax1e, ax1f, ax1g, ax1h, ax1i, ax1j, ax1k, ax1l, ax1m = [ax1[_] for _ in ax1.keys()]

    # have more space between suplots
    fig1.subplots_adjust(hspace=0.5, wspace=0.5)

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Fig 1A: Slice, polygon projection, and recording electrodes
    ax1a.text(-0.1, 1.1, 'A', transform=ax1a.transAxes, size=20, weight='bold')
    image1_path = paper_figure_export_location / "Fig1A.png"
    im1 = Image.open(image1_path)
    imw, imh = im1.size
    im1 = ax1a.imshow(im1, extent=[0, imw, 0, imh])
    ax1a.set_aspect(imw/imh)
    ax1a.axis('off')
    ax1a.set_anchor('NW')

    ax1b.text(-0.1, 1.1, 'B', transform=ax1b.transAxes, size=20, weight='bold')
    image2_path = paper_figure_export_location / "Fig1B.png"
    im2 = Image.open(image2_path)
    imw, imh = im2.size
    im2 = ax1b.imshow(im2, extent=[0, imw, 0, imh])
    ax1b.set_aspect(imw/imh)
    ax1b.axis('off')

    ax1c.text(-0.1, 1.1, 'C', transform=ax1c.transAxes, size=20, weight='bold')
    image3_path = paper_figure_export_location / "Fig1G.png"
    im3 = Image.open(image3_path)
    imw, imh = im3.size
    im3 = ax1c.imshow(im3, extent=[0, imw, 0, imh])
    ax1c.set_aspect(imw/imh)
    ax1c.axis('off')

    # # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # Fig 1D: Protocol Freq Sweep
    ax1d.text(-0.1, 1.1, 'D', transform=ax1d.transAxes, size=20, weight='bold')
    sample_cell = 1931  #screened_cc_cells[2] # 3131
    celldata = xc_FS_longdf_slice[ (xc_FS_longdf_slice['clampMode']=='CC')& (xc_FS_longdf_slice['AP']==0) & (xc_FS_longdf_slice['numChannels']==4)]
    data = celldata[(celldata['stimFreq']==20) & (celldata['numSq']==15)]
    fig1, ax1d, signal_mapping = plot_tools.plot_data_from_df(data, data_start_column=49,combine=True, fig=fig1, ax=ax1d, )
    scalebar_cell = signal_mapping['scalebar_cell']
    scalebar_field = signal_mapping['scalebar_field']
    # set legend inside the plot
    ax1d.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9), frameon=True) # legend removed to save space
    # simplify
    plot_tools.simplify_axes(ax1d, splines_to_keep=[], )
    # remove legend
    ax1d.legend([],[], frameon=False)

    # # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # # Fig 1E: Current clamp responses
    ax1e.text(-0.1, 1.1, 'E', transform=ax1e.transAxes, size=20, weight='bold')
    dftemp = xc_FS_longdf_slice[(xc_FS_longdf_slice['cellID']==4041) ]
    ax1e.set_xlim([0, 0.5])
    ax1e.set_ylim([-0.15, 0.45])
    ax1e, ax1e_insets = plot_tools.draw_pulse_response_snippets(dftemp, ax1e, signal='field', window=0.15, patterns=range(56), between='clampPotential', hue='numSq', hues=[1,5,15],
                                                                grid_size=11,stim_scale=2.5, stim_offset=0.3 ,palette=color_squares_EI, invert=True, filter_data=True, passband=[100, 500])
    ax1e.set_xticks([])
    ax1e.set_yticks([])
    # # axes spines to remove
    # plot_tools.split_axes(ax1e, which_axes=['left', 'bottom'], offset=10)
    sns.despine(ax=ax1e, top=True, right=True, left=True, bottom=True, offset=10, )
    # add floating scalebar
    plot_tools.add_floating_scalebar(ax1e, scalebar_origin=[0.05,0.4], xlength=0.1, ylength=0.1, labelx='100', labely='', unitx='ms', unity='mV', fontsize=12, color='black', linewidth=2, pad=0.01, show_labels=True)

    # # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # # Fig 1F: CA3 cell_fpr vs field_fpr
    ax1f.text(-0.1, 1.1, 'F', transform=ax1f.transAxes, size=20, weight='bold')
    # plot a relationship between cell_fpr and field_fpr using a scatter plot (options: box, strip, swarm, violin, boxen)
    dftemp1 = xc_FS_shortdf_slice[(xc_FS_shortdf_slice['numSq'].isin([1,5,15])) & (xc_FS_shortdf_slice['fieldunit']!='mV')]
    # multiply field_fpr_p2p by 0.05 to get the correct scale
    dftemp1['field_fpr_p2p'] = dftemp1['field_fpr_p2p']*0.05
    dftemp2 = xc_FS_shortdf_slice[(xc_FS_shortdf_slice['numSq'].isin([1,5,15])) & (xc_FS_shortdf_slice['fieldunit']=='mV')]
    #join the df back
    dftemp = pd.concat([dftemp1, dftemp2], axis=0)
    dftemp = dftemp.dropna(subset=['field_fpr_p2p'])

    sns.violinplot(data=dftemp, x="numSq", y="field_fpr_p2p", hue="numSq", palette=color_squares, ax=ax1f, alpha=0.5, inner='point', split=False, dodge=False, cut=0, zorder=3)
    [part.set_edgecolor((part.get_edgecolor()[:],  0)) for part in ax1f.get_children() if isinstance(part, mpl.collections.PolyCollection)]
    print(dftemp2[dftemp2['clampMode']=='VC'].shape, dftemp2[dftemp2['clampMode']=='CC'].shape)
    # # ax1f.set_ylabel('CA3 cell Response (mV)')
    # # ax1f.set_xlabel('Number of Squares per Pattern')
    # # no legend
    ax1f.legend([],[], frameon=False)

    # # remove top and right spines
    ax1f.spines['top'].set_visible(False)
    ax1f.spines['right'].set_visible(False)

    # # statistics
    importlib.reload(stat_annotate)
    ax1f_pvalues, ax1f_nvalues = stat_annotate.pairwise_annotate_violin_plot(ax1f, dftemp, x='numSq', y='field_fpr_p2p', stat=mannwhitneyu, add_line=True, offset=0.15, color='grey',
                                                                            add_n=False, coord_system='', fontsize=12, zorder=10)

    # set labels
    ax1f.set_ylabel('First Peak Field Response (mV)')
    ax1f.set_xlabel('Num Squares')
    # legend off
    ax1f.legend([],[], frameon=False)
    # remove top and right spines
    ax1f.spines['top'].set_visible(False)
    ax1f.spines['right'].set_visible(False)
    sns.despine(ax=ax1f, top=True, right=True, left=False, bottom=False, offset=10, trim=True) 

    # #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # # Fig 1G: CA3 cell_fpr vs field_fpr
    ax1g.text(-0.1, 1.1, 'G', transform=ax1g.transAxes, size=20, weight='bold')
    dftemp = xc_all_shortdf[ (xc_all_shortdf['cellID']==3161) ]
    # # add scatterplot on ax1f
    sns.scatterplot(data=dftemp, x="cell_fpr_max", y="field_fpr_p2p", hue='numSq', palette=color_squares, ax=ax1g, alpha=0.8, s=75, legend=True)
    # add regression line to the scatterplot
    sns.regplot(data=dftemp, x="cell_fpr_max", y="field_fpr_p2p", ax=ax1g, scatter=False, color='black', ci=95)
    # calculate the correlation coefficient
    resultsCA3 = linregress(dftemp["cell_fpr_max"], dftemp["field_fpr_p2p"])
    # add r2 and p value to the plot as text annotation
    ax1g.text(0.05, 0.9, fr"$r^2$ = {resultsCA3.rvalue**2:.2f}, p <<< 0.001", transform=ax1g.transAxes, fontsize=16)
    ax1g.set_xlabel('First Peak Optic Current (pA)')
    ax1g.set_ylabel('Field response (mV)')
    # despine
    ax1g.set_xlim([0, 65])
    ax1g.set_xticks([0, 20, 40, 60,])
    ax1g.set_yticks([0, 0.5, 1.0, 1.5])
    sns.despine(ax=ax1g, top=True, right=True, left=False, bottom=False, offset=10, )

    # # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # # Fig 1H: Current clamp responses
    ax1h.text(-0.1, 1.1, 'H', transform=ax1h.transAxes, size=20, weight='bold')
    ax1h.set_xlim([0, 0.5])
    ax1h.set_ylim([-3.3, 10])
    dftemp = xc_FS_longdf_slice[(xc_FS_longdf_slice['cellID']==4041)]
    ax1h, ax1h_insets = plot_tools.draw_pulse_response_snippets(dftemp, ax1h, patterns=range(55), between='clampPotential', hue='numSq',  palette=color_squares_EI,grid_size=11,stim_offset=9, stim_scale=50,)
    # # set ticks
    ax1h.set_xticks([])
    ax1h.set_yticks([])
    # # axes spines to remove
    sns.despine(ax=ax1h, top=True, right=True, left=True, bottom=True, offset=10, )
    plot_tools.add_floating_scalebar(ax1h, scalebar_origin=[0.05,0.4], xlength=0.1, ylength=2.0, labelx='100', labely='', unitx='ms', unity='mV', fontsize=12, color='black', linewidth=2, pad=0.01, show_labels=True)

    # # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # # Fig 1I: First peak response | cell_fpr_max vs numSq
    ax1i.text(-0.1, 1.1, 'I', transform=ax1i.transAxes, size=20, weight='bold')
    # # ax1g.set_title('First Peak Response')
    dftemp = xc_FS_shortdf_slice[ (xc_FS_shortdf_slice['clampMode']=='CC') & (xc_FS_shortdf_slice['probePulseStart'] == 0.2) & (xc_FS_shortdf_slice['AP'] == 0)]
    dftemp = dftemp.dropna(subset=['PSC_0'])
    # # sns.stripplot(data=df_temp,  x="numSq", y="cell_fpr_max", hue="numSq", palette=color_squares, ax=ax1g, alpha=0.8, s=2, jitter=0.25, orient="v", linewidth=0.25)
    sns.violinplot(data=dftemp, x="numSq", y="PSC_0", hue="numSq", palette=color_squares, ax=ax1i, alpha=0.5, inner='quart', split=False, dodge=False, cut=0, zorder=3)
    [part.set_edgecolor((part.get_edgecolor()[:],  0)) for part in ax1i.get_children() if isinstance(part, mpl.collections.PolyCollection)]

    ax1i.set_ylabel('First Peak CA1 Response (mV)')
    ax1i.set_xlabel('Number of Squares per Pattern')
    ax1i.legend([],[], frameon=False)
    # remove top and right spines
    ax1i.spines['top'].set_visible(False)
    ax1i.spines['right'].set_visible(False)

    # ## Statistics for the violin plots: mannwhitneyu Rank Sum Test across numSq values
    ax1i_pvalues, ax1i_nvalues = stat_annotate.pairwise_annotate_violin_plot(ax1i, dftemp, x='numSq', y='PSC_0', stat=mannwhitneyu, add_line=True, offset=0.2, color='grey',
                                                                            add_n=False, coord_system='', fontsize=12, zorder=10)
    sns.despine(ax=ax1i, top=True, right=True, left=False, bottom=False, offset=10, trim=True)   

    # # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # # Fig 1J: Trend in the field response across pulses
    ax1j.text(-0.1, 1.1, 'J', transform=ax1j.transAxes, size=20, weight='bold')
    dftemp = xc_FS_shortdf_slice[(xc_FS_shortdf_slice['numSq'].isin([1,5,15])) & (xc_FS_shortdf_slice['numChannels']==4)]
    dftemp = dftemp.reset_index(drop=True)
    # explode 'peaks_field' column to get a 9 columns for 9 pulses for each row
    dfx = pd.DataFrame(columns=['pulse_index'])
    for i in range(len(dftemp)):
        x = np.array(range(9), dtype='object')
        dfx.at[i, 'pulse_index'] = x
    # concetnate the two dataframes
    dftemp = pd.concat([dftemp, dfx], axis=1)
    dftemp = dftemp.explode(['peaks_field_norm','pulse_index'])
    # # convert 'peaks_field_norm' and 'pulse_index' to float
    dftemp['peaks_field_norm'] = dftemp['peaks_field_norm'].astype(float)
    dftemp['pulse_index'] = dftemp['pulse_index'].astype(int)

    dftemp = dftemp.reset_index(drop=True)
    dftemp = dftemp.dropna(subset=['peaks_field_norm', 'pulse_index'])
    # # draw pointplot
    sns.pointplot(data=dftemp, x='pulse_index', y='peaks_field_norm', hue='pulse_index', markersize=5, palette='viridis', dodge=True, ax=ax1j, errorbar=('ci', 95), zorder=3)

    # # remove 0th pulse data for the regression to avoide the bias
    dftemp1 = dftemp[dftemp['pulse_index']!=0]
    results = linregress(dftemp1['pulse_index'], dftemp1['peaks_field_norm'])
    # # add r2 and p value to the plot as text annotation
    ax1j.text(0.05, 0.76, fr"y  = {results.slope:.2f}x + {results.intercept:.2f},  $r^2 = {results.rvalue**2:.2f}$ , p <<< 0.001", transform=ax1j.transAxes, fontsize=12)

    # try an exponential decay function
    # def exp_decay(x, a, tau):
    #     return a*np.exp(-x/tau)

    # dftemp2 = dftemp[dftemp['pulse_index']>1]
    # popt, pcov = curve_fit(exp_decay, dftemp2['pulse_index'], dftemp2['peaks_field_norm'], )
    # # exponent fit plot
    # x = np.linspace(2, 8, 20)
    # y = exp_decay(x, *popt)
    # ax1j.plot(x, y, color='blue', linestyle='-', linewidth=0.5)
    # # # add r2 and p value to the plot as text annotation
    # ax1j.text(0.05, 0.68, fr"y  = {popt[0]:.2f}e$^{{-x/{popt[1]:.2f}}}$, $r^2 = {results.rvalue**2:.2f}$ , p  = {results.pvalue:.2}", transform=ax1j.transAxes, fontsize=12) # result was a=1.1, tau = 1/37.3

    # # set labels
    ax1j.set_ylabel('Normalized Field Response (mV)')
    ax1j.set_xlabel('Pulse #')
    ax1j.plot([1,8], [results.intercept + results.slope, results.intercept + results.slope*8], color='black', linestyle='-', linewidth=0.5)
    ax1j.set_ylim([0.5,1.5])
    ax1j.set_yticks([0.5,1.0,1.5])
    ax1j.legend([],[], frameon=False)
    # # despine
    sns.despine(ax=ax1j, top=True, right=True, left=False, bottom=False, offset=10, trim=True)

    # # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # # Fig 1K: Trend in the field response across pulses
    ax1k.text(-0.1, 1.1, 'K', transform=ax1k.transAxes, size=20, weight='bold')
    # ax1k.set_xlim([0, 0.5])
    # ax1k.set_ylim([-400,1200])
    importlib.reload(plot_tools)
    dftemp = xc_FS_longdf_slice[(xc_FS_longdf_slice['cellID'].isin([6201, 1621]))]
    print( dftemp['clampPotential'].unique())
    ax1k, ax1j_insets = plot_tools.draw_pulse_response_snippets(dftemp, ax1k, patterns=range(55), palette=color_squares_EI, between='clampPotential', hue='numSq',stim_offset=750, stim_scale=200, grid_size=11)
    # set ticks
    ax1k.set_xticks([])
    ax1k.set_yticks([])
    # axes spines to remove
    # plot_tools.split_axes(ax1k, which_axes=['left', 'bottom'], offset=10)
    sns.despine(ax=ax1k, top=True, right=True, left=True, bottom=True, offset=10, )
    plot_tools.add_floating_scalebar(ax1k, scalebar_origin=[0.05,0.4], xlength=0.1, ylength=200, labelx='100', labely='', unitx='ms', unity='pA', fontsize=12, color='black', linewidth=2, pad=0.01, show_labels=True)

    # # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # # Fig 1l: Trend in the field response across pulses
    ax1l.text(-0.1, 1.1, 'L', transform=ax1l.transAxes, size=20, weight='bold')
    dftemp = xc_FS_shortdf_slice[ (xc_FS_shortdf_slice['clampMode']=='VC') & (xc_FS_shortdf_slice['probePulseStart'] == 0.2)]
    dftemp = dftemp.dropna(subset=['PSC_0'])
    pscdict = {-70: [], 0: []}
    for k,v in pscdict.items():
        for sq in [1,5,15]:
            dftemp1 = dftemp[(dftemp['clampPotential']==k) & (dftemp['numSq']==sq)]
            sqlist = dftemp1['PSC_0'].to_list()
            pscdict[k].append(sqlist)
    labels=[]
    for clamp in [-70,0]:
        side = 'low' if clamp == 0 else 'high'
        P ='EPSC' if clamp == -70 else 'IPSC'
        parts = ax1l.violinplot(pscdict[clamp], positions=[1,2,3], vert=True, widths=0.8, showmeans=False,
                            showextrema=False, quantiles=[[0.25,0.75],[0.25,0.75],[0.25,0.75]] ,showmedians=True, side='both', )
        a,b,c = pscdict[clamp]
        _, pval = kruskal(a,b,c)
        stat_annotate.annotate_stat_stars(ax1l, pval, alpha=0.05, star_loc=[2,1950], add_line=True, line_locs=[1,3,1900,1900],
                                            offset_btw_star_n_line=0.1, color='grey', coord_system='data',)
        for s, violin in enumerate(parts['bodies']):
            sq = [1,5,15][s]
            color = color_squares_EI[clamp][sq]
            color = mpl.colors.to_rgba(color, alpha=0.6)
            violin.set_facecolor(color)
            violin.set_edgecolor(color)
            labels.append((mpatches.Patch(color=color), f'{P}, {sq} squares'))
        for partname in ['cquantiles','cmedians',]:
            vp = parts[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(0.5)

    ax1l.set_ylabel('First Peak CA1 Response (pA)')
    ax1l.set_xlabel('Number of Squares per Pattern')
    ax1l.legend(*zip(*labels), loc='lower left', bbox_to_anchor=(0,0), frameon=True, ncols=1, fontsize=14)
    # remove top and right spines
    ax1l.spines['top'].set_visible(False)
    ax1l.spines['right'].set_visible(False)
    # set ticklabels
    ax1l.set_xticks([1,2,3])
    ax1l.set_xticklabels(['1','5','15'])
    # set yticks
    ax1l.set_yticks([-4000,-2000, 0,2000])
    sns.despine(ax=ax1l, top=True, right=True, left=False, bottom=False, offset=10, trim=True)  

    # # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # # Fig 1m: Trend in the field response across pulses
    ax1m.text(-0.1, 1.1, 'M', transform=ax1m.transAxes, size=20, weight='bold')
    sample_cell = 111  #screened_vc_cells[2] # 111, 7492, 1621
    celldataE = xc_FS_longdf_slice[ (xc_FS_longdf_slice['clampMode']=='VC') & (xc_FS_longdf_slice['cellID']==sample_cell) & (xc_FS_longdf_slice['clampPotential']==-70) & (xc_FS_longdf_slice['stimFreq']==20) & (xc_FS_longdf_slice['numSq']==15)]
    celldataI = xc_FS_longdf_slice[ (xc_FS_longdf_slice['clampMode']=='VC') & (xc_FS_longdf_slice['cellID']==sample_cell) & (xc_FS_longdf_slice['clampPotential']==  0) & (xc_FS_longdf_slice['stimFreq']==20) & (xc_FS_longdf_slice['numSq']==15)]
    time = np.linspace(0,1,20000)
    for i in range(celldataE.shape[0]):
        ax1m.plot(time, celldataE.iloc[i,49:20049], color=color_EI[-70], linewidth=0.2, alpha=0.5)
        ax1m.plot(time, celldataI.iloc[i,49:20049], color=color_EI[0], linewidth=0.2, alpha=0.5)
    # plot mean
    ax1m.plot(time, celldataE.iloc[:,49:20049].mean(axis=0), color=color_EI[-70], linewidth=1.5, alpha=0.8, label='EPSC')
    ax1m.plot(time, celldataI.iloc[:,49:20049].mean(axis=0), color=color_EI[0], linewidth=1.5, alpha=0.8, label='IPSC')
    # set legend inside the plot
    ax1m.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9), frameon=True) # legend removed to save space
    # remove legend
    ax1m.legend(loc='upper right', frameon=False)
    # add a floating scalebar
    sns.despine(ax=ax1m, left=True, right=True, top=True, bottom=True)
    # remove ticks
    ax1m.set_xticks([])
    ax1m.set_yticks([])
    plot_tools.add_floating_scalebar(ax1m, scalebar_origin=[0.0,0.8], xlength=0.1, ylength=200, labelx='100', labely='200', unitx='ms', unity='pA', fontsize=16, color='black', linewidth=2, pad=0.02, show_labels=True)
    # # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    for ax in [ax1a, ax1b, ax1c, ax1d, ax1e, ax1f, ax1g, ax1h, ax1i, ax1j, ax1k, ax1l, ax1m]:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(16)

    for ax in [ax1a, ax1b, ax1c, ax1d, ax1e, ax1f, ax1g, ax1h, ax1i, ax1j, ax1k, ax1l, ax1m]:
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)

    # # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # tight layout
    plt.tight_layout()
    # # save figure
    figure_name = 'Figure1'
    fig1.savefig(paper_figure_export_location /  (figure_name + '.png'), dpi=300, bbox_inches='tight')
    fig1.savefig(paper_figure_export_location /  (figure_name + '.svg'), dpi=300, bbox_inches='tight')
    fig1.savefig(paper_figure_export_location /  (figure_name + '.pdf'), dpi=300, bbox_inches='tight')


def extended_data_figure():
    # ## Generate image composites separately for high res
    # Fig 1A: Slice, polygon projection, and recording electrodes
    Fig1A_hr, ax1A_hr = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    ax1A_hr.text(-0.1, 1.1, 'A', transform=ax1A_hr.transAxes, size=20, weight='bold')
    image1_path = all_cells.project_path_root / r"Lab\Projects\EI_Dynamics\Analysis\paper_figure_matter\slice_electrode_expression_cropped_with_scalebar_blue_polygon.png"
    im1 = Image.open(image1_path)
    # get the size of the image
    im1_width, im1_height = im1.size
    # get the aspect ratio of the ax1A_hr axes object
    ax1a_ratio = ax1A_hr.get_window_extent().width / ax1A_hr.get_window_extent().height

    # change the axis limits so that verticle side of the image fits the axis and horizontal side is aligned left on the axis
    ax1A_hr.set_ylim(0, im1_height)
    ax1A_hr.set_xlim(0, im1_height*ax1a_ratio)
    # plot the image
    ax1A_hr.imshow(im1, extent=[0, im1_width, 0, im1_height], aspect=1)
    ax1A_hr.axis('off')

    # # save figure
    figure_name = 'Figure1A_highres'
    Fig1A_hr.savefig(paper_figure_export_location /  (figure_name + '.png'), dpi=300, bbox_inches='tight')
    Fig1A_hr.savefig(paper_figure_export_location /  (figure_name + '.svg'), dpi=300, bbox_inches='tight')
    Fig1A_hr.savefig(paper_figure_export_location /  (figure_name + '.pdf'), dpi=300, bbox_inches='tight')
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Fig 1B: Grid Pattern in Space overlaid on CA3 slice at 40x
    Fig1B_hr, ax1B_hr = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    ax1B_hr.text(-0.1, 1.1, 'B', transform=ax1B_hr.transAxes, size=20, weight='bold')
    image2_path = all_cells.project_path_root / r'Lab\Projects\EI_Dynamics\Analysis\paper_figure_matter\CA3-polygonFrame_figure_with_cellboundaries.png'
    im2 = Image.open(image2_path)
    im2_width, im2_height = im2.size
    # ax1B_hr.imshow(im2)

    ax1B_hr.set_ylim(0, im1_height)
    ax1B_hr.set_xlim(0, im1_height*ax1a_ratio)
    ax1B_hr.imshow(im2, extent=[0, im1_width, 0, im1_height], aspect=1)
    ax1B_hr.axis('off')
    # ax1B_hr.set_anchor('W')

    # # save figure
    figure_name = 'Figure1B_highres'
    Fig1B_hr.savefig(paper_figure_export_location /  (figure_name + '.png'), dpi=300, bbox_inches='tight')
    Fig1B_hr.savefig(paper_figure_export_location /  (figure_name + '.svg'), dpi=300, bbox_inches='tight')
    Fig1B_hr.savefig(paper_figure_export_location /  (figure_name + '.pdf'), dpi=300, bbox_inches='tight')


    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Fig extended data figure 1: CA3 heatmap
    plt.close('all')
    Fig1s1, [ax1s1, ax1s2] = plt.subplots(2, 1, figsize=(10, 10))
    ax1s1.text(-0.1, 1.1, '8A', transform=ax1s1.transAxes, size=20, weight='bold')
    ax1s2.text(-0.1, 1.1, '8B', transform=ax1s2.transAxes, size=20, weight='bold')

    ax1s1.invert_yaxis()
    # import heatmap data
    df_CA3_heatmap = pd.read_hdf(r"parsed_data\second_order\CA3_recording_3161_grid_response_pivot.h5")
    grid_aspect_ratio = pattern_index.polygon_frame_properties['aspect_ratio']
    h = 8
    w = grid_aspect_ratio*h
    pulse = 0

    # make heatmaps
    CA3p0 = pd.DataFrame(df_CA3_heatmap[pulse])
    CA3p0.reset_index(inplace=True)
    # rename column '0' as 'pA'
    CA3p0.rename(columns={pulse: 'pA'}, inplace=True)
    CA3p0.pop('sweep')
    CA3p0['x'] = (CA3p0['coord'] -1) // 24
    CA3p0['y'] = (CA3p0['coord']-1) % 24
    #  pivot
    CA3p0 = CA3p0.pivot(index='x', columns='y', values='pA')

    # heatmap
    ax1s1.imshow(CA3p0, cmap='viridis', aspect='auto', interpolation='nearest', origin='upper')
    
    # add a small asterisk in the middle of the heatmap to show the recording spot
    ax1s1.text(11.5, 12, '*', color='white', fontsize=16, ha='center', va='center')
    
    # add colorbar to the axis correponding the heatmap image
    cbar = plt.colorbar(ax1s1.get_children()[1], ax=ax1s1, orientation='vertical', pad=0.01)
    cbar.set_label('Optical Depolarization (pA)', rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=16)
    # add text to the plot on top left corner of the heatmap in white color
    ax1s1.text(0.0, 0.9, 'Basal', transform=ax1s1.transAxes, size=16,  color='white')
    # add text to the plot on bottom left corner of the heatmap
    ax1s1.text(0.0, 0.05, 'Apical', transform=ax1s1.transAxes, size=16, color='white')
    ax1s1.set_title('CA3 Response to grid spots')
    # make a horizontal line to show scale
    ax1s1.plot([1, 3], [4, 4], color='white', linewidth=2)
    ax1s1.plot([1, 1], [4, 6], color='white', linewidth=2)
    ax1s1.text(1.5, 2.5, '26 $\mu$m', color='white', fontsize=12)

    # a histogram of CA3 responses to grid patterns of different sizes
    ax1s2.set_ylim([0, 30])
    df_temp = xc_all_shortdf[ (xc_all_shortdf['cellID']==3161) ]
    df_temp2 = df_temp.copy()
    # if the cell_fpr_max is greater than 15, make it 25
    df_temp['cell_fpr_max'] = np.where(df_temp['cell_fpr_max']>15, 25, df_temp['cell_fpr_max'])
    # plot a relationship between cell_fpr and field_fpr using a scatter plot (options: box, strip, swarm, violin, boxen)
    sns.stripplot(data=df_temp[df_temp['cell_fpr_max']>15], y="cell_fpr_max",  x='numSq', jitter=0.2, ax=ax1s2, hue='numSq', palette=color_squares, linewidth=0.5, size=10, alpha=0.5, zorder=5)
    sns.violinplot(data=df_temp2, x="numSq", y="cell_fpr_max", hue="numSq", palette=color_squares, ax=ax1s2, alpha=0.0, inner=None, split=False, dodge=False, zorder=3)
    [part.set_edgecolor((part.get_edgecolor()[:],  0)) for part in ax1s2.get_children() if isinstance(part, mpl.collections.PolyCollection)]
    stat_annotate.pairwise_annotate_violin_plot(ax1s2, df_temp, x='numSq', y='cell_fpr_max', stat=mannwhitneyu, add_line=True, offset=0.05, color='grey', coord_system='data', fontsize=12, zorder=10)

    sns.violinplot(data=df_temp[df_temp['cell_fpr_max']<15], x="numSq", y="cell_fpr_max", hue="numSq", palette=color_squares, ax=ax1s2, alpha=0.5, inner='quart', split=False, dodge=False, zorder=3)
    ax1s2.set_ylabel('CA3 cell Response (mV)')
    ax1s2.set_xlabel('Number of Squares per Pattern')
    yticks = [0, 5, 10, 15, 20, 25]
    yticklabels = [0, 5, 10, 15, 20, 'AP']
    ax1s2.set_yticks(yticks)
    ax1s2.set_yticklabels(yticklabels)
    # despine
    sns.despine(ax=ax1s2, top=True, right=True, left=False, bottom=False, offset=10, trim=True)
    ax1s2.legend([],[], frameon=False)

    # tight layout
    plt.tight_layout()

    # save figure
    figure_name = 'Figure1_ExtData1_CA3response'
    Fig1s1.savefig(paper_figure_export_location /  (figure_name + '.png'), dpi=300, bbox_inches='tight')
    Fig1s1.savefig(paper_figure_export_location /  (figure_name + '.svg'), dpi=300, bbox_inches='tight')
    Fig1s1.savefig(paper_figure_export_location /  (figure_name + '.pdf'), dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
    extended_data_figure()