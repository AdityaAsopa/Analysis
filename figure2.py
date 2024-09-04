# Short-term dynamics of Excitation-Inhibition Balance in Hippocampal CA3-CA1 circuit
# Aditya Asopa, Upinder Singh Bhalla, NCBS
# Figure 2

# Imports -----------------------------------------------------------------------------------------------
from   pathlib      import Path

import numpy                as np
import matplotlib           as mpl
import matplotlib.pyplot    as plt
import seaborn              as sns
import pandas               as pd

from scipy.stats   import kruskal, wilcoxon, mannwhitneyu, ranksums
from scipy.optimize import curve_fit

# from eidynamics     import utils, data_quality_checks, ephys_classes, plot_tools, expt_to_dataframe
# from eidynamics     import pattern_index
# from eidynamics     import abf_to_data
from eidynamics.fit_PSC     import find_sweep_expected
# from Findsim        import tab_presyn_patterns_LR_43
# import parse_data
from eidynamics     import utils, plot_tools
import all_cells
import plotFig2
from stat_annotate import *

# sns.set_context('paper')
# sns.set_context('paper')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['svg.fonttype'] = 'none'

# make a colour map viridis
viridis = mpl.colormaps["viridis"]
flare   = mpl.colormaps["flare"]
crest   = mpl.colormaps["crest"]
magma   = mpl.colormaps["magma"]
edge    = mpl.colormaps['edge']

color_E = "flare"
color_I = "crest"
color_freq = {1:magma(0.05), 5:magma(0.1), 10:magma(0.2), 20:magma(.4), 30:magma(.5), 40:magma(.6), 50:magma(.7), 100:magma(.9)}
color_squares = color_squares = {1:viridis(0.2), 5:viridis(.4), 7:viridis(.6), 15:viridis(.8), 20:viridis(1.0)}
color_EI = {-70:flare(0), 0:crest(0)}

Fs = 2e4

freq_sweep_pulses = np.arange(9)

# Data -----------------------------------------------------------------------------------------------
figure_raw_material_location = Path(r"paper_figure_matter\\")
paper_figure_export_location = Path(r"paper_figures\\Figure2\\")
data_path                    = Path(r"parsed_data\\")

# Load the dataset
freq_sweep_vc_datapath =  r"C:\Users\adity\OneDrive\NCBS\Lab\Projects\EI_Dynamics\Analysis\parsed_data\all_cells_FreqSweep_VC_long.h5" 
df = pd.read_hdf(freq_sweep_vc_datapath, key='data')

# expanded dataframe (processed dataframe with metadata and analysed params)
expanded_data_path = r"C:\Users\adity\OneDrive\NCBS\Lab\Projects\EI_Dynamics\Analysis\parsed_data\all_cells_FreqSweep_combined_expanded.h5"
xc_FS_analyseddf = pd.read_hdf(expanded_data_path, key='data')

# Screening -----------------------------------------------------------------------------------------------
# data screening based on dataflag_fields
dfslice = df[
            (df['location'] == 'CA1') &
            (df['numSq'].isin([1,5,15])) &
            (df['AP'] == 0) &
            (df['IR'] >50) & (df['IR'] < 300) &
            (df['tau'] < 40) & 
            (df['intensity'] == 100) &
            (df['pulseWidth'] == 2) &
            # (df['sweepBaseline'] < -50) &
            (df['condition'] == 'Control') &
            (df['ch0_response']==1) &
            # (df['spike_in_stim_period'] == 0) &
            (df['spike_in_baseline_period'] == 0) &
            (df['ac_noise_power_in_ch0'] < 40) 
        ]

vc_screened_trialIDs = dfslice['trialID'].unique()

print(f"Unique cells in screened data: {dfslice['cellID'].nunique()}")
print(f"Unique sweeps in screened data: {dfslice['trialID'].nunique()}")

df3 = xc_FS_analyseddf[xc_FS_analyseddf['trialID'].isin(vc_screened_trialIDs)]

# Plotting -----------------------------------------------------------------------------------------------
def main(plot_kind='strip'):
    # plot_kind = 'strip' # 'line' or 'violin' or 'strip'
    column_name_abbreviations = ['pc','pcn','ac','sc','dc','pf','pfn']

    # ---------------------------------------------------------------------------------------------------------------------------------
    # Setup the figure
    w, h =  21, 29.7
    # make a figure of 7 subplots in 2 columns, in which the first subplot spans two columns and first row, and the rest span 1row, 1 column each
    Fig2, ax2 = plt.subplot_mosaic([['A', 'B'],['C', 'C'],['D','D'],['E', 'E'],['F', 'G'],['H','I'],['J','K']], figsize=(w, h), )
    # change the spacing between plots
    Fig2.subplots_adjust(hspace=1.0, wspace=0.25)
    # linearize the ax2 list
    ax2 = [ax2[key] for key in ax2.keys()]

    # ---------------------------------------------------------------------------------------------------------------------------------
    # Fig 2A: STP vs numSq for a sample cell, cp = -70mV
    ax2[0].text(-0.1, 1.1, 'A', transform=ax2[0].transAxes, size=20, weight='bold')
    cell =  7492
    sq   =  15
    cp   = -70        # clamping potential subset for the plot
    # f = 20          # stimFreq, Hz
    to_plot = [f'pcn{i}' for i in freq_sweep_pulses]
    df_temp = df3[ (df3['cellID'] == cell) & (df3['numSq'] == sq) & (df3['clampPotential'] == cp)] #
    df_melt = pd.melt( df_temp, id_vars=['cellID', 'stimFreq', 'clampPotential', 'numSq', 'patternList'], value_vars=to_plot, var_name='pulseIndex', value_name='peak_response',)

    pairwise_draw_and_annotate_line_plot(   ax2[0], df_melt, x='pulseIndex', y='peak_response', hue='stimFreq', draw=True, kind=plot_kind, palette=color_E, 
                                            stat_across='hue', stat=kruskal, skip_first_xvalue=True, annotate_wrt_data=False, offset_btw_star_n_line=0.1, color='grey', coord_system='data', fontsize=12, zorder=10,)
    ax2[0].set_xlabel('Pulse Index')
    # set xtick location and labels
    ax2[0].set_xticks(range(9))
    ax2[0].set_xticklabels(freq_sweep_pulses)

    # y-axis
    ax2[0].set_ylabel('Norm. Response')
    ax2[0].set_ylim(0,3)
    # no legend
    # ax2[0].legend([],[], frameon=False)

    # # get the legend labels of ax2d_top and add ' Sq' to each one
    # handles, labels = ax2['A'].get_legend_handles_labels()
    # labels = [label + ' Sq.' for label in labels]
    # ax2d_top.legend(handles, labels, loc='upper right', borderaxespad=0., frameon=False)
    # # add a text in the top left corner of ax2d_top showing '20 Hz' in color = color_freq[20]
    # ax2d_top.text(0.0, 0.9, f'{f} Hz', transform=ax2d_top.transAxes, size=12, color=color_freq[f], zorder=10)

    # ---------------------------------------------------------------------------------------------------------------------------------
    # Fig 2B: STP vs numSq for the same cell, cp = 0mV
    ax2[1].text(-0.1, 1.1, 'B', transform=ax2[1].transAxes, size=20, weight='bold')

    cp = 0        # clamping potential subset for the plot
    # f = 20          # stimFreq, Hz
    to_plot = [f'pcn{i}' for i in freq_sweep_pulses]
    df_temp = df3[ (df3['cellID'] == cell) & (df3['numSq'] == sq)  & (df3['clampPotential'] == cp)] #
    df_melt = pd.melt( df_temp, id_vars=['cellID', 'stimFreq', 'clampPotential', 'numSq', 'patternList'], value_vars=to_plot, var_name='pulseIndex', value_name='peak_response',)

    pairwise_draw_and_annotate_line_plot(   ax2[1], df_melt, x='pulseIndex', y='peak_response', hue='stimFreq', draw=True, kind=plot_kind, palette=color_I, 
                                            stat_across='hue', stat=kruskal, skip_first_xvalue=True, annotate_wrt_data=False, offset_btw_star_n_line=0.1, color='grey', coord_system='data', fontsize=12, zorder=10,)
    # plot_tools.simplify_axes( ax2[1], splines_to_keep=['bottom'], axis_offset=10, remove_ticks=False, xtick_locs=range(9), xtick_labels=freq_sweep_pulses, ytick_locs=range(3), ytick_labels=range(3),)
    ax2[1].set_xlabel('Pulse Index')
    # set xtick location and labels
    ax2[1].set_xticks(range(9))
    ax2[1].set_xticklabels(freq_sweep_pulses)
    # remove y-ticks and y-axis label
    ax2[1].set_ylim(0,3)
    ax2[1].set_ylabel('')
    # ax2[1].legend([],[], frameon=False)


    # ---------------------------------------------------------------------------------------------------------------------------------
    # Fig 2C: STP vs numSq for all screnned cells cp = -70mV
    ax2[2].clear()
    ax2[2].text(-0.1, 1.1, 'C', transform=ax2[2].transAxes, size=20, weight='bold')

    cp = -70        # clamping potential subset for the plot
    # f = 20          # stimFreq, Hz
    to_plot = [f'pcn{i}' for i in freq_sweep_pulses]
    df_temp = df3[ (df3['numSq'] == sq) & (df3['clampPotential'] == cp) ] #
    df_melt = pd.melt( df_temp, id_vars=['cellID', 'stimFreq', 'clampPotential', 'numSq', 'patternList'], value_vars=to_plot, var_name='pulseIndex', value_name='peak_response',)

    pairwise_draw_and_annotate_line_plot(   ax2[2], df_melt, x='pulseIndex', y='peak_response', hue='stimFreq', draw=True, kind='strip', palette=color_E, 
                                            stat_across='hue', stat=kruskal, skip_first_xvalue=True, annotate_wrt_data=False, offset_btw_star_n_line=0.1, color='grey', coord_system='data', fontsize=12, zorder=10,)
    # plot_tools.simplify_axes( ax2[2], splines_to_keep=['left'], axis_offset=10, remove_ticks=False, xtick_locs=range(9), xtick_labels=freq_sweep_pulses, ytick_locs=range(5), ytick_labels=range(5),)
    ax2[2].set_ylabel('Norm. Response')
    # ax2[2].set_ylim(0,5)
    # remove x-ticks and x-axis label
    ax2[2].set_xlabel('')
    ax2[2].set_xticklabels([])
    # ax2[2].legend([],[], frameon=False)

    # ---------------------------------------------------------------------------------------------------------------------------------
    # Fig 2D: STP vs numSq for all screnned cells for cp = 0mV
    ax2[3].clear()
    ax2[3].text(-0.1, 1.1, 'D', transform=ax2[3].transAxes, size=20, weight='bold')

    cp = 0        # clamping potential subset for the plot
    # f = 20          # stimFreq, Hz
    to_plot = [f'pcn{i}' for i in freq_sweep_pulses]
    df_temp = df3[  (df3['numSq'] == sq) & (df3['stimFreq'] <100 )& (df3['clampPotential'] == cp)  ] #
    df_melt = pd.melt( df_temp, id_vars=['cellID', 'stimFreq', 'clampPotential', 'numSq', 'patternList'], value_vars=to_plot, var_name='pulseIndex', value_name='peak_response',)

    pairwise_draw_and_annotate_line_plot(   ax2[3], df_melt, x='pulseIndex', y='peak_response', hue='stimFreq', draw=True, kind='strip', palette=color_I, 
                                            stat_across='hue', stat=kruskal, skip_first_xvalue=True, annotate_wrt_data=False, offset_btw_star_n_line=0.1, color='grey', coord_system='data', fontsize=12, zorder=10,)
    # plot_tools.simplify_axes( ax2[3], splines_to_keep=['bottom', 'left'], axis_offset=10, remove_ticks=False, xtick_locs=range(9), xtick_labels=freq_sweep_pulses, ytick_locs=range(5), ytick_labels=range(5),)
    ax2[3].set_xlabel('Pulse Index')
    ax2[3].set_ylabel('Norm. Response')
    ax2[3].set_ylim(0,5)

    # ---------------------------------------------------------------------------------------------------------------------------------
    # Upi's panels
    _ = plotFig2.main(df, Fig2, ax2[4:], cellNum=cell, numSq=sq)

    # ---------------------------------------------------------------------------------------------------------------------------------
    for ax in ax2:
        sns.despine(fig=Fig2, ax=ax, top=True, right=True, left=False, bottom=False, offset=0.1, trim=True)

    # Save Fig-------------------------------------------------------------------------------------------------------------------------
    # Save figure2
    figure_name = 'Figure2_with_legend'
    Fig2.savefig(paper_figure_export_location / (figure_name + '.png'), dpi=300, bbox_inches='tight')
    Fig2.savefig(paper_figure_export_location / (figure_name + '.svg'), dpi=300, bbox_inches='tight')


def old():
    # aspect ration of the figure = 1
    w, h = [15, 9]
    fig2, [[ax2a, ax2b, ax2c],[ax2d_top, ax2e_top, ax2f_top],[ax2d_bottom, ax2e_bottom, ax2f_bottom]] = plt.subplots(3,3, figsize=(w, h), sharey=False)
    fig2.subplots_adjust(hspace=0.5, wspace=0.5)
    plot_kind = 'violin' # 'line' or 'violin' or 'strip'

    column_name_abbreviations = ['pc','pcn','ac','sc','dc','pf','pfn']
    cell =  7492
    sq   =  15

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Fig 2A,B,C: Upi's deconvolution fits
    ax2a.text(-0.1, 1.1, 'A', transform=ax2a.transAxes, size=20, weight='bold')
    ax2b.text(-0.1, 1.1, 'B', transform=ax2b.transAxes, size=20, weight='bold')
    ax2c.text(-0.1, 1.1, 'C', transform=ax2c.transAxes, size=20, weight='bold')

    sweepnum = 0
    df_temp = dfslice[(dfslice['cellID'] == cell) & (dfslice['numSq'] == sq)]
    sweep = df_temp.iloc[sweepnum, :]
    tracecell, tracestim, stimfreq = sweep[49:20049], sweep[40049:60049], sweep['stimFreq']

    ax2a.plot(np.linspace(0, 1, 20000),       tracecell, label='recording') # Raw data
    ax2a.plot(np.linspace(0, 1, 20000), 100 * tracestim, label='Stim')      # Stimulus

    # find the expected response, add to the axes
    fits, _, _ = find_sweep_expected(tracecell, stimfreq, fig2, [ax2a, ax2b, ax2c])

    # simplify axes
    plot_tools.simplify_axes(ax2a, splines_to_keep=['bottom', 'left'], axis_offset=10, remove_ticks=False, xtick_locs=[0, 0.2, 0.4, 0.6, 0.8, 1.0], xtick_labels=[0, 0.2, 0.4, 0.6, 0.8, 1.0], ytick_locs=[-150, -100, -50, 0, 50], ytick_labels=[-150, -100, -50, 0, 50],)
    ax2a.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=0., frameon=False)
    ax2a.set_ylabel('membrane potential (pA)')

    plot_tools.simplify_axes(ax2b, splines_to_keep=['bottom', 'left'], axis_offset=10, remove_ticks=False, xtick_locs=[0, 0.2, 0.4, 0.6, 0.8, 1.0], xtick_labels=[0, 0.2, 0.4, 0.6, 0.8, 1.0], ytick_locs=[-50, 0, 50], ytick_labels=[-50, 0, 50],)
    ax2b.legend([], frameon=False)
    ax2b.set_ylabel('Residual (pA)')

    plot_tools.simplify_axes( ax2c, splines_to_keep=['bottom', 'left'], axis_offset=10, remove_ticks=False, xtick_locs=range(9), xtick_labels=freq_sweep_pulses, ytick_locs=[0, 1.0, 2.0], ytick_labels=[0, 1.0, 2.0],)
    ax2c.legend([], frameon=False)
    ax2c.set_ylabel('Normalized Response')

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Fig 2D: STP vs numSq
    ax2d_top.text(-0.1, 1.1, 'D', transform=ax2d_top.transAxes, size=20, weight='bold')

    cp = -70        # clamping potential subset for the plot
    f = 20          # stimFreq, Hz
    to_plot = [f'pcn{i}' for i in freq_sweep_pulses]
    df_temp = df3[ (df3['cellID'] == cell) & (df3['stimFreq'] == f) & (df3['clampPotential'] == cp) ] 
    df_melt = pd.melt( df_temp, id_vars=['cellID', 'stimFreq', 'clampPotential', 'numSq', 'patternList'], value_vars=to_plot, var_name='pulseIndex', value_name='peak_response',)

    pairwise_draw_and_annotate_line_plot(   ax2d_top, df_melt, x='pulseIndex', y='peak_response', hue='numSq', draw=True, kind=plot_kind, palette=color_squares, 
                                            stat_across='hue', stat=kruskal, skip_first_xvalue=True, annotate_wrt_data=False, offset_btw_star_n_line=0.1, color='grey', coord_system='data', fontsize=12, zorder=10,)
    plot_tools.simplify_axes( ax2d_top, splines_to_keep=['bottom', 'left'], axis_offset=10, remove_ticks=False, xtick_locs=range(9), xtick_labels=freq_sweep_pulses, ytick_locs=[0, 1, 2], ytick_labels=[0, 1, 2],)
    ax2d_top.set_xlabel('Pulse Index')
    ax2d_top.set_ylabel('Norm. Response')

    # get the lgened labels of ax2d_top and add ' Sq' to each one
    handles, labels = ax2d_top.get_legend_handles_labels()
    labels = [label + ' Sq.' for label in labels]
    ax2d_top.legend(handles, labels, loc='upper right', borderaxespad=0., frameon=False)
    # add a text in the top left corner of ax2d_top showing '20 Hz' in color = color_freq[20]
    ax2d_top.text(0.0, 0.9, f'{f} Hz', transform=ax2d_top.transAxes, size=12, color=color_freq[f], zorder=10)
    ax2d_top.text(0.2, 0.9, f'{cp} mV', transform=ax2d_top.transAxes, size=12, color=color_EI[cp], zorder=10)

    ### Fig 2D_Bottom: STP vs numSq
    # field plot in the bottom subplot
    to_plot = [f'pfn{i}' for i in range(9)]

    df_temp = df3[ (df3['cellID'] == cell) & (df3['stimFreq'] == f) & (df3['clampPotential'] == cp) ]
    df_melt = pd.melt( df_temp, id_vars=['cellID', 'stimFreq', 'clampPotential', 'numSq', 'patternList'], value_vars=to_plot, var_name='pulseIndex', value_name='peak_response',)

    pairwise_draw_and_annotate_line_plot(   ax2d_bottom, df_melt, x='pulseIndex', y='peak_response', hue='numSq', draw=True, kind=plot_kind, palette=color_squares,
                                            stat_across='hue', stat=kruskal, skip_first_xvalue=True, annotate_wrt_data=False, offset_btw_star_n_line=0.1, color='grey', coord_system='data', fontsize=12, zorder=10,)
    plot_tools.simplify_axes( ax2d_bottom, splines_to_keep=['bottom', 'left'], axis_offset=10, remove_ticks=False, xtick_locs=range(9), xtick_labels=freq_sweep_pulses, ytick_locs=[0, 1, 2], ytick_labels=[0, 1, 2],)

    ax2d_bottom.set_xlabel('Pulse Index')
    ax2d_bottom.set_ylabel('Norm. Response')
    ax2d_bottom.legend([], frameon=False)

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Fig 2E: STP vs Frequency
    ax2e_top.text(-0.1, 1.1, 'E', transform=ax2e_top.transAxes, size=20, weight='bold')

    s = 5  # squares
    cp = -70
    to_plot = [f'pcn{i}' for i in freq_sweep_pulses]
    df_temp = df3[ (df3['cellID'] == cell) & (df3['clampPotential'] == cp) & (df3['numSq'] == s) ]
    df_melt = pd.melt( df_temp[df_temp['stimFreq'] < 100], id_vars=['cellID', 'stimFreq', 'clampPotential', 'numSq', 'patternList'], value_vars=to_plot, var_name='pulseIndex', value_name='peak_response',)

    pairwise_draw_and_annotate_line_plot(   ax2e_top, df_melt, x='pulseIndex', y='peak_response', hue='stimFreq', draw=True, kind='violin', palette=color_freq, stat_across='hue',
                                            stat=kruskal, skip_first_xvalue=True, annotate_wrt_data=False, offset_btw_star_n_line=0.1, color='grey', coord_system='data', fontsize=12, zorder=10,)
    plot_tools.simplify_axes( ax2e_top, splines_to_keep=['bottom', 'left'], axis_offset=10, remove_ticks=False, xtick_locs=range(9), xtick_labels=freq_sweep_pulses, ytick_locs=[0, 1, 2], ytick_labels=[0, 1, 2],)
    ax2e_top.set_xlabel('Pulse Index')
    ax2e_top.set_ylabel('Norm. Response')

    # get the lgened labels of ax2e_top and add ' Hz' to each one
    handles, labels = ax2e_top.get_legend_handles_labels()
    labels = [label + ' Hz' for label in labels]
    ax2e_top.legend(handles, labels, loc='upper right', borderaxespad=0., frameon=False)

    # add a text in the top left corner of ax2d_top showing '5 Sq' in color = color_squares[5]
    ax2e_top.text(0.0, 0.9, f'{s} Sq', transform=ax2e_top.transAxes, size=12, color=color_squares[s], zorder=10)
    ax2e_top.text(0.2, 0.9, f'{cp} mV', transform=ax2e_top.transAxes, size=12, color=color_EI[cp], zorder=10)


    ### 2E Bottom: Field plot in the bottom
    to_plot = [f'pfn{i}' for i in range(9)]

    df_temp = df3[ (df3['cellID'] == cell) & (df3['clampPotential'] == cp) & (df3['numSq'] == s) ]
    df_melt = pd.melt( df_temp[df_temp['stimFreq'] < 100], id_vars=['cellID', 'stimFreq', 'clampPotential', 'numSq', 'patternList'], value_vars=to_plot, var_name='pulseIndex', value_name='peak_response',)

    pairwise_draw_and_annotate_line_plot(   ax2e_bottom, df_melt, x='pulseIndex', y='peak_response', hue='stimFreq', draw=True, kind='violin', palette=color_freq,
                                            stat_across='hue', stat=kruskal, skip_first_xvalue=True, annotate_wrt_data=False, offset_btw_star_n_line=0.1, color='grey', coord_system='data', fontsize=12, zorder=10,)
    plot_tools.simplify_axes( ax2e_bottom, splines_to_keep=['bottom', 'left'], axis_offset=10, remove_ticks=False, xtick_locs=range(9), xtick_labels=freq_sweep_pulses, ytick_locs=[0, 1, 2], ytick_labels=[0, 1, 2],)
    ax2e_bottom.set_xlabel('Pulse Index')
    ax2e_bottom.set_ylabel('Norm. Response')
    ax2e_bottom.legend([], frameon=False)

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Fig 2F: STP vs E/I
    ax2f_top.text(-0.1, 1.1, 'F', transform=ax2f_top.transAxes, size=20, weight='bold')

    f = 20  # Hz
    s = 5  # squares
    to_plot = [f'pcn{i}' for i in freq_sweep_pulses]
    df_temp = df3[ (df3['cellID'] == cell) & (df3['stimFreq'] == f) & (df3['numSq'] == s)]
    df_melt = pd.melt( df_temp, id_vars=['cellID', 'stimFreq', 'clampPotential', 'numSq', 'patternList'], value_vars=to_plot, var_name='pulseIndex', value_name='peak_response',)

    pairwise_draw_and_annotate_line_plot(   ax2f_top, df_melt, x='pulseIndex', y='peak_response', hue='clampPotential', draw=True, kind='violin', palette=color_EI,
                                            stat_across='hue', stat=kruskal, skip_first_xvalue=True, annotate_wrt_data=False, offset_btw_star_n_line=0.1, color='grey', coord_system='data', fontsize=12, zorder=10,)

    plot_tools.simplify_axes( ax2f_top, splines_to_keep=['bottom', 'left'], axis_offset=10, remove_ticks=False, xtick_locs=range(9), xtick_labels=freq_sweep_pulses, ytick_locs=[0, 1, 2], ytick_labels=[0, 1, 2],)
    ax2f_top.set_xlabel('Pulse Index')
    ax2f_top.set_ylabel('Norm. Response')

    # get the lgened labels of ax2f_top and add ' mV' to each one
    handles, labels = ax2f_top.get_legend_handles_labels()
    labels = [label + ' mV' for label in labels]
    ax2f_top.legend(handles, labels, loc='upper right', borderaxespad=0., frameon=False)

    # add a text in the top left corner of ax2d_top showing '20 Hz' in color = color_freq[20]
    ax2f_top.text(0.0, 0.9, f'{f} Hz', transform=ax2f_top.transAxes, size=12, color=color_freq[f], zorder=10)
    ax2f_top.text(0.2, 0.9, f'{s} Sq', transform=ax2f_top.transAxes, size=12, color=color_squares[s], zorder=10)


    ### Fig 2F Bottom: Field plot in the bottom
    to_plot = [f'pfn{i}' for i in range(9)]
    df_temp = df3[ (df3['cellID'] == cell) & (df3['stimFreq'] == f) & (df3['numSq'] == s)]
    df_melt = pd.melt( df_temp, id_vars=['cellID', 'stimFreq', 'clampPotential', 'numSq', 'patternList'], value_vars=to_plot, var_name='pulseIndex', value_name='peak_response',)

    pairwise_draw_and_annotate_line_plot(   ax2f_bottom, df_melt, x='pulseIndex', y='peak_response', hue='clampPotential', draw=True, kind='violin', palette=color_EI,
                                            stat_across='hue', stat=kruskal, skip_first_xvalue=True, annotate_wrt_data=False, offset_btw_star_n_line=0.1, color='grey', coord_system='data', fontsize=12, zorder=10)
    plot_tools.simplify_axes(ax2f_bottom, splines_to_keep=['bottom','left'], axis_offset=10, remove_ticks=False, xtick_locs=range(9), xtick_labels=freq_sweep_pulses, ytick_locs=[0,1,2], ytick_labels=[0,1,2])
    ax2f_bottom.set_xlabel('Pulse Index')
    ax2f_bottom.set_ylabel('Norm. Response')
    ax2f_bottom.legend([], frameon=False)

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # save figure
    figure_name = 'Figure2_VC_old'
    fig2.savefig(paper_figure_export_location / 'misc' / (figure_name + '.png'), dpi=300, bbox_inches='tight')
    fig2.savefig(paper_figure_export_location / 'misc' / (figure_name + '.svg'), dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
    # old()