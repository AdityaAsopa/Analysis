#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/16 15:58
# @Author  : adityaasopa, Bhalla Lab, NCBS
# @Project : eidynamics

'''
Description: This script is used to generate figure 5 for the paper
### Figure 5
* Example E-I traces over a burst (train)
* Separate panel to compare E and I trace shapes (you can scale E to I and have their own scale bars)
* Plot field responses corresponding to E and I
* E/I ratio of successive pulses across freq and numSq
* Summary figure across cells
Note: This figure should explain why we dont see escape from E-I balance
* Current clamp summary
* Difference between diff freq over pulses
* Difference between numSq over pulses
 
### Figure 6: Why no escape (can be merged with figure 5)
* Comparison of current clamp response with response generated from a model cell with summated E-I currents
Likely reason: inhibition builds up slowly during the train
* Gabazine vs control, which pulse, selectivity
* for the sweeps that cause escape, which pulse index, which numSq, and which frequency are they from?
* EPSP peak time should be in the same range as spike peak time in the sweeps where there is escape

Figure Design: (Update 24th Jan 24)
Function to calculate SDN
Panel 1: SDN curves for first pulse: x='expected', y='observed', hue='stimFreq'
Panel 2: SDN curves for second pulse: x='expected', y='observed', hue='stimFreq'
Panel 3: SDN curves for last pulse: x='expected', y='observed', hue='stimFreq'
Panel 4: /gamma value across stimFreq: x='pulseIndex', y='gamma', hue='stimFreq'
Panel 5: Delay value across stimFreq: x='pulseIndex', y='delay', hue='stimFreq'
Panel 6: P(spike) across ?
'''

# imports
from   pathlib      import Path

import numpy                as np
import matplotlib           as mpl
import matplotlib.pyplot    as plt
import seaborn              as sns
import pandas               as pd

from scipy.signal   import find_peaks, peak_widths
from scipy.signal   import butter, bessel, decimate, sosfiltfilt
from scipy.stats    import kruskal, wilcoxon, mannwhitneyu, ranksums
from scipy.signal   import filter_design
# from scipy.optimize import curve_fit

# from PIL            import Image

import all_cells
from eidynamics     import utils, plot_tools
# from eidynamics     import ephys_classes, pattern_index, data_quality_checks, expt_to_dataframe
# from eidynamics     import abf_to_data
# from eidynamics     import fit_PSC
# from Findsim        import tab_presyn_patterns_LR_43
# import parse_data

sns.set_context('paper')
# %matplotlib widget
# %tb
# import plotly.express as px
# import plotly.graph_objects as go

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

Fs = 2e4

freq_sweep_pulses = np.arange(9)

# Load data
figure_raw_material_location = Path(r"paper_figure_matter\\")
paper_figure_export_location = Path(r"paper_figures\\")
data_path                    = Path(r"parsed_data\\")
cell_data_path               = Path(r"C:\Users\adity\OneDrive\NCBS\Lab\Projects\EI_Dynamics\Data\Screened_cells\\")


#