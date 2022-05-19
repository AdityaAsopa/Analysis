import pandas as pd
import matplotlib
#matplotlib.use("Agg") # to suppress default matplotlib bitmap backend engine (QtAgg), prevents resource overload

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os
sns.set_context("paper")


def dataframe_to_plots(cellpickleFile,ploty="PeakRes",gridRow="NumSquares",gridColumn="StimFreq",plotby="EI",clipSpikes=False):
    with open(cellpickleFile,'rb') as fin:
        x = pickle.load(fin)
    resp = x.response

    # load file
    # resp = cell.response
    resp1 = resp.query("NumSquares != 1")  # dropping 1sq sweeps from plots

    unit = resp1.iloc[1,resp1.columns.get_loc("Unit")]

    # tag: improve feature (to permanently remove hard coding the indices of calculated values,
    # use of tiers in pandas dataframe needs to be implemented)
    
    if ploty == "PeakRes":
        vals = np.arange(1,9)
        valName = "PSR Value (" + unit + ")"
    elif ploty == "OnsetDelay":
        vals = np.arange(9, 17)
        valName = "Onset Delay (ms)"
    elif ploty == "PeakTime":
        vals = np.arange(9,17)
        valName = "Time of Peak (ms)"
    elif ploty == "AUC":
        vals = np.arange(17,25)
        valName = "AuC of Responses (a.u.)"
    else:
        print("Don't know what to plot. Plotting peak responses.")
        vals = np.arange(1,9)
        valName = "PSR Value (" + unit + ")"

    # Separate the identifier variables from valua variables by melting the dataframe
    respMelt = pd.melt(resp1,id_vars=["Repeat","StimFreq","EI","PatternID","NumSquares"],value_vars=vals,var_name='PulseIndex', value_name=valName)

    if clipSpikes and valName == "PSR Value (mV)":
        respMelt.loc[respMelt["PSR Value (mV)"] >= 30, "PSR Value (mV)"] = 30

    '''Initialize a grid of plots with an axis each for different fields'''
    plt.figure()
    grid = sns.FacetGrid(respMelt, row=gridRow, col=gridColumn, hue=plotby, palette="viridis", legend_out=True)

    # Draw a scatter plot to show the PSP/PSC amplitude
    grid.map(plt.scatter,"PulseIndex",valName,marker="o")  # additional kwargs for lineplot: estimator=None,lw=1,units="Repeat", not working though
    # grid.map(sns.scatterplot,data=respMelt,x="pulseIndex",y=valName,)#estimator=None,units="Repeat"
    grid.add_legend()

    # # Adjust the tick positions and labels
    grid.set(xlim=(vals[0] - 1, vals[-1] + 1))
    if clipSpikes and unit == 'mV':
        grid.set(ylim=(0, 1.1 * np.max(respMelt[valName])))  # FIXME: hardcoding the plot limits for clipping spikes

    # '''trying out relplot'''
    # fig = sns.relplot(
    # data=respMelt,
    # col=gridColumn,row=gridRow,
    # x="pulseIndex", y=valName,
    # hue="PatternID", style="EI",
    # kind="scatter",estimator=None,units="PatternID",
    # palette="viridis"
    # )

    # '''lineplot'''
    # grid.map(sns.lineplot,"pulseIndex",valName)

    # Save and show figure
    exptDir = os.path.dirname(cellpickleFile)
    imageFile = exptDir + "\\" + "plot_" + ploty + "-vs-" + gridRow + "-for-" + plotby + ".png"
    plt.savefig(imageFile)
    print("saved plotfile: ",imageFile)
    plt.close('all')

    '''## Initialize a grid of plots with an axis each for different fields
    plt.figure()
    grid2 = sns.FacetGrid(respMelt, col=gridColumn, row=gridRow, hue=plotby,palette="viridis")

    grid2.map(sns.lineplot,"pulseIndex",valName)
    grid2.add_legend()

    # # Adjust the tick positions and labels
    grid2.set(xlim=(vals[0]-1,vals[-1]+1))
    if clipSpikes:
        grid2.set(ylim=(-200,200))


    # Save and show figure
    exptDir = os.path.dirname(cellpickleFile)
    imageFile2 = exptDir + "\\" + "plot_" + ploty + "-vs-" + plotby + "-lineplot.png"
    plt.savefig(imageFile2)

    Initialize a grid of plots with an axis each for different fields
    plt.figure()
    grid3 = sns.FacetGrid(respMelt, col=gridColumn, row=gridRow, hue=plotby,palette="viridis")

    # Draw a scatter plot to show the PSP/PSC amplitude
    grid3.map(plt.scatter,"pulseIndex",valName,marker="o")
    plt.legend()

    grid3.map(sns.lineplot,"pulseIndex",valName)
    grid3.add_legend()

    # # Adjust the tick positions and labels
    grid3.set(xlim=(vals[0]-1,vals[-1]+1))
    if clipSpikes:
        grid3.set(ylim=(-200,200))

    # Save and show figure
    exptDir = os.path.dirname(cellpickleFile)
    imageFile3 = exptDir + "\\" + "plot_" + ploty + "-vs-" + plotby + "-scatlineplot.png"
    plt.savefig(imageFile3)
    plt.close('all')'''

    return grid

# FIXME: remove hardcoded variables, fields, and values
