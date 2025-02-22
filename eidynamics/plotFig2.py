# Original Author: U. S. Bhalla, NCBS, Bangalore
# https://github.com/BhallaLab/STP_EI_paper_figs
# Modified by: Aditya Asopa, NCBS, Bangalore
# Date: 26 Jan 2024

'''
This Python script is designed to analyze voltage-clamp data and plot various figures related to short-term plasticity (STP) and the dynamics of excitatory and inhibitory synaptic currents. Here's a breakdown of the main functions:

1. `tauFit`: Fits an exponential curve to the kernel data and returns the fit parameters (a, tau).
2. `calcKernel`: Calculates the kernel from the input data and returns the peak value, kernel, baseline, and fit parameters.
3. `findStpScale`: Finds the scale factor for short-term plasticity (STP) from the kernel and fit parameters.
4. `findPkVal`: Finds the peak and trough values for a given stimulus in the input data.
5. `deconv`: Performs deconvolution on the input data and plots the results.
6. `plotFromKernel`: Plots the synthetic trace from the kernel and scale factors.
7. `plotA`: Plots the mean inhibitory and excitatory synaptic currents in panel A.
8. `plotB`: Plots the min-to-max ratio for inhibitory and excitatory deconvolved data in panel B.
9. `plotC`: Plots the peak-to-trough ratio for inhibitory and excitatory deconvolved data in panel C.
10. `plotDE`: Plots the min-to-max ratio for inhibitory and excitatory deconvolved data in panels D and E.
11. `plotFG`: Plots the min-to-max ratio for inhibitory and excitatory deconvolved data in panels F and G.
12. `innerAnalysis`: Performs inner analysis on the given data and returns the results.
13. `main`: The main function that sets up the analysis and calls the necessary functions to generate the plots.

The `main` function takes several arguments, including the input data (`alldat`), a figure object (`fig`), a list of axes objects (`axs`), a cell number (`cellNum`), and a number of sweeps (`numSq`). It then processes the data, calls the appropriate functions for analysis and plotting, and returns the updated figure, axes, and analysis results.

The script also includes some constants and helper functions for data processing and curve fitting.
'''

import pandas
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math
import scipy.optimize as sci

# datadir = "../../../2022/VC_DATA"
#datadir = "/home1/bhalla/adityaa/Lab/Projects/EI_Dynamics/Analysis/parsed_data"
sampleRate = 20000.0
OnsetDelay = 0.009  # Delay from the light stimulus to onset of syn current
# startT = 0.2 + OnsetDelay
# endT = 0.5 + OnsetDelay
runtime = 1.0

def tauFit( kernel, baseline):
    # y = kernel[ int( round(0.05*sampleRate )): ]
    y = kernel[ int( round(0.25*len(kernel) )): ]
    pk = y[0]
    x = np.linspace( 0, len(y)/sampleRate, len(y), endpoint = False )
    ret, cov = sci.curve_fit(lambda t,a,tau: a*np.exp(-t/tau), x, y, p0=(pk-baseline,0.02) )
    #print( "len = {}, a = {:.4f}, tau = {:4f}, range = {:4f}, bl = {:.4f}".format( len( y ), ret[0], ret[1], pk - baseline, baseline ) )
    # plt.plot( x + startT + 0.05, ret[0] * np.exp( -x/ret[1] ), ":" )
    return ret

def calcKernel( dat ):

    startIdx = int( round(startT*sampleRate ) )
    endIdx = int( round(0.8*endT*sampleRate ) )
    # endIdx  = int(startIdx + 0.65*(int( endT*sampleRate ) - startIdx))
    #baseline  = np.mean( dat[startIdx - int( 0.005*sampleRate ):startIdx] )
    baseline  = np.mean( dat.iloc[startIdx - int( 0.005*sampleRate ):startIdx] )
    
    rawKernel = np.array( dat.iloc[startIdx:endIdx] )
    # print( "SHAPE = ", dat.shape, rawKernel.shape, startT, endT, startIdx, endIdx )
    #t = np.arange( 0.0, endT - startT - 1e-6, 1.0/sampleRate )
    try:
        kmax = max( rawKernel )
        kmin = min( rawKernel )
    except:
        print( "SHAPE = ", dat.shape, rawKernel.shape, startT, endT, startIdx, endIdx )
        raise FloatingPointError( "calcKernel: kmax or kmin is a nan" )
    if math.isnan( kmax ) or math.isnan( kmin ):
        #print( "idx = ", startIdx, endIdx )
        #print( dat.iloc[startIdx:endIdx] )
        raise FloatingPointError( "calcKernel: kmax or kmin is a nan" )


    if abs( kmax ) > abs( kmin ): # Inhib trace has a positive peak.
        return kmax, rawKernel, baseline, tauFit(rawKernel, baseline)
    else:
        return kmin, rawKernel, baseline, tauFit(rawKernel, baseline)

def findStpScale( kernel, kpk, ret, si, stimWidth, tau, ax ):
    '''
    This function finds the scale factor for short-term plasticity (STP) based on the kernel, peak value, and other parameters.
    It calculates the rise time and total rise for the given stimulus and returns the scale factor.
    '''
    # ret[0,1] = valley_t, y; ret[1,2] = pk_t, y
    if ret[0] < endT and si < (endT * sampleRate):
        return 1.0
    if kpk < 0 : # Exc
        kpkIdx = np.argmin( kernel[:-stimWidth] )
    elif kpk > 0: # Inh
        kpkIdx = np.argmax( kernel[:-stimWidth] )

    pkIdx = int( round ( (ret[2]) * sampleRate )) - si
    riseIdx = int( round ( (ret[2] - ret[0]) * sampleRate ))
    riseDelta1 = kernel[kpkIdx + stimWidth - riseIdx ] - kernel[kpkIdx + stimWidth]
    riseDelta = ret[1] - ret[1]*np.exp( -(ret[2] - ret[0]) / tau[1] )
    if ax:
        label = "Min to Max" if (si < 11000 and kpk > 0) else None
        ax.plot( [ret[2], ret[2]], [-riseDelta + ret[1], ret[3]], "ro-", label = label )
    if ret[0] < endT + 0.01:   # First pulse after ref.
        riseTotal = ret[3] - ret[1]
    else:
        riseTotal = riseDelta + ret[3] - ret[1]
    return riseTotal / kpk

def findPkVal( dat, freq, startIdx, isExc ):
    # Returns times and absolute peak and val preceding peak of specified stim.
    stimWidth = int( round( 0.7 * sampleRate / freq ) )
    # print(dat.shape, startIdx, stimWidth)
    d2 = np.array(dat.iloc[startIdx:startIdx + stimWidth])
    if isExc: # Sign flipped in current. imin is peak.
        imin = np.argmin(d2)
        # Look for valley preceding this.
        d3 = np.array(dat.iloc[startIdx + imin-stimWidth:imin + startIdx])
        #d3 = np.array(dat.iloc[startIdx:startIdx+stimWidth])
        #print( "Exc startIdx={}, imin = {}, stimWidth = {}, tot={}".format( startIdx, imin, stimWidth, startIdx + imin-stimWidth))
        imax = np.argmax(d3)
        if imax + imin - stimWidth > imin:
            print( "WARNING: reversal" )
        return [(imax + startIdx + imin - stimWidth)/sampleRate, d3[imax], (startIdx+imin)/sampleRate, d2[imin]]
    
    else:   # This is the inhibitory input, which has positive currents.
        imax = np.argmax(d2)
        d3 = np.array(dat.iloc[startIdx + imax-stimWidth:imax + startIdx])
        #print( "Inh startIdx={}, imax = {}, stimWidth = {}, tot={}".format( startIdx, imax, stimWidth, startIdx + imax-stimWidth))
        try:
            imin = np.argmin(d3)
        except:
            print(freq, d3.shape, startIdx, imax, stimWidth)
            # abort the program
            raise FloatingPointError( "findPkVal: imin is a nan" )

        if imax < imin + imax - stimWidth:
            print( "WARNING: reversal" )
        return [(imin + startIdx+imax - stimWidth )/sampleRate, d3[imin], (startIdx + imax)/sampleRate, d2[imax]]

def deconv(dat, freq, start_time, end_time , ax, noprobepulse=False):
    """
    Deconvolves the given data using a specified frequency and returns the scale list, synthetic plot, and normalized peak values.

    Parameters:
    - dat (numpy.ndarray): The input data array.
    - freq (float): The frequency used for deconvolution.
    - ax (matplotlib.axes.Axes): The matplotlib axes object to plot the synthetic plot.

    Returns:
    - scaleList (numpy.ndarray): The array of scale values.
    - synthPlot (matplotlib.lines.Line2D): The synthetic plot.
    - npv (numpy.ndarray): The normalized peak and valley values.

    """
    # set startT and endT as global variables
    global startT, endT
    startT = start_time
    endT   = end_time

    startIdx  = int(round(endT * sampleRate))
    stimWidth = int(round(sampleRate / freq))
    # Don't reuse stimWidth for stimIdx because of cumulative error.
    stimIdx = [int(startT * sampleRate)] + [int(round(sampleRate * (endT + i / freq))) for i in range(8)]
    
    if noprobepulse:
        stimIdx = [int(startT * sampleRate)] + [int(round(sampleRate * (endT + i / freq))) for i in range(9)]
        stimIdx = stimIdx[1:]
        startT  = stimIdx[0]/2e4
        endT    = stimIdx[1]/2e4
    # use calcKernel to get the kernel, peak, and baseline
    kpk, kernel, baseline, tau = calcKernel(dat) 
    kpkidx = np.argmax(kernel) if kpk > 0 else np.argmin(kernel)
    # ax.plot(np.linspace(startT,endT,len(kernel)),kernel, "m.-")
    scaleList = []
    absPk = [kpk]
    absVal = [baseline]
    pv = []
    correctedStimIdx = []
    print(stimIdx, freq, kpkidx, kpk, startT, endT, stimWidth, kernel.shape, tau)
    if kpkidx == 0 or math.isnan(kpk) or math.isinf(kpk):
        raise FloatingPointError( "WARNING: kpk is zero, nan or inf" )
        
    for si in stimIdx:
        ret = findPkVal(dat, freq, si + kpkidx // 2, (kpk < 0))
        pv.append(ret)
        scale = findStpScale(kernel, kpk, ret, si, stimWidth, tau, None)
        scaleList.append(scale)
        if kpk > 0:
            label = "Inh"
        else:
            label = "Exc"

    npv = np.array(pv).transpose()
    synthPlot = plotFromKernel(scaleList, stimIdx, kernel, freq, npv, label, None)

    return np.array(scaleList), synthPlot, npv, stimIdx


def plotFromKernel( scaleList, stimIdx, kernel, freq, npv, label, ax ):
    ret = np.zeros( int( round( sampleRate * 1.5 ) ) ) 
    ret = ret.astype(np.float64)  # Convert ret array to float64 data type
    ret[int(round(sampleRate* endT )):] += npv[1][1]
    # print(scaleList)
    for ii in range( len( scaleList ) ):
        ss = scaleList[ii]
        idx = stimIdx[ii]
        if idx > 0 :
            ks = kernel * ss
            if label == "Inh":
                offset = npv[3,ii] - max( ks + ret[idx:len(kernel)+idx] )
            else: 
                offset = npv[3,ii] - min( ks + ret[idx:len(kernel)+idx] )

            # print(idx, type(kernel), type(ss), type(ret), type(ks), type(offset))
            # print(idx, kernel.shape, ss, ret.shape, ks.shape, offset)
            ret[idx:len(kernel)+idx] = ret[idx:len(kernel)+idx]  + ks + offset

    t = np.arange( 0.0, 1.0 - 1e-6, 1.0/sampleRate )
    if ax:
        el1 = None if label == "Inh" else "Troughs"
        el2 = None if label == "Inh" else "Peaks"
        ax.plot( npv[0], npv[1], "c*-", label = el1 )
        ax.plot( npv[2], npv[3], "y.-", label = el2 )
    return ret[:len(t)]

def plotA( ax, imean, emean ):
    t = np.arange( 0.0, runtime - 1e-6, 1.0/sampleRate )
    ax.plot( t, imean, "g-", label="IPSC" )
    ax.plot( t, emean, "b-", label="EPSC" )

    ax.set_xlabel( "Time (s)" )
    ax.set_ylabel( "Synaptic current (pA)" )
    # ax.set_ylim( -300, 800 )
    ax.legend( loc="upper left", frameon = False )
    ax.text( -0.12, 1.05, "A", fontsize = 22, weight = "bold", transform=ax.    transAxes )

def plotB( ax, ideconv, edeconv ):
    ax.plot( range( len( ideconv ) ), ideconv/ideconv[0], label="Inh" )
    ax.plot( range( len( edeconv ) ), edeconv/edeconv[0], label="Exc" )
    ax.set_xlabel( "Pulse # in burst" )
    ax.set_ylabel( "Min-to-Max ratio" )
    ax.legend( loc="upper right", frameon = False )
    # ax.set_ylim( 0, 1.4)
    ax.text( -0.24, 1.05, "B", fontsize = 22, weight = "bold", transform=ax.    transAxes )

def plotC( ax, ipv, epv ):
    y = ipv[1] - ipv[3]
    ax.plot( range( len( y ) ), y/y[0], label="Inh" )
    y = epv[1] - epv[3]
    ax.plot( range( len( y ) ), y/y[0], label="Exc" )
    ax.set_xlabel( "Pulse # in burst" )
    ax.set_ylabel( "Peak-to-Trough ratio" )
    # ax.set_ylim( 0, 1.4)
    ax.legend( loc="upper right", frameon = False )
    ax.text( -0.24, 1.05, "C", fontsize = 22, weight = "bold", transform=ax.    transAxes )

def plotDE( ax, sqDat, freq, patternList, panelName ):
    numSamples = int( round( sampleRate * runtime ) )
    elabel = "Exc"
    ilabel = "Inh"
    for pp in patternList:
        dataStartColumn = len( sqDat.columns ) - 80000 # Was 29, new is 39.
        inh = sqDat.loc[ (sqDat['stimFreq'] == freq) & (sqDat['clampPotential'] > -0.05 ) & (sqDat['patternList'] == pp ) ]
        exc = sqDat.loc[ (sqDat['stimFreq'] == freq) & (sqDat['clampPotential'] < -0.05 ) & (sqDat['patternList'] == pp ) ]
        yexc = exc.iloc[:,dataStartColumn: numSamples+dataStartColumn]
        yinh = inh.iloc[:,dataStartColumn: numSamples+dataStartColumn]
        emean = yexc.mean()
        imean = yinh.mean()
        try:
            ii, iSynthPlot, ipv = deconv( imean, freq, None )
            ee, eSynthPlot, epv = deconv( emean, freq, None )
        except FloatingPointError as error:
            print( "innerAnalysis: freq = {}, pattern = {}, cellNum = {}".format( freq, pattern, cellNum ) )
            raise
        ax.plot( range( len( ii ) ), ii/ii[0], "g*-", label=ilabel )
        ax.plot( range( len( ee ) ), ee/ee[0], "b*-", label=elabel )
        ilabel = None
        elabel = None
    ax.set_xlabel( "Pulse # in burst" )
    ax.set_ylabel( "Min-to-Max ratio" )
    ax.legend( loc="upper right", frameon = False )
    # ax.set_ylim( 0, 1.8)
    ax.text( -0.24, 1.05, panelName, fontsize = 22, weight = "bold", transform=ax.    transAxes )

def plotFG( fig, gs, sqDat, freqList, pattern, axF, axG ):
    # axF = fig.add_subplot( gs[3,0] )
    # axG = fig.add_subplot( gs[3,1] )
    numSamples = int( round( sampleRate * runtime ) )
    elabel = "Exc"
    ilabel = "Inh"
    for ff in sorted( freqList ):
        label = "{} Hz".format( ff )
        dataStartColumn = len( sqDat.columns ) - 80000 # Was 29, new is 39.
        inh = sqDat.loc[ (sqDat['stimFreq'] == ff) & (sqDat['clampPotential'] > -0.05 ) & (sqDat['patternList'] == pattern ) ]
        exc = sqDat.loc[ (sqDat['stimFreq'] == ff) & (sqDat['clampPotential'] < -0.05 ) & (sqDat['patternList'] == pattern ) ]
        yexc = exc.iloc[:,dataStartColumn: numSamples+dataStartColumn]
        yinh = inh.iloc[:,dataStartColumn: numSamples+dataStartColumn]
        emean = yexc.mean()
        imean = yinh.mean()
        try:
            ii, iSynthPlot, ipv = deconv( imean, ff, None )
            ee, eSynthPlot, epv = deconv( emean, ff, None )
        except FloatingPointError as error:
            print( "innerAnalysis: freq = {}, pattern = {}, cellNum = {}".format( freq, pattern, cellNum ) )
            raise
        #axF.plot( range( len( ii ) ), ii/ii[0], "g*-", label=label )
        #axG.plot( range( len( ee ) ), ee/ee[0], "b*-", label=label )
        axF.plot( range( len( ii ) ), ii/ii[0], "*-", label=label )
        axG.plot( range( len( ee ) ), ee/ee[0], "*-", label=label )
        ilabel = None
        elabel = None
    axF.set_xlabel( "Pulse # in burst" )
    axF.set_ylabel( "Min-to-Max ratio" )
    axF.legend( loc="upper right", frameon = False )
    axF.set_ylim( 0, 2.0)
    axF.text( -0.24, 1.05, "F", fontsize = 22, weight = "bold", 
            transform=axF.    transAxes )
    axG.set_xlabel( "Pulse # in burst" )
    axG.set_ylabel( "Min-to-Max ratio" )
    axG.legend( loc="upper right", frameon = False )
    axG.set_ylim( 0, 2.0)
    axG.text( -0.24, 1.05, "G", fontsize = 22, weight = "bold", 
            transform=axG.    transAxes )

def innerAnalysis(fig, gs, sqDat, freq, pattern, cellNum, axA, axB, axC):
    """
    Perform inner analysis on the given data.

    Args:
        fig (matplotlib.figure.Figure): The figure object.
        gs (matplotlib.gridspec.GridSpec): The gridspec object.
        sqDat (pandas.DataFrame): The data frame containing the data.
        freq (float): The stimulus frequency.
        pattern (str): The pattern list.
        cellNum (int): The cell number.
        axA (matplotlib.axes.Axes): The first subplot axes object.
        axB (matplotlib.axes.Axes): The second subplot axes object.
        axC (matplotlib.axes.Axes): The third subplot axes object.

    Returns:
        tuple: A tuple containing the results of the inner analysis.

    Raises:
        FloatingPointError: If a floating point error occurs during the analysis.
    """
    numSamples = int(round(sampleRate * runtime))
    dataStartColumn = len(sqDat.columns) - 80000  # Was 29, new is 39.
    
    # Select the data for inh and exc
    inhData = sqDat.loc[(sqDat['stimFreq'] == freq) & (sqDat['clampPotential'] > -0.05) & (sqDat['patternList'] == pattern)]
    excData = sqDat.loc[(sqDat['stimFreq'] == freq) & (sqDat['clampPotential'] < -0.05) & (sqDat['patternList'] == pattern)]
    
    # Extract the relevant columns for inh and exc
    inhColumns = inhData.iloc[:, dataStartColumn: numSamples + dataStartColumn]
    excColumns = excData.iloc[:, dataStartColumn: numSamples + dataStartColumn]
    
    # Calculate the mean of inh and exc
    inhMean = inhColumns.mean()
    excMean = excColumns.mean()
    #deconv(dat, freq, start_time, end_time , ax)
    # print(inhData.iloc[0,20:40])
    start_time, end_time = inhData
    try:
        inhDeconv, inhSynthPlot, inhPv = deconv(inhMean, freq, axA)
        excDeconv, excSynthPlot, excPv = deconv(excMean, freq, axA)
    except FloatingPointError as error:
        print("innerAnalysis: freq = {}, pattern = {}, cellNum = {}".format(freq, pattern, cellNum))
        raise

    assert(len(inhSynthPlot) == len(inhMean))
    assert(len(excSynthPlot) == len(excMean))

    plotA(axA, inhMean, excMean)  # Complex pattern showing STP estimation
    plotB(axB, inhDeconv, excDeconv)
    plotC(axC, inhPv, excPv)

    finalInh = np.append(inhDeconv, inhPv)
    finalExc = np.append(excDeconv, excPv)
    return finalInh, finalExc

def main(alldat, fig, axs, cellNum=7492, numSq=5):
    # parser = argparse.ArgumentParser( description = "This program analyzes Aditya's voltage clamp data from a pandas file" )
    # parser.add_argument( "-o", "--output", type = str, help = "Optional: output pandas hdf file for model params, default = STP_pks_and_refs.h5", default = "STP_pks_and_refs.h5")
    # args = parser.parse_args()

    frame = []
    # alldat = pandas.read_hdf( "{}/all_cells_FreqSweep_VC_long.h5".format( datadir ) )
    alldat['patternList'] = alldat['patternList'].astype( int )
    #print( alldat.columns[:100] )
    # cellNum = 7492
    # numSq = 5
    dat = alldat.loc[ alldat["cellID"] == cellNum ]
    sqDat = dat.loc[ dat['numSq'] == numSq ]
    freqList = sqDat['stimFreq'].unique()
    patternList = sqDat['patternList'].unique()

    ########## Set up graphics #########
    if not axs:
        print('making new fig')
        fig, axs = plt.subplots(7,1, figsize = (10,15) )
    [axA, axB, axC, axD, axE, axF, axG] = axs

    ########## Panels A, B, C  #########
    freq = freqList[0]
    pattern = int( patternList[1] )
    finh, fexc = innerAnalysis( fig, "", sqDat, freq, pattern, cellNum, axA, axB, axC )

    ########## Panel D, E, F, G  #########
    # plotDE( fig.add_subplot( gs[2,0] ), sqDat, freq, patternList, "D" )
    plotDE( axD, sqDat, freq, patternList, "D" )
    sqDat15 = dat.loc[ dat['numSq'] == 15 ]
    patternList15 = sqDat15['patternList'].unique()
    # plotDE( fig.add_subplot( gs[2,1] ), sqDat15, freq, patternList15, "E" )
    plotDE( axE, sqDat15, freq, patternList15, "E" )

    plotFG( fig, gs, sqDat, freqList, pattern, axF, axG )

    axs = [axA, axB, axC, axD, axE, axF, axG]
    ########## Wrap up graphics #########
    # plt.tight_layout()
    # plt.show()
    return fig, axs, finh, fexc

if __name__ == "__main__":
    main()
