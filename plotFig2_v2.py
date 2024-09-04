# Original Author: U. S. Bhalla, NCBS, Bangalore
# https://github.com/BhallaLab/STP_EI_paper_figs
# Modified by: Aditya Asopa, NCBS, Bangalore
# Date: 26 Jan 2024

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
startT = 0.2 + OnsetDelay
endT = 0.5 + OnsetDelay
runtime = 1.0

def tauFit(kernel, baseline):
   """
   Fit an exponential curve to the kernel and return the fit parameters.

   Args:
       kernel (numpy.ndarray): The kernel data.
       baseline (float): The baseline value of the kernel.

   Returns:
       tuple: A tuple containing the fit parameters (a, tau).
   """
   y = kernel[int(round(0.05 * sampleRate)):]
   pk = y[0]
   x = np.linspace(0, len(y) / sampleRate, len(y), endpoint=False)
   ret, cov = sci.curve_fit(lambda t, a, tau: a * np.exp(-t / tau), x, y, p0=(pk - baseline, 0.02))
   plt.plot(x + startT + 0.05, ret[0] * np.exp(-x / ret[1]), ":")
   return ret

def calcKernel(dat):
   """
   Calculate the kernel from the input data.

   Args:
       dat (pandas.Series): The input data series.

   Returns:
       tuple: A tuple containing the peak value, kernel, baseline, and fit parameters (a, tau).
   """
   startIdx = int(round(startT * sampleRate))
   endIdx = int(round(endT * sampleRate))
   baseline = np.mean(dat.iloc[startIdx - int(0.005 * sampleRate):startIdx])

   rawKernel = np.array(dat.iloc[startIdx:endIdx])
   kmax = max(rawKernel)
   kmin = min(rawKernel)
   if math.isnan(kmax) or math.isnan(kmin):
       raise FloatingPointError("calcKernel: kmax or kmin is a nan")

   if abs(kmax) > abs(kmin):  # Inhib trace has a positive peak.
       return kmax, rawKernel, baseline, tauFit(rawKernel, baseline)
   else:
       return kmin, rawKernel, baseline, tauFit(rawKernel, baseline)

def findStpScale(kernel, kpk, ret, si, stimWidth, tau, ax):
   """
   Find the scale factor for short-term plasticity (STP) from the kernel and fit parameters.

   Args:
       kernel (numpy.ndarray): The kernel data.
       kpk (float): The peak value of the kernel.
       ret (tuple): The fit parameters (a, tau) from tauFit.
       si (int): The start index of the stimulus.
       stimWidth (int): The width of the stimulus in samples.
       tau (tuple): The fit parameters (a, tau) from tauFit.
       ax (matplotlib.axes.Axes): The axes object for plotting (optional).

   Returns:
       float: The scale factor for STP.
   """
   # ret[0,1] = valley_t, y; ret[1,2] = pk_t, y
   if ret[0] < endT and si < (endT * sampleRate):
       return 1.0
   if kpk < 0:  # Exc
       kpkIdx = np.argmin(kernel[:-stimWidth])
   else:
       kpkIdx = np.argmax(kernel[:-stimWidth])
   pkIdx = int(round((ret[2]) * sampleRate)) - si
   riseIdx = int(round((ret[2] - ret[0]) * sampleRate))
   riseDelta1 = kernel[kpkIdx + stimWidth - riseIdx] - kernel[kpkIdx + stimWidth]
   riseDelta = ret[1] - ret[1] * np.exp(-(ret[2] - ret[0]) / tau[1])
   if ax:
       label = "Min to Max" if (si < 11000 and kpk > 0) else None
       ax.plot([ret[2], ret[2]], [-riseDelta + ret[1], ret[3]], "ro-", label=label)
   if ret[0] < endT + 0.01:  # First pulse after ref.
       riseTotal = ret[3] - ret[1]
   else:
       riseTotal = riseDelta + ret[3] - ret[1]
   return riseTotal / kpk

def findPkVal(dat, freq, startIdx, isExc):
   """
   Find the peak and trough values for a given stimulus in the input data.

   Args:
       dat (pandas.Series): The input data series.
       freq (float): The stimulus frequency.
       startIdx (int): The start index of the stimulus.
       isExc (bool): True if the input is excitatory, False if inhibitory.

   Returns:
       tuple: A tuple containing the time, value of the preceding trough, time, and value of the peak.
   """
   stimWidth = int(round(0.7 * sampleRate / freq))
   d2 = np.array(dat.iloc[startIdx:startIdx + stimWidth])
   if isExc:  # Sign flipped in current. imin is peak.
       imin = np.argmin(d2)
       # Look for valley preceding this.
       d3 = np.array(dat.iloc[startIdx + imin - stimWidth:imin + startIdx])
       imax = np.argmax(d3)
       if imax + imin - stimWidth > imin:
           print("WARNING: reversal")
       return [(imax + startIdx + imin - stimWidth) / sampleRate, d3[imax],
               (startIdx + imin) / sampleRate, d2[imin]]
   else:  # This is the inhibitory input, which has positive currents.
       imax = np.argmax(d2)
       d3 = np.array(dat.iloc[startIdx + imax - stimWidth:imax + startIdx])
       imin = np.argmin(d3)
       if imax < imin + imax - stimWidth:
           print("WARNING: reversal")
       return [(imin + startIdx + imax - stimWidth) / sampleRate, d3[imin],
               (startIdx + imax) / sampleRate, d2[imax]]

def deconv(dat, freq, ax):
   """
   Perform deconvolution on the input data and plot the results.

   Args:
       dat (pandas.Series): The input data series.
       freq (float): The stimulus frequency.
       ax (matplotlib.axes.Axes): The axes object for plotting.

   Returns:
       tuple: A tuple containing the deconvolved data, synthetic plot, and peak/trough values.
   """
   startIdx = int(round(endT * sampleRate))
   stimWidth = int(round(sampleRate / freq))
   stimIdx = [int(startT * sampleRate)] + [int(round(sampleRate * (endT + i / freq))) for i in range(8)]

   kpk, kernel, baseline, tau = calcKernel(dat)
   kpkidx = np.argmax(kernel) if kpk > 0 else np.argmin(kernel)

   scaleList = []
   absPk = [kpk]
   absVal = [baseline]
   pv = []
   correctedStimIdx = []

   for si in stimIdx:
       ret = findPkVal(dat, freq, si + kpkidx // 2, (kpk < 0))
       pv.append(ret)
       scale = findStpScale(kernel, kpk, ret, si, stimWidth, tau, ax)
       scaleList.append(scale)
       if kpk > 0:
           label = "Inh"
       else:
           label = "Exc"

   npv = np.array(pv).transpose()
   synthPlot = plotFromKernel(scaleList, stimIdx, kernel, freq, npv, label, ax)
   return np.array(scaleList), synthPlot, npv

def plotFromKernel(scaleList, stimIdx, kernel, freq, npv, label, ax):
   """
   Plot the synthetic trace from the kernel and scale factors.

   Args:
       scaleList (list): The list of scale factors.
       stimIdx (list): The list of stimulus indices.
       kernel (numpy.ndarray): The kernel data.
       freq (float): The stimulus frequency.
       npv (numpy.ndarray): The array of peak/trough values.
       label (str): The label for the plot ("Inh" or "Exc").
       ax (matplotlib.axes.Axes): The axes object for plotting.

   Returns:
       numpy.ndarray: The synthetic trace data.
   """
   ret = np.zeros(int(round(sampleRate * 1.5)))
   ret[int(round(sampleRate * endT)):] += npv[1][1]
   for ii in range(len(scaleList)):
       ss = scaleList[ii]
       idx = stimIdx[ii]
       if idx > 0:
           ks = kernel * ss
           if label == "Inh":
               offset = npv[3, ii] - max(ks + ret[idx:len(kernel) + idx])
           else:
               offset = npv[3, ii] - min(ks + ret[idx:len(kernel) + idx])
           ret[idx:len(kernel) + idx] += ks + offset

   t = np.arange(0.0, 1.0 - 1e-6, 1.0 / sampleRate)
   if ax:
       el1 = None if label == "Inh" else "Troughs"
       el2 = None if label == "Inh" else "Peaks"
       ax.plot(npv[0], npv[1], "c*-", label=el1)
       ax.plot(npv[2], npv[3], "y.-", label=el2)
   return ret[:len(t)]

def plotA(ax, imean, emean):
   """
   Plot the mean inhibitory and excitatory synaptic currents in panel A.

   Args:
       ax (matplotlib.axes.Axes): The axes object for plotting.
       imean (numpy.ndarray): The mean inhibitory synaptic current data.
       emean (numpy.ndarray): The mean excitatory synaptic current data.
   """
   t = np.arange(0.0, runtime - 1e-6, 1.0 / sampleRate)
   ax.plot(t, imean, "g-", label="IPSC")
   ax.plot(t, emean, "b-", label="EPSC")

   ax.set_xlabel("Time (s)")
   ax.set_ylabel("Synaptic current (pA)")
   ax.legend(loc="upper left", frameon=False)
   ax.text(-0.12, 1.05, "A", fontsize=22, weight="bold", transform=ax.transAxes)

def plotB(ax, ideconv, edeconv):
   """
   Plot the min-to-max ratio for inhibitory and excitatory deconvolved data in panel B.

   Args:
       ax (matplotlib.axes.Axes): The axes object for plotting.
       ideconv (numpy.ndarray): The inhibitory deconvolved data.
       edeconv (numpy.ndarray): The excitatory deconvolved data.
   """
   ax.plot(range(len(ideconv)), ideconv / ideconv[0], label="Inh")
   ax.plot(range(len(edeconv)), edeconv / edeconv[0], label="Exc")
   ax.set_xlabel("Pulse # in burst")
   ax.set_ylabel("Min-to-Max ratio")
   ax.legend(loc="upper right", frameon=False)
   ax.text(-0.24, 1.05, "B", fontsize=22, weight="bold", transform=ax.transAxes)

def plotC(ax, ipv, epv):
   """
   Plot the peak-to-trough ratio for inhibitory and excitatory deconvolved data in panel C.

   Args:
       ax (matplotlib.axes.Axes): The axes object for plotting.
       ipv (numpy.ndarray): The inhibitory peak/trough values.
       epv (numpy.ndarray): The excitatory peak/trough values.
   """
   y = ipv[1] - ipv[3]
   ax.plot(range(len(y)), y / y[0], label="Inh")
   y = epv[1] - epv[3]
   ax.plot(range(len(y)), y / y[0], label="Exc")
   ax.set_xlabel("Pulse # in burst")
   ax.set_ylabel("Peak-to-Trough ratio")
   ax.legend(loc="upper right", frameon=False)
   ax.text(-0.24, 1.05, "C", fontsize=22, weight="bold", transform=ax.transAxes)

def plotDE(ax, sqDat, freq, patternList, panelName):
   """
   Plot the min-to-max ratio for inhibitory and excitatory deconvolved data in panels D and E.

   Args:
       ax (matplotlib.axes.Axes): The axes object for plotting.
       sqDat (pandas.DataFrame): The input data frame.
       freq (float): The stimulus frequency.
       patternList (list): The list of pattern IDs.
       panelName (str): The panel name ("D" or "E").
   """
   numSamples = int(round(sampleRate * runtime))
   elabel = "Exc"
   ilabel = "Inh"
   for pp in patternList:
       dataStartColumn = len(sqDat.columns) - 80000  # Was 29, new is 39.
       inh = sqDat.loc[
           (sqDat['stimFreq'] == freq) & (sqDat['clampPotential'] > -0.05) & (sqDat['patternList'] == pp)]
       exc = sqDat.loc[
           (sqDat['stimFreq'] == freq) & (sqDat['clampPotential'] < -0.05) & (sqDat['patternList'] == pp)]
       yexc = exc.iloc[:, dataStartColumn: numSamples + dataStartColumn]
       yinh = inh.iloc[:, dataStartColumn: numSamples + dataStartColumn]
       emean = yexc.mean()
       imean = yinh.mean()
       try:
           ii, iSynthPlot, ipv = deconv(imean, freq, None)
           ee, eSynthPlot, epv = deconv(emean, freq, None)
       except FloatingPointError as error:
           print("innerAnalysis: freq = {}, pattern = {}, cellNum = {}".format(freq, pattern, cellNum))
           raise
       ax.plot(range(len(ii)), ii / ii[0], "g*-", label=ilabel)
       ax.plot(range(len(ee)), ee / ee[0], "b*-", label=elabel)
       ilabel = None
       elabel = None
   ax.set_xlabel("Pulse # in burst")
   ax.set_ylabel("Min-to-Max ratio")
   ax.legend(loc="upper right", frameon=False)
   ax.text(-0.24, 1.05, panelName, fontsize=22, weight="bold", transform=ax.transAxes)

def