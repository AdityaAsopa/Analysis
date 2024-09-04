## Modified from deconv11.py
##      Author: U S Bhalla
##      Reference: https://labnotes.ncbs.res.in/bhalla/aditya-vc-data-production-runs-7-cells-using-deconv11py

import pandas
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math
import time
import scipy.optimize as sci

Fs = 2e4

def alphaFunc( t, tp ):
    if ( t < 0 ):
        return 0.0
    return (t/tp) * np.exp(1-t/tp)

def dualAlphaFunc( t, t1, t2 ):
    if t < 0:
        return 0.0
    if abs( t1 - t2 ) < 2e-5 or t1 < 5e-4 or t2 < 5e-4:
        return alphaFunc( t, max(t1, t2) )
    return (1.0/(t1-t2)) * (np.exp(-t/t1) - np.exp(-t/t2))


class DualAlpha():
    def __init__( self, kernel, power ):
        self.kernel = kernel
        self.power = power

    def scaledAlphaVec( self, t1, t2, delay, length ):
        dt = 1.0/Fs
        alpha = np.array([dualAlphaFunc( i*dt-delay, t1, t2 ) for i in range(length)] )
        if abs( max( alpha ) ) > 1e-9:
            alpha = alpha / max( alpha )
        alpha = np.power( alpha, self.power )
        return alpha

    def score( self, params ):
        lk = len( self.kernel )
        dt = 1.0/Fs
        alpha = self.scaledAlphaVec( params[0], params[1], params[2], lk )
        delta = alpha - self.kernel
        return np.dot( delta, delta )

    def fit( self ):
        initGuess = [0.005, 0.04, 0.01 ]
        t0 = time.time()
        result = sci.minimize( self.score, initGuess, method = "BFGS" )
        # print( "mintime={:.3f}, t1 = {:.3f}, t2={:.3f}, del={:.3f}, score = {:.2f}, initScore = {:.2f}".format( time.time() - t0, result.x[0], result.x[1], result.x[2], self.score( result.x ), self.score( initGuess ) ) )
        return result.x

    def plotFit( self, freq ):
        plt.figure()
        [t1,t2,delay] = self.fit()
        t = np.arange( 0.0, 1.0- 1e-6, 1.0/Fs )
        alpha = self.scaledAlphaVec( t1, t2, len(t) )
        plt.plot( t[:len(self.kernel)], self.kernel, label="data" )
        plt.plot( t, alpha, label = "fit" )
        plt.xlabel( "Time (s)" )
        plt.ylabel( "frac open" )
        plt.title( "dual alpha func fit for " + str(freq) + " Hz" )
        plt.title( "dual alpha func freq= {} Hz, delay = {} sec, power = {}".format(freq, delay, self.power) )
        plt.legend()


class DualChanDualAlpha():
    def __init__( self, kernel, kpk ):
        self.alpha1 = DualAlpha( kernel, 1.0 )
        self.alpha2 = DualAlpha( kernel, 1.0 )
        self.kpk = kpk

    def score( self, params ):
        # Params have 6 entries: tp1.1, tp1.2, tp2.1, tp2.2, delay, ratio
        ratio = params[5]
        klen = len( self.alpha1.kernel )
        g1 = self.alpha1.scaledAlphaVec( params[0], params[1], params[4], klen )
        g2 = self.alpha2.scaledAlphaVec( params[2], params[3], params[4], klen )
        delta = g1 * ratio + g2 * (1.0-ratio) - self.alpha1.kernel
        return np.dot( delta, delta )

    def fit( self ):
        # Params have 6 entries: tp1.1, tp1.2, tp2.1, tp2.2, delay, ratio
        if self.kpk > 0:    # inhib
            initGuess = [0.005, 0.05, 0.010, 0.100, 0.01, 0.8 ]
        else:
            initGuess = [0.002, 0.02, 0.005, 0.05, 0.01, 0.8 ]
        t0 = time.time()
        #result = sci.minimize( self.score, initGuess, method = "COBYLA", tol = 0.00001 )
        result = sci.minimize( self.score, initGuess, method = "L-BFGS-B", bounds = [ (0.001, 0.04), (0.005, 0.3), (0.001, 0.04), (0.005, 0.3), (0.002, 0.02), (0.0, 1.0)] )
        ans = result.x
        #print( "kpk = {:.1f}, runtime={:.3f}, t1=({:.3f}, {:.3f}), t2=({:.3f},{:.3f}), delay={:.3f}, ratio={:.3f}, score = {:.2f}, initScore = {:.2f}".format(self.kpk, time.time() - t0, ans[0], ans[1], ans[2], ans[3], ans[4], ans[5], self.score( ans ), self.score( initGuess ) ) )
        # print( "{:4.1f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.2f}{:8.2f}".format(self.kpk, time.time() - t0, ans[0], ans[1], ans[2], ans[3], ans[4], ans[5], self.score( ans ), self.score( initGuess ) ) )
        return ans

    def plotFit( self, freq ):
        #plt.figure()
        ret = self.fit()
        delay = ret[4]
        ratio = ret[5]
        t = np.arange( 0.0, 1.0- 1e-6, 1.0/Fs )
        klen = len(t)
        g1 = self.alpha1.scaledAlphaVec( ret[0], ret[1], delay, klen )
        g2 = self.alpha2.scaledAlphaVec( ret[2], ret[3], delay, klen )
        sumAlpha = g1 * ratio + g2 * (1.0-ratio)
        '''
        plt.plot( t[:len(self.alpha1.kernel)], self.alpha1.kernel, label="data" )
        plt.plot( t, sumAlpha, label = "fit" )
        plt.xlabel( "Time (s)" )
        plt.ylabel( "frac open" )
        #plt.title( "dual chan alpha func fit for " + str(freq) + " Hz" )
        label = "Inh"
        if self.kpk < 0:
            label = "Exc"
        plt.title( "{} dual chan alpha func freq= {} Hz, delay = {} sec".format(label, freq, self.alpha1.delay) )
        plt.legend()
        '''
        return sumAlpha


def calcKernel( dat ):
    startT = 0.2
    endT = 0.5

    startIdx = int( round(startT*Fs ) )
    endIdx = int( round(endT*Fs ) )
    
    rawKernel = np.array( dat[startIdx:endIdx] )
    #t = np.arange( 0.0, endT - startT - 1e-6, 1.0/Fs )
    #print( "SHAPE = ", dat.shape, rawKernel.shape, t.shape, startIdx, endIdx )
    maxk = max( rawKernel )
    mink = min( rawKernel )

    # Can't do this because large offset may shift sign of the mean.
    #if np.mean( rawKernel ) > 0.0: 

    if abs( maxk ) > abs( mink ):
        return maxk, rawKernel
    else:
        return mink, rawKernel

    #plt.plot( t, rawKernel )


def findResidual( kernel, kpk, residual, si, stimWidth ):
    #pkScale = [0.94, 0.96,0.98, 0.99, 1.0, 1.01, 1.02, 1.04, 1.06]
    pkScale = [0.98, 0.99, 1.0, 1.01, 1.02]
    truncatedKernel = kernel[ : stimWidth ]

    if kpk > 0:
        stimPk = max( residual[ si : si+stimWidth] )
    else:
        stimPk = min( residual[ si : si+stimWidth] )

    scale = stimPk / kpk

    bestErr = 1.0e12
    bestScale = 1.0
    for pp in pkScale:
        ss = pp * scale
        rr = residual[si: si + stimWidth] - kernel[:stimWidth]*ss
        err = np.dot( rr, rr )
        if err < bestErr:
            bestErr = err
            bestScale = ss

    #print( bestErr, bestScale/scale )
    lk = len( kernel)
    residual[si:si+lk] -= kernel[ : lk ] * bestScale
    return bestScale


def deconv( dat, freq, trainStart=0.5, probepulsetime=0.2 ):
    startIdx = int( round( trainStart * Fs ) ) # Start at 0.5 sec, convert to datapoints by multiplying by Fs
    stimWidth = int( round( Fs/freq ) ) # what I call IPI
    # Don't reuse stimWidth for stimIdx because of cumulative error.
    stimIdx = [ int( round( i * Fs/freq ) ) for i in range( 8 ) ]

    kpk, kernel = calcKernel( dat )

    dualChan = DualChanDualAlpha( kernel/kpk, kpk )
    newkernel = dualChan.plotFit( freq ) * kpk
    #plt.show()

    #print("STIMIDX = ",  stimIdx, stimWidth, kpk, np.mean( newkernel ) )

    # Make a nice long array so that we don't have to worry about
    # end truncation of the kernel
    residual = np.append( dat[startIdx:], np.zeros( len(dat) ), 0 )
    scaleList = []
    correctedStimIdx = []
    #print( "SHAPE = ", residual.shape, k.shape )

    for si in stimIdx:
        scale = findResidual( newkernel, kpk, residual, si, stimWidth )
        scaleList.append( scale )
        correctedStimIdx.append( si )

    #print( "ScaleList = ", scaleList )
    dfit = deconvFit( scaleList, correctedStimIdx, newkernel, trainStart )

    kfit = np.zeros( len( dat ) )
    stimStartIdx = int( round( probepulsetime * Fs ) )
    kfit[stimStartIdx:] = newkernel[:-stimStartIdx]
    #plotFromKernel( scaleList, correctedStimIdx, newkernel, label )
    #t = np.arange( 0.0, len( residual ) - 1e-6, 1.0 ) / Fs
    #plt.plot( t, residual, label = label )

    # the residual that is returned should be same length as dat and should have the residual in the right place.
     
    return np.array(scaleList), residual, dfit, kfit, kernel


def deconvFit( scaleList, correctedStimIdx, kernel, trainStart ):
    ret = np.zeros( int( round( Fs ) ) )
    for ss, idx in zip( scaleList, correctedStimIdx ):
        if idx > 0:
            ret[idx:] += kernel[:-idx] * ss
        else:
            ret += kernel * ss

    #t = np.arange( 0.5, 1.0 - 1e-6, 1.0/Fs )
    #plt.plot( t, ret[0:len(t)], ":", label = label + "_est" )
    r2 = np.zeros( int( round( Fs ) ) )
    startIdx = int( round( trainStart * Fs ) )
    r2[startIdx:] = ret[:-startIdx]
    return r2


def main(time, sweep_data, freq, trainStart=0.5, show_plots=False, fig='', ax=''):
        
    deconvo, residual, dfit, kfit, kernel = deconv( sweep_data, freq, trainStart=trainStart )
    if fig == '':
        fig, ax = plt.subplots(1,3, figsize = (18, 4.8) )
    # print(deconvo, residual, dfit, kfit)
    if show_plots:
        fig, ax = plot_fits( freq, time, deconvo, residual, dfit, kfit , fig, ax, trainStart)
        # fig.show()

    return {"deconv": deconvo, "residual": residual, "dfit": dfit, "kfit": kfit, 'kernel': kernel}, fig, ax #


def plot_fits(freq, t, deconv, residual, dfit, kfit, fig, ax, trainStart):
    # fig, ax = plt.subplots(1,3, figsize = (18, 4.8) )

    # plt.subplot( 1, 3, 1 )
    ax[0].plot( t, dfit, ":", label="deconv" )
    ax[0].plot( t, kfit, ":", label="kernel" )

    ax[0].set_xlabel( "Time (s)" )
    ax[0].set_ylabel( "Synaptic current (pA)" )
    ax[0].legend()
    ax[0].set_title( "Currents" )
    # plt.title( "Currents: cell {}, Freq = {}, NumSq = {}".format( cellnum, freq, args.numSquares ) )

    # plt.subplot( 1, 3, 2 )
    lenres = len( residual ) // 2
    tr = np.arange( trainStart, 1.0, 1.0/Fs )
    ax[1].plot( tr, residual[:len(tr)], label = "Fit Residual" )
    print("length of residual", len(residual), len(tr), np.min(tr), np.max(tr))
    ax[1].set_xlabel( "Time (s)" )
    ax[1].set_ylabel( "Residual (pA)" )
    ax[1].legend()
    ax[1].set_title( "Fit Residual" )
    # plt.title( "cell {}, Freq = {}: Residual ".format( cellnum, freq ) )

    # plt.subplot( 1, 3, 3 )
    ax[2].plot( range( 1,len( deconv )+1 ), deconv/deconv[0], label="Deconv" )
    ax[2].set_xlabel( "Pulse # in burst" )
    ax[2].set_ylabel( "Synaptic response as ratio to first pulse." )
    ax[2].legend()
    ax[2].set_title( "Deconv or Relative PSC strength" )
    # plt.title( "cell {}, Freq = {}: Deconv ".format( cellnum, freq ) )
    #plt.show()
    #plt.pause( 0.001 )

    return fig, ax


def find_sweep_expected(trace, freq, fig, ax):
    
    time = np.linspace(0, len(trace)/20000, len(trace))
    fits, fig, ax = main(time, trace, freq, show_plots=True, fig=fig, ax=ax)

    return fits, fig, ax

if __name__ == "__main__":
    main()
