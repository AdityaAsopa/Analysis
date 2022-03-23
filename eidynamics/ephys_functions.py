from codecs import IncrementalDecoder
import numpy as np
import pandas as pd
from scipy          import signal
from scipy.optimize import curve_fit

from eidynamics.utils import epoch_to_datapoints as e2dp
from eidynamics.utils import charging_membrane
from eidynamics.utils import PSP_start_time

def tau_calc(recordingData, IRBaselineEpoch, IRchargingPeriod, IRsteadystatePeriod, clamp='CC', Fs=2e4):
    ''' recordingData is the data dictionary with sweeps numbers as keys.
        Provide steadystateWindow values in seconds.
    '''
    _info = "In current clamp(CC), after bridge balance, the charging/discharging depends only on Rm and Cm.\n\
            In voltage clamp(VC), both pipette capacitance and cell capacitance contribute to charging and discharging.\n\
            Therefore, in VC, after pipette capacitance (Cp), whole cell (Cm) and series resistance compensation (Rs),\n\
            the tau and Cm values are not reflected in cell responses to voltage pulses. Instead, they should be noted\n\
            down from multiclamp commander or clampex."        
    
    tau_trend = []

    if clamp=='VC':
        print(_info)
        Cm = np.nan
        tau_trend = np.zeros(tau_trend.shape)
        return np.nan*tau_trend, 0, Cm
    
    for s in recordingData.values():
        cmdTrace    = s['Cmd']
        resTrace    = s[0]
        time        = s['Time']

        chargeTime  = time[e2dp(IRchargingPeriod,Fs)] - IRchargingPeriod[0]
        chargeRes   = resTrace[e2dp(IRchargingPeriod,Fs)]
        Icmd        = cmdTrace[int(Fs*IRsteadystatePeriod[0])]
        
        # check the charging_membrane function help for info on bounds and p0
        try:
            popt,_      = curve_fit( charging_membrane, chargeTime, chargeRes, bounds=([-10,-10,0],[10,10,0.05]), p0=([0.01,-2.0,0.02]) )
            tau_trend.append(popt[2])
        except:
            tau_trend.append(0)
        

    # Tau change flag
    # Tau change screening criterion is 20% change in Tau during the recording OR tau going above 0.5s
    tau_flag      = 0
    if (np.percentile(tau_trend,95) / np.median(tau_trend) > 0.5) | (np.max(np.percentile(tau_trend,95)) > 0.5):
        tau_flag  = 1
    
    # median Cm and Rm values
    Rm = 1000*popt[1]/Icmd # MegaOhms
    Cm = 1e6*np.median(tau_trend)/Rm #picoFarads
    tau_trend = np.array(tau_trend)

    return tau_trend, tau_flag, Cm


def IR_calc(recordingData,clamp,IRBaselineEpoch,IRsteadystatePeriod,Fs=2e4):
    ''' recordingData is the data dictionary with sweeps numbers as keys.
        Provide steadystateWindow values in seconds'''

    IRtrend = np.zeros(len(recordingData))
    for i,k in enumerate(recordingData):
        s = recordingData[k]
        cmdTrace = s['Cmd']
        ss1_cmd = np.mean(cmdTrace[e2dp(IRBaselineEpoch,Fs)])
        ss2_cmd = np.mean(cmdTrace[e2dp(IRsteadystatePeriod,Fs)])
        delCmd = ss2_cmd - ss1_cmd

        recSig = s[0]
        ss1_rec = np.mean(recSig[e2dp(IRBaselineEpoch,Fs)])
        ss2_rec = np.mean(recSig[e2dp(IRsteadystatePeriod,Fs)])
        delRes = ss2_rec - ss1_rec

        if clamp == 'VC':
            ir = 1000 * delCmd / delRes  # mult with 1000 to convert to MOhms
        else:
            ir = 1000 * delRes / delCmd  # mult with 1000 to convert to MOhms

        IRtrend[i] = ir
        
        #putting a hard coded range of acceptable IR from percentile calculation done on data obtained so far,
        # 25%ile = 86MOhm, median = 137MOhm, 75%ile = 182 MOhm
        # 10%ile = 10MOhm, 90%ile = 282MOhm
        # TAG TODO remove hard coded variable        
        
        # IR change flag
        # IR change screening criterion is 20% change in IR during the recording
        IRflag = 0
        if (np.max(IRtrend) - np.min(IRtrend)) / np.mean(IRtrend) > 0.2:
            IRflag = 1
        # OR
        if np.max(IRtrend)>300 or np.min(IRtrend)<= 0:  #putting a hard coded range of acceptable IR
            IRflag = 1
    IRtrend = np.array(IRtrend)
    sweepwise_irtrend = np.logical_or(IRtrend<15, IRtrend>400) 
    return IRtrend, sweepwise_irtrend, IRflag


def pulseResponseCalc(recordingData,eP):
    pulsePeriods    = []
    PeakResponses   = []
    AUCResponses    = []
    df_peaks        = pd.DataFrame()

    APflag          = bool(0)

    for sweepID,sweep in recordingData.items():
        ch0_cell        = sweep[0]
        ch1_frameTTL    = sweep[1]
        ch2_photodiode  = sweep[2]

        stimfreq        = eP.stimFreq  # pulse frequency
        Fs              = eP.Fs
        IPI_samples     = int(Fs * (1 / stimfreq))          # inter-pulse interval in datapoints
        firstPulseStart = int(Fs * eP.opticalStimEpoch[0])
        
        res             = []
        t1              = firstPulseStart
        for i in range(eP.numPulses):
            t2 = t1 + IPI_samples
            res.append(ch0_cell[t1:t2])
            t1 = t2
        res             = np.array(res)
        
        peakTimes       = []
        df_peaks.loc[sweepID + 1, "firstPulseDelay"],_ = PSP_start_time(ch0_cell,eP.clamp,eP.EorI,stimStartTime=eP.opticalStimEpoch[0],Fs=Fs)

        if eP.EorI == 'I' or eP.clamp == 'CC':
            maxRes = np.max(res, axis=1)
            aucRes = np.trapz(res,axis=1)
            PeakResponses.append(np.max(maxRes))
            
            df_peaks.loc[sweepID + 1, [1,2,3,4,5,6,7,8]] = maxRes
            
            for resSlice in res:
                maxVal = np.max(resSlice)
                pr = np.where(resSlice == maxVal)[0]  # signal.find_peaks(resSlice,height=maxVal)
                peakTimes.append(pr[0] / 20)
            df_peaks.loc[sweepID + 1, [9,10,11,12,13,14,15,16]] = peakTimes[:]
            
            df_peaks.loc[sweepID + 1, "AP"] = bool(False)
            if eP.clamp == 'CC':
                df_peaks.loc[sweepID + 1, "AP"] = bool(np.max(maxRes) > 80) # 80 mV take as a threshold above baseline to count a response as a spike
                APflag = bool(df_peaks.loc[sweepID + 1, "AP"] == True)
            
            df_peaks.loc[sweepID + 1, [17,18,19,20,21,22,23,24]] = aucRes

        elif eP.EorI == 'E' and eP.clamp == 'VC':
            minRes = np.min(res, axis=1)
            aucRes = np.trapz(res,axis=1)
            PeakResponses.append(np.min(minRes))

            df_peaks.loc[sweepID + 1, [1,2,3,4,5,6,7,8]] = minRes
            
            for resSlice in res:
                minVal = np.min(resSlice)
                pr = np.where(resSlice == minVal)[0]  # pr,_ = signal.find_peaks(-1*resSlice,height=np.max(-1*resSlice))
                peakTimes.append(pr[0] / 20)
            
            df_peaks.loc[sweepID + 1, [9,10,11,12,13,14,15,16]] = peakTimes[:]
            df_peaks.loc[sweepID + 1, "AP"] = bool(np.max(-1 * minRes) > 80)
            APflag = bool(df_peaks.loc[sweepID + 1, "AP"] == True)

            df_peaks.loc[sweepID + 1, [17,18,19,20,21,22,23,24]] = aucRes

    df_peaks.astype({"AP":'bool'})    
    df_peaks["PeakResponse"]                     = PeakResponses
    df_peaks["datafile"]                         = eP.datafile

    return df_peaks, APflag

# FIXME: remove hardcoded variables, fields, and values
