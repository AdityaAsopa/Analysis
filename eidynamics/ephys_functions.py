from codecs import IncrementalDecoder
import numpy as np
import pandas as pd
from scipy          import signal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from eidynamics.utils import epoch_to_datapoints as e2dp
from eidynamics.utils import charging_membrane
from eidynamics.utils import PSP_start_time, get_pulse_times


def tau_calc(recordingData, IRBaselineEpoch, IRchargingPeriod, IRsteadystatePeriod, clamp='CC', Fs=2e4):
    ''' recordingData is the data dictionary with sweeps numbers as keys.
        Provide steadystateWindow values in seconds.
    '''
    _info = "In current clamp(CC), after bridge balance, the charging/discharging depends only on Rm and Cm.\n\
            In voltage clamp(VC), both pipette capacitance and cell capacitance contribute to charging and discharging.\n\
            Therefore, in VC, after pipette capacitance (Cp), whole cell (Cm) and series resistance compensation (Rs),\n\
            the tau and Cm values are not reflected in cell responses to voltage pulses. Instead, they should be noted\n\
            down from multiclamp commander or clampex. However, if Cm and Rs compensation is not done in VC, the transient \n\
            amplitude is proportinal Vcmd."        
    
    tau_trend = []

    if clamp=='VC':
        # print(_info)
        Cm = np.nan
        # tau_trend = np.zeros(len(tau_trend))
        for s in recordingData.values():
            cmdTrace    = s['Cmd']
            resTrace    = s[0]
            time        = s['Time']

            pulse_step_time = e2dp( [IRchargingPeriod[0]-0.001, IRchargingPeriod[0]+0.001], Fs)
            
            # minimum because the Vcmd pulse is hyperpolarizing
            Vcmd   = np.min(cmdTrace[pulse_step_time])
            Itrans = np.min(resTrace[pulse_step_time])

            Ra_eff = np.round(1000* (Vcmd) / (Itrans), 1)
            tau_trend.append(Ra_eff)

        Cm = tau_trend / Ra_eff

    elif clamp=='CC':
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

        # median Cm and Rm values
        Rm = 1000*popt[1]/Icmd # MegaOhms
        Cm = 1e6*np.median(tau_trend)/Rm #picoFarads
        

    # Tau change flag
    # Tau change screening criterion is 20% change in Tau during the recording OR tau going above 0.5s
    tau_flag      = 0
    
    if (np.percentile(tau_trend,95) / np.median(tau_trend) > 0.5) | (np.max(np.percentile(tau_trend,95)) > 0.5):
        tau_flag  = 1
    
    tau_trend = np.array(tau_trend)

    sweepwise_tau_flag = np.logical_or(tau_trend<5, tau_trend>40)

    return tau_trend, sweepwise_tau_flag, tau_flag, Cm


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
        if np.max(IRtrend)>300 or np.min(IRtrend)<= 50:  #putting a hard coded range of acceptable IR
            IRflag = 1
    IRtrend = np.array(IRtrend)
    sweepwise_IRflag = np.logical_or(IRtrend<50, IRtrend>200) 
    return IRtrend, sweepwise_IRflag, IRflag


def pulseResponseCalc(recordingData,eP):
    pulsePeriods    = []
    PeakResponses   = []
    AUCResponses    = []
    df_peaks        = pd.DataFrame()

    APflag          = bool(0)

    stimfreq        = eP.stimFreq  # pulse frequency
    Fs              = eP.Fs
    IPI_samples     = int(Fs * (1 / stimfreq))          # inter-pulse interval in datapoints
    try:
        firstPulseStart = int(Fs * eP.pulseTrainEpoch[0])
    except:
        firstPulseStart = int(Fs * eP.opticalStimEpoch[0])

    for sweepID,sweep in recordingData.items():
        ch0_cell        = sweep[0]
        ch1_frameTTL    = sweep[1]
        ch2_photodiode  = sweep[2]

        res             = []
        t1              = firstPulseStart
        for i in range(eP.numPulses):
            t2 = t1 + IPI_samples
            res.append(ch0_cell[t1:t2])
            t1 = t2
        res             = np.array(res)
        
        peakTimes       = []
        df_peaks.loc[sweepID + 1, "firstPulseDelay"],_,_ = PSP_start_time(ch0_cell,eP.clamp,eP.EorI,stimStartTime=eP.opticalStimEpoch[0],Fs=Fs)

        if eP.EorI == 'I' or eP.clamp == 'CC':
            maxRes = np.max(res, axis=1)
            aucRes = np.trapz(res,axis=1)
            PeakResponses.append(np.max(maxRes))
            
            df_peaks.loc[sweepID + 1, [1,2,3,4,5,6,7,8]] = maxRes #sweep number should start at 1 in the stored data, not from 0
            
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
    df_peaks["PeakResponse"]    = PeakResponses
    df_peaks["datafile"]        = eP.datafile

    return df_peaks, APflag

def extract_pulse_response_features(response_trace,stim_trace,stim_start, eP, Fs=2e4):
    '''A more general purpose function to extract features from any ephys response
    Features to be extracted:
        1. Peak response (pi),
        2. Area under the curve (ai),
        3. 10%-90% slope (si),
        4. Time to peak (ti),
        5. Delay to onset (di)
    '''
    ipi = 1/eP.stimFreq    

    pulse_times = get_pulse_times(eP.numPulses, eP.pulseTrainEpoch[0], eP.stimFreq)

    df  = pd.DataFrame(columns=['peak','auc','time_to_peak','delay','slope']) 

    try:
        single_pulse_time = eP.singlePulseEpoch[0]
        pulse_times = np.append(single_pulse_time, pulse_times)
    except:
        single_pulse_time = ''

    print(pulse_times)

    for Pn,PTn in enumerate(pulse_times):
        t0 = int(Fs*(PTn-ipi/2))
        t1 = int(Fs*PTn )
        t2 = int(Fs*(PTn+ipi))

        pulseStim = stim_trace[t0:t2]
        pulseRes  = response_trace[t0:t2]
        pulseRes_Abs  = np.abs(pulseRes)
        # peak
        pi = np.max(pulseRes_Abs)

        # time to peak
        # ti = (np.where(pulseRes_Abs==pi)[0][0] - ipi/2 )/Fs/1000
        p,pp = signal.find_peaks(pulseRes_Abs,height=0.9*pi,distance=500)
        
        ti = p[0]/Fs - ipi/2
        # Area under the curve
        ai = np.trapz(pulseRes_Abs)

        # delay to onset
        # di = PSP_start_time(pulseRes, eP.clamp, eP.EorI, stimStartTime=PTn)

        # 10-90% slope
        
        p90 = 0.9*pi
        p10 = 0.1*pi

        t90 = np.where(pulseRes_Abs==p90)
        t10 = np.where(pulseRes_Abs==p10)

        si  = p90-p10 / t90-t10
        di=0

        df.loc[Pn] = {'peak':pi,'auc':ai,'time_to_peak':ti,'delay':di,'slope':si}

    return df
           
        
# FIXME: remove hardcoded variables, fields, and values
