import sys
import os
import datetime
import importlib
import pathlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.signal import filter_design
from scipy.signal import butter, bessel, decimate, sosfiltfilt
from scipy.signal import find_peaks, peak_widths

frameSize = [13032.25, 7419.2]  # Aug 2021 calibration


def gridSizeCalc(sqSize,objMag,frameSz=frameSize):

    gridSize = np.array([1,1])

    frameSize = (1.0 / objMag) * np.array(frameSz)
    print('frame Size is (um):', frameSize)

    gridSize[0] = frameSize[0] / sqSize[0]
    gridSize[1] = frameSize[1] / sqSize[1]

    print(f"A grid of {gridSize[0]} x {gridSize[1]} squares will create squares of"
          f" required {sqSize[0]} x {sqSize[1]} µm with an aspect ratio of {sqSize[0]/sqSize[1]}")
    print('Nearest grid Size option is...')
    print('A grid of {} squares x {} squares'.format(int(np.ceil(gridSize[0])),int(np.ceil(gridSize[1]))))

    squareSizeCalc(np.ceil(gridSize),objMag)


def squareSizeCalc(gridSize,objMag,frameSz=frameSize):
    '''
    Pass two values as the arguments for the file: [gridSizeX, gridSizeY], objectiveMag
    command line syntax should look like:  [24 24] 40
    '''
    squareSize_1x = np.array(frameSz) * (1 / objMag)
    ss = np.array([1, 1])

    if len(gridSize) == 2:
        ss[0] = squareSize_1x[0] / gridSize[0]
        ss[1] = squareSize_1x[1] / gridSize[1]
    else:
        ss = squareSize_1x / gridSize

    print(f"Polygon Square will be {ss[0]} x {ss[1]} µm with an aspect ratio of {ss[0]/ ss[1]}.")
    return ss


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def filter_data(x, filter_type='butter', low_cutoff=0.1, high_cutoff=500,sampling_freq=2e4):
    if filter_type == 'butter':
        sos = butter(N=2, Wn=high_cutoff, fs=sampling_freq, output='sos')
        y = sosfiltfilt(sos,x)
    elif filter_type == 'bessel':
        sos = bessel(4, high_cutoff, fs=sampling_freq, output='sos')
        y = sosfiltfilt(sos,x)
    elif filter_type == 'decimate':
        y = decimate(x, 10, n=4)
    elif filter_type == 'butter_bandpass':
        sos = butter_bandpass(lowcut=low_cutoff, highcut=high_cutoff, fs=sampling_freq, order=5)
        y = sosfiltfilt(sos, x)
    else:
        y = x
    return y


# map one range of values to another
def map_range( input_signal, in_min, in_max, out_min, out_max ):
    return (input_signal - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def baseline(x):
    baselineWindow = int(0.1*len(x))
    return x - np.mean(x[:baselineWindow])


def binarize_trace(trace, max_value='signal_max', method='derivative', threshold=0.5):
    '''
    max_value:    'signal_max' or 1
    method: 'derivative' or 'threshold' or '3std_dev'
    threshold: as a fraction of max value of the trace
    '''
    # baseline subtract using first 100 points as baseline
    trace = trace - np.mean(trace[:100])
    std_dev = np.std(trace[:100])
    trace = np.where(trace > 3*std_dev, trace, 0)

    if max_value == 'signal_max':
        max_trace = np.max(trace)
    else:
        max_trace = 1
    
    if method == 'derivative':
        # find derivative of the trace and filter it
        d_trace = np.diff(trace)
        d_trace[:10] = 0
        d_trace[-10:] = 0
        d_trace = filter_data(d_trace, filter_type='butter', high_cutoff=1000, sampling_freq=2e4)
        
        # get positive peaks
        max_of_d_trace = np.max(d_trace)
        # get location of the peaks
        pos_peaks, _ = find_peaks(d_trace, height=0.25*max_of_d_trace, distance=100)
        
        # get negative peaks
        d_trace = -1*d_trace
        min_of_d_trace = np.max(d_trace)
        # get location of the peaks
        neg_peaks, _ = find_peaks(d_trace, height=0.25*min_of_d_trace, distance=100)
        # assert  that the peaks are equal in number otherwise pass assertion error
        
        assert len(pos_peaks) == len(neg_peaks), "Error in photodiode signal. pos_peaks and neg_peaks are not of the same length. Peak detection fault."

        
        peak_locs = {'left': pos_peaks, 'right': neg_peaks}

        y = np.zeros(len(trace))
        # change y value between pos_peaks and neg_peaks
        
        for start,end in zip(pos_peaks, neg_peaks):
            y[start:end] = max_trace

        return y, peak_locs

    elif method == 'threshold':
        y = np.where(led_trace > threshold*max_led_trace, max_led_trace, 0)
        return y

    elif method == '3std_dev':
        std_dev = np.std(led_trace[:100])
        y = np.where(led_trace > 3*std_dev, max_led_trace, 0)
        return y

def extract_channelwise_data(sweepwise_dict,exclude_channels=[]):
    '''
    Returns a dictionary holding channels as keys,
    and sweeps as keys in an nxm 2-d array format where n is number of sweeps
    and m is number of datapoints in the recording per sweep.
    '''
    chLabels    = list(sweepwise_dict[0].keys())
    numSweeps   = len(sweepwise_dict)
    sweepLength = len(sweepwise_dict[0][chLabels[0]])
    channelDict = {}
    tempChannelData = np.zeros((numSweeps,sweepLength))

    included_channels = list( set(chLabels) - set(exclude_channels) )
    for ch in included_channels:
        for i in range(numSweeps):
            tempChannelData[i,:] = sweepwise_dict[i][ch]
        channelDict[ch] = tempChannelData
        tempChannelData = 0.0*tempChannelData            
    return channelDict


def find_response_start(x, method='stdDev'):
    '''
    Standard deviation method works on all photodiode traces, but the slope method only works for
    the photodiode traces after installation of OPT101. For recordings from Jan 2022 onwards, use slope method.
    '''
    if method == 'stdDev':    
        y       = np.abs(baseline(x))
        stdX    = np.std(y[:3000])
        movAvgX = moving_average(y,10)
        z       = np.where((movAvgX > 5. * stdX) & (movAvgX > 1.1*np.max(y[:3999])))
        # z       = find_peaks(movAvgX, height=10.0*stdX, distance=40)
        return z[0]-10

    elif method == 'slope':
        y       = np.abs(baseline(x))
        d2y     = np.diff(y,n=2) # using second derivative
        z       = np.where(d2y > 0.8 * np.max(d2y))
        return z[0][::2]+1 # +1 because taking a double derivative causes the signal to shift


def epoch_to_datapoints(epoch,Fs=2e4):
    t1 = epoch[0]
    t2 = epoch[1]
    x = np.arange(t1, t2, 1 / Fs)
    return (x * Fs).astype(int)


def charging_membrane(t, A0, A, tau):
    '''
    A0 : initial value, before command pulse is applied
    A  : steady state max value after Cm charges across Rm completely
    tau: Rm*Cm, time constant of charging, given in same units as t.

    If used for curve fitting, provide boundd and p0 (best guess) values.
    For current clamp recordings with a command pulse of -20pA and Rm of ~100MOhm:
    -10 < A0  <   10, p0 =    0
    -10 < A   <    0, p0 = -2.0
      0 > tau < 0.05, p0 = 0.02
    '''
    y = A0 + A * (1 - np.exp(-t / tau))
    return y


def alpha_synapse(t,Vmax,tau):
    a = 1/tau
    y = Vmax*(a*t)*np.exp(1-a*t)
    return y


def delayed_alpha_function(t,A,tau,delta):
    tdel = np.zeros(int(2e4*delta))
    T   = np.append(tdel,t)
    T = T[:len(t)]
    a   = 1/tau
    y   = A*(a*T)*np.exp(1-a*T)
    return y


def PSP_start_time(response_array,clamp='CC',EorI='E',stimStartTime=0.2,Fs=2e4):
    '''
    Input: nxm array where n is number of frames, m is datapoints per sweep
    '''
    if len(response_array.shape)==1:
        baseline        = mean_at_least_rolling_variance(response_array,window=500,slide=50)
        avgAllSpots     = response_array - baseline
        avgAllSpots     = avgAllSpots
        avgAllSpots     = np.where(avgAllSpots>30,30,avgAllSpots)        
    else:
        avgAllSpots     = np.mean(response_array,axis=0) 
    
    w                   = 40 if np.max(avgAllSpots)>=30 else 60
    
    if clamp == 'VC' and EorI == 'E':
        avgAllSpots     = -1*avgAllSpots
        w               = 60

    
    stimStart           = int(Fs*stimStartTime)
    avgAllSpots         = filter_data(avgAllSpots, filter_type='butter',high_cutoff=300,sampling_freq=Fs)
    movAvgAllSpots      = moving_average(np.append(avgAllSpots,np.zeros(19)),20)
    response            = movAvgAllSpots - avgAllSpots
    stdDevResponse      = np.std(response[:stimStart])
    responseSign        = np.sign(response-stdDevResponse)
    peaks               = find_peaks(responseSign[stimStart:],distance=100,width=w)

    zeroCrossingPoint   = peaks[1]['left_ips']

    PSPStartTime    = stimStart + zeroCrossingPoint + w
    
    PSPStartTime    = PSPStartTime/Fs
    
    try:
        synDelay_ms        = 1000*(PSPStartTime[0] - stimStartTime)
        valueAtPSPstart    = avgAllSpots[int(PSPStartTime[0])]
    except:
        synDelay_ms        = 0
        valueAtPSPstart    = avgAllSpots[stimStart]

    
    return synDelay_ms,valueAtPSPstart,responseSign


def rolling_variance_baseline(vector,window=500,slide=50):
    t1          = 0
    leastVar    = 1000
    leastVarTime= 0
    lastVar     = 1000
    mu          = 0
    count       = int(len(vector)/slide)
    for i in range(count):
        t2      = t1+window        
        sigmaSq = np.var(vector[t1:t2])
        if sigmaSq<leastVar:
            leastVar     = sigmaSq
            leastVarTime = t1
            mu           = np.mean(vector[t1:t2])
        t1      = t1+slide
    
    baselineAvg      = mu
    baselineVariance = sigmaSq
    baselineAvgWindow= np.arange(leastVarTime,leastVarTime+window)
    return [baselineAvg,baselineVariance,baselineAvgWindow]


def mean_at_least_rolling_variance(vector,window=2000,slide=50):
    t1          = 0
    leastVar    = np.var(vector)
    leastVarTime= 0
    lastVar     = 1000
    mu          = np.mean(vector)
    count       = int(len(vector)/slide)
    for i in range(count):
        t2      = t1+window        
        sigmaSq = np.var(vector[t1:t2])
        if sigmaSq<leastVar:
            leastVar     = sigmaSq
            leastVarTime = t1
            mu           = np.mean(vector[t1:t2])
        t1      = t1+slide
    return mu


def get_pulse_times(numPulses,firstPulseStartTime,stimFreq):
    '''Theoretical values i.e. calculated from stim frequency and 
    number of pulses. The actual light stim may have a delay of ≈20µs.
    To parse out actual stim times from stim trace, use get_pulse_times_from_stim() function.
    '''
    IPI = 1/stimFreq
    lastPulseTime = firstPulseStartTime+(numPulses-1)*IPI
    pulseTimes = np.linspace(firstPulseStartTime, lastPulseTime, num=numPulses, endpoint=True)
    return pulseTimes


def show_experiment_table(cellDirectory):
    '''Prints out a summary of all the experiments contained in a cell folder. The information
    is read from _experiment_parameter.py files.
    '''

    fileExt = "_experiment_parameters.py"
    epFiles = [os.path.join(cellDirectory, epFile) for epFile in os.listdir(cellDirectory) if epFile.endswith(fileExt)]
    df = pd.DataFrame(columns=['Cell ID','Polygon Protocol','Expt Type','Condition','Stim Freq (Hz)','Stim Intensity (%)','Pulse Width (ms)','Clamp',\
                                'Clamping Potential (mV)','EorI','sex','Age','DateOfExpt', 'Field Data Exists'])
    for epFile in epFiles:
        epfileName = pathlib.Path(epFile).stem
        epfilePath = str(pathlib.Path(epFile).parent)
        sys.path.append(epfilePath)
        ep = importlib.import_module(epfileName, epfilePath)
        exptID = ep.datafile
        if type(ep.location) is dict:
            field_data_exists = 1 if ep.location[3] != '' else 0
        else:
            field_data_exists = 0
        
        df.loc[exptID] ={
                            'Cell ID'                : ep.cellID,
                            'Polygon Protocol'       : ep.polygonProtocol[9:-4],
                            'Expt Type'              : ep.exptType,
                            'Condition'              : ep.condition,
                            'Stim Freq (Hz)'         : ep.stimFreq,
                            'Stim Intensity (%)'     : ep.intensity,
                            'Pulse Width (ms)'       : ep.pulseWidth,
                            'Clamp'                  : ep.clamp,
                            'Clamping Potential (mV)': ep.clampPotential,
                            'EorI'                   : ep.EorI,
                            'sex'                    : ep.sex,
                            'Age'                    : ep.ageAtExp,
                            'DateOfExpt'             : ep.dateofExpt,
                            'Field Data Exists'      : field_data_exists,


                        } 
    print('The Cell Directory has following experiments')
    print(df)

    return df


def cut_trace(trace1d, startpoint, numPulses, frequency, fs, prePulsePeriod = 0.020):
    ipi             = 1/frequency
    pulseStartTimes = get_pulse_times(numPulses, startpoint, frequency) - prePulsePeriod
    pulseEndTimes   = ((pulseStartTimes + ipi + prePulsePeriod)*fs).astype(int)
    pulseStartTimes = ((pulseStartTimes)*fs).astype(int)
    trace2d = np.zeros((numPulses,pulseEndTimes[0]-pulseStartTimes[0]))

    for i in range(numPulses):
        t1,t2 = pulseStartTimes[i],pulseEndTimes[i]
        trace2d[i,:] = trace1d[t1:t2]

    return trace2d


def poisson_train(avg_firing_rate, num_trials, trial_duration, firing_rate_high_cutoff=100, time_step=0.1, Fs=2e4, plot_raster=False):
    dt       = 1/Fs
    num_bins = np.floor(trial_duration/dt).astype(int)
    # np.random.seed(111)
    spikes   = np.random.rand(num_trials, num_bins)
    spikes   = np.where(spikes<avg_firing_rate*dt, 1, 0)
    time     = np.linspace(0, trial_duration, int(trial_duration/dt))

    spiketrain = spikes[0]
    
    spike_locs = np.where(spiketrain)[0]
    spiketrain_filtered = spiketrain.copy()

    omit_spikes = []

    # remove spikes that occur earlier than firing rate high cutoff ISI
    for i,pp in enumerate(spike_locs[:-1]):

        spike_loc1 = spike_locs[i]
        spike_loc2 = spike_locs[i+1]
        
        if (spike_loc2-spike_loc1) < (Fs/firing_rate_high_cutoff):
            omit_spikes.append(spike_loc2)

    spiketrain_filtered[omit_spikes] = 0

    spike_times = get_event_times([spiketrain_filtered])

    isi = np.array([])
    for trial in spike_times:
        isi_trial =  np.diff(trial,1)
        isi = np.concatenate((isi,isi_trial),axis=0)

    acfr = Fs * kernel_convoluted_firing_rate(spiketrain_filtered, 0.1, kernel='alpha')[0]

    if plot_raster:
        fig = plt.figure(1)
        fig.suptitle('Generated Poisson Spike Train Data')
        gridspec.GridSpec(3,2)

        plt.subplot2grid((3,2), (0,0), colspan=1, rowspan=1)
        plt.title('Spike Train')
        plt.xlabel('Time')
        plt.ylabel('Trials')
        plt.eventplot(spike_times)

        # small subplot 1
        plt.subplot2grid((3,2), (0,1), colspan=1, rowspan=1)
        plt.title('Inter Spike Interval Distribution')
        plt.xlabel('ISI (second)')
        plt.ylabel('Count')
        plt.hist(isi, bins=int(max(isi)/0.005), density=True)

        # small subplot 2
        plt.subplot2grid((3,2), (1,0), colspan=2, rowspan=1)
        plt.title('Spike Train')
        plt.xlabel('Time')
        plt.ylabel('Spikes')
        plt.plot(spiketrain_filtered, color='b')

        # small subplot 2
        plt.subplot2grid((3,2), (2,0), colspan=2, rowspan=1)
        plt.title('Alpha convoluted firing rate')
        plt.xlabel('Time')
        plt.ylabel('ACFR (Hz)')
        plt.plot(acfr, color='k')

        fig.show()



    return spiketrain_filtered, spike_times, isi, time, acfr


def kernel_convoluted_firing_rate(spiketrain, sigma, kernel='alpha'):
    '''
    Reference: 1.2 Spike Trains and Firing Rates, Computational Neuroscience, Dayan and Abbott, page 12-13 
    '''
    size = 6*sigma
    tau  = np.linspace(-size/2, size/2, int(2e4*size) )
    alpha= 1/sigma

    alphafilt = ( ((alpha**2)*tau ) * np.exp(-alpha*tau) )
    # rectification
    alphafilt = np.where(alphafilt<0, 0, alphafilt)
    alphafilt = alphafilt / np.sum(alphafilt)

    kcfr = np.convolve(spiketrain, alphafilt, mode="valid")

    return kcfr, alphafilt


def get_event_times(spike_matrix, Fs=2e4):
    spike_times = []
    for trial in spike_matrix:
        spike_locs = (np.where(trial)[0]/Fs).tolist() # dividing by Fs to get spike times in seconds
        # as number of spike events in a trial vary, it is better to store spike times as list of lists rather than
        # numpy 2D array as the latter does not like rows to have different lengths.
        spike_times.append(spike_locs)  
    return spike_times


def _find_fpr(stimFreq_array, res_window_matrix, clamp_pot_array, clamp_array):
    '''
    'clamp_pot_array' is the array of clamp potentials for each trial
    clamp_array = 'VC' or 'CC'
    '''
    stimFreq_array = stimFreq_array.to_numpy(copy=True)
    clamp_pot_array = clamp_pot_array.to_numpy(copy=True)
    clamp_array = clamp_array.to_numpy(copy=True)
    
    if stimFreq_array.shape[0] != res_window_matrix.shape[0]:
        raise ValueError

    fpr      = np.zeros([stimFreq_array.shape[0]])
    fpr_time = np.zeros([stimFreq_array.shape[0]])

    for i in range(stimFreq_array.shape[0]):
        f   = stimFreq_array[i]
        ipi = int(2e4/f)

        trace = res_window_matrix.iloc[i,:ipi].to_numpy()

        if (clamp_array[i]=='VC'):
            if (clamp_pot_array[i]== -70.0):
                # trace      *= -1.0
                fpr[i]      = np.min(trace)
                fpr_time[i] = np.where(trace<= np.min(trace))[0][0] + 1
            else:
                fpr[i]      = np.max(trace)
                fpr_time[i] = np.where(trace>= np.max(trace))[0][0] + 1
        else:
            fpr[i]      = np.max(trace)
            fpr_time[i] = np.where(trace>= np.max(trace))[0][0] + 1

    return fpr, fpr_time


def generate_optical_stim_waveform():
    spikedata = poisson_train(30, 1, 10, plot_raster=True)

    Fs = 2e4
    spiketrain = spikedata[0]
    kept_spike_locs = np.where(spiketrain)[0]
    pulse_width = 0.002 # ms
    for loc in kept_spike_locs:
        spiketrain[loc:loc+(int(pulse_width*Fs))] = 1
    
    total_sweep_duration = 12 #seconds
    full_sweep = np.zeros(int(12*Fs))

    zeroth_pulse = epoch_to_datapoints([0.2, 0.202], Fs)
    train_epoch  = epoch_to_datapoints([0.5,  10.5], Fs) 
    full_sweep[zeroth_pulse] = 1
    full_sweep[train_epoch] = spiketrain

    fig2 = plt.figure(2)
    plt.plot(full_sweep)
    fig2.show()

    fig3 = plt.figure(3)
    plt.plot(spikedata[-1])
    fig3.show()

    output_trace = np.concatenate([[full_sweep]]*5, axis = 0).T

    np.savetxt("spike_train_12s_5sweeps.txt", output_trace)


def progress_bar(current, total, bar_length=80):
    filled_length = int(bar_length * current / total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f'\rProgress: |{bar}| {100 * current / total:.2f}%', end='\n')
    if current == total:
        print()


def reset_and_print(current, total, clear=False, message=''):
    
    if clear:
        if(os.name == 'posix'):
            os.system('clear')
        else:
            os.system('cls')
    print(message)
    progress_bar(current, total)


def generate_expt_sequence(exptIDs):
    '''
    ['2022_06_01_0001',
    '2022_06_01_0004',
    '2022_06_01_0005',
    '2022_06_01_0006',
    '2022_06_01_0007',
    '2022_06_01_0009',
    '2022_06_01_0010',
    '2022_06_01_0011']
    '''
    exptIDs = np.unique(exptIDs)
    exptSeq = np.arange(len(exptIDs))
    exptSeq_LUT = {}

    for i, exptFullName in enumerate(exptIDs):
        x = exptFullName.split('.')[0].split('_rec')[0][-3:]
        ID = int(x)
        exptSeq_LUT[ID] = exptSeq[i]

    return exptSeq_LUT


def parse_other_experiment_param_file(parameterFilePath, user='Sulu'):
    
    parameterFilePath = Path(parameterFilePath)
    paramfileName = parameterFilePath.stem
    parameterFilePath = str(parameterFilePath.parent)

    print("parameterFilePath: ", parameterFilePath, '\n', "paramfileName: ", paramfileName)
    sys.path.append(parameterFilePath)
    ep = importlib.import_module(paramfileName, parameterFilePath)


    # Fill in the missing parametes

    '''
    Required parameters:
    'cellID', 'sex','ageAtInj','ageAtExpt','incubation', 'unit',
    'protocol','exptSeq','exptID','sweep', 'stimFreq', 'numSq', 'intensity',
    'pulseWidth', 'clampMode', 'clampPotential', 'condition', 'AP', 'IR', 'tau',
    'numPatterns','patternList', 'sweepBaseline'
    '''
    ep.datafile = Path(ep.datafilepath).stem + Path(ep.datafilepath).suffix

    if not 'cellID' in dir(ep):
        ep.cellID = str(ep.animalID)[1:] + str(1)

    ep.clampPotential = int(ep.clampPotential)
    ep.repeats = ep.NUM_TRIALS
    ep.Fs = ep.SAMPLING_RATE

    ep.dateofExpt = ep.dateofExpt.today()
    ep.bathTemp = ''
    ep.sex = 'X'
    ep.location = 'CA1'
    ep.dateofBirth = datetime.date(2021, 1, 1)
    ep.dateofInj = datetime.date(2021, 2, 1)
    ep.ageAtInj        = (ep.dateofInj	- ep.dateofBirth)
    ep.ageAtExp        = (ep.dateofExpt	- ep.dateofBirth)
    ep.incubation      = (ep.ageAtExp	- ep.ageAtInj)
    
    ep.numPulses= 2
    ep.opticalStimEpoch = [0, ep.PRE_STIM_DURATION]
    ep.sweepDuration = 1.0
    ep.sweepBaselineEpoch = [0, 0.2]

    ep.IRBaselineEpoch = [0, 0.2]
    ep.IRpulseEpoch = [0.765, 0.815]
    ep.IRchargingPeriod = [0.765, 0.775]
    ep.IRsteadystatePeriod = [0.790, 0.835]

    ep.unit = 'pA' if ep.clamp == 'VC' else 'mV' if ep.clamp == 'CC' else 'a.u.'

    ep.site = {'RC':1.9, 'ML':2.0, 'DV':1.5}
    ep.injectionParams = {'Pressure':8, 'pulseWidth':28, 'duration':30}
    
    ep.virus = 'ChR2'
    ep.virusTitre = 6e12
    
    ep.volumeInj = 5e-4
    
    ep.objMag = 40
    ep.frameSize = [0, 0]
    
    ep.gridSize = [ ep.GRID_SIZE[0], ep.GRID_SIZE[1] ]
    ep.squareSize = [0 , 0]

    return ep    


def get_pulse_response(x, start_time, end_time, Fs, prop='auc'):
    '''
    This function returns the response value of the signal
    x: 1D array
    start_time: in seconds
    end_time: in seconds
    Fs: sampling rate
    prop:   'auc'=area under the response,
            'peak' = max,
            'slope' = 10-90% of the peak
            'onset_delay' = time to reach 10% of the peak
            'peak_time' = time to reach the peak
            'p2p' = peak to peak amplitude
            'abs_auc' = absolute area under the response
    '''
    start_index = int(start_time*Fs)
    end_index = int(end_time*Fs)

    if prop == 'auc':
        return np.trapz(x[start_index:end_index], dx=1/Fs)
    
    elif prop == 'peak':
        return np.max(x[start_index:end_index])
    
    elif prop == 'slope':
        # get 10% and 90% of the peak
        peak = np.max(x[start_index:end_index])
        peak_index = np.argmax(x[start_index:end_index])
        peak_10 = peak*0.1
        peak_90 = peak*0.9

        # use np.where to get the index of the first value that is greater than peak_10 and peak_90
        peak_10_index = np.where(x[start_index:end_index] > peak_10)[0][0]
        peak_90_index = np.where(x[start_index:end_index] > peak_90)[0][0]

        # convert index to time
        dy = peak_90 - peak_10
        dx = (peak_90_index - peak_10_index) / (Fs/1000) # divide Fs by 1000 to get time in ms

        # get slope
        slope = dy/dx #mV/ms
        return slope
    
    elif prop == 'onset_delay':
        # find time point where response suddenly changes
        flick_time = np.argmax( np.diff(x[start_index:end_index], 2) )

        # find the index of the time point
        # print(flick_time, start_index)
        # onset_delay = flick_time

        return flick_time / Fs
    
    elif prop == 'time_to_peak':
        peak_time = np.argmax(x[start_index:end_index])
        return peak_time / Fs

    elif prop == 'p2p':
        # get peak to peak amplitude
        # first get the minimum value and maximum value
        min_x = np.min(x[start_index:end_index])
        max_x = np.max(x[start_index:end_index])
        return max_x - min_x

    elif prop == 'abs_auc':
        # get absolute area under the curve
        return np.trapz(np.abs(x[start_index:end_index]), dx=1/Fs)


def add_row_to_df(df, row):
    # write a function to add a row to the dataframe
    df.loc[-1] = row
    df.index = df.index + 1
    return df.sort_index()