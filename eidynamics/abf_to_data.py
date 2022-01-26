import sys
import pyabf
import numpy as np
import matplotlib.pyplot as plt

# tag FIXME HACK
# if the module is imported 
try: 
    from eidynamics.utils import epoch_to_datapoints, extract_channelwise_data, filter_data, plot_abf_data
# when the module is run from command line
except ModuleNotFoundError:
    from utils            import epoch_to_datapoints, extract_channelwise_data, filter_data, plot_abf_data

def abf_to_data(abf_file,exclude_channels=[],
                baseline_criterion=0.1, sweep_baseline_epoch=[0, 0.2], baseline_subtraction=True,
                signal_scaling=1, sampling_freq=2e4, filter_type='', filter_cutoff=2e3,
                data_order="sweepwise", plot_data=False ):
    """
    A wrapper around pyabf module to generate sweep wise dictionary of recorded traces.
    Returns a list with [dict holding sweepwisedata, mean baseline value, baseline fluctuation flag]
    Usual Channel Labels:
        0       : Cell,
        1       : FrameTTL,
        2       : Photodiode,
        3       : Field,
        'Time'  : Time Axis,
        'Cmd'   : Ch0 Command Signal

    arguments:
        abf_file            = path to abf file
        exclude_channels    = list of which channels not to include in the output, default = '', Example: ['Cmd','Time',0,1,2,3]
        baseline_criterion  = whether to flag a sweep if baseline fluctuates more than the specified fraction
        sweep_baseline_epoch= [t_start, t_stop], window in seconds to use for baseline calculation
        baseline_subtration = whether to offset the traces to zero baseline or not
        signal_scaling      = due to a DAQ glitch, sometimes the units and signal scaling are not proper. Whether to correct for that or not.
        sampling_freq       = default 20000 samples/second
        filter_type         = default: 'None'. Others: 'bessel', 'butter', or 'decimate'. Check eidynamics.utils.filter_data()
        filter_cutoff       = upper cutoff. default: 2000Hz. to filter spikes use 100-200Hz. Check eidynamics.utils.filter_data()
        data_order          = whether to return a sweep wise or a channel wise dictionary of data
        plot_data           = whether to display all channels data, expensive    
    
    returns:
        data                = sweepwise, or if optionally specified channelwise, all recorded traces
        medianBaseline      = median baseline value of Ch#0 in the baseline epoch, median so that extreme fluctuations don't affect
        baselineFlag        = default: False. True if fluctuations in baseline values is higher than baseline_criterion
    """

    try:
        if abf_file:
            print('Loading ABF file')
    except FileNotFoundError as err:
        print(err)
        print('Please provide valid file')

    abf             = pyabf.ABF(abf_file)
    numSweeps       = abf.sweepCount
    numChannels     = abf.channelCount

    data            = {}
    sweepArray      = {}

    baselineValues  = np.zeros([numSweeps,1])
    medianBaseline  = 0.0
    baselineFlag    = False

    for sweep in range(numSweeps):
        abf.setSweep(sweepNumber=sweep)
        # Optionally exporting time axis and Ch0 command waveform for every sweep
        if not 'Cmd' in exclude_channels:
            sweepArray.update({'Cmd':abf.sweepC})
        if not 'Time' in exclude_channels:
            sweepArray.update({'Time':abf.sweepX})

        for ch in range(numChannels):
            abf.setSweep(sweepNumber=sweep, channel=ch)
            if ch==0 and not ch in exclude_channels:
                _parsedSweep,_swpBaseline   = _baseline_subtractor(abf.sweepY,sweep_baseline_epoch,sampling_freq,subtract_baseline=baseline_subtraction)
                baselineValues[sweep]       = _swpBaseline
                parsedSweep                 = filter_data(_parsedSweep,filter_type=filter_type,high_cutoff=filter_cutoff,sampling_freq=sampling_freq)
                sweepArray.update({ch: parsedSweep/signal_scaling})
            elif ch!=0 and not ch in exclude_channels:
                parsedSweep,_               = _baseline_subtractor(abf.sweepY,sweep_baseline_epoch,sampling_freq,subtract_baseline=True)
                sweepArray.update({ch: parsedSweep/signal_scaling})
            else:
                pass

        
        data[sweep] = sweepArray
        sweepArray  = {}

    if np.all(baselineValues):    
        medianBaseline       = np.median(baselineValues)
        baseline_fluctuation = abs( np.std(baselineValues,ddof=1) / np.mean(baselineValues) ) # coefficient of variation of the baseline values, in fraction
        baselineFlag         =  (baseline_fluctuation > baseline_criterion) # Baseline flag is set True if fluctuation > screening criterion

    print(f'Datafile has {numSweeps} sweeps in {len(data[0])} channels.')

    if data_order == 'channelwise':
        data = extract_channelwise_data(data)

    if plot_data:
        plot_abf_data(data)

    return data,medianBaseline,baselineFlag

    '''pyABF sweep extraction reference'''
    # abf.setSweep(sweepNumber: 3, channel: 0)
    # print(abf.sweepY) # displays sweep data (ADC)
    # print(abf.sweepX) # displays sweep times (seconds)
    # print(abf.sweepC) # displays command waveform (DAC)

def _baseline_subtractor(sweep,sweep_baseline_epoch,sampling_freq,subtract_baseline):
    baselineWindow = epoch_to_datapoints(sweep_baseline_epoch,sampling_freq)
    '''
    Methods to calculate baseline:
        Method 1: Mean of the baseline epoch, default, standard, and preferred
        Method 2: Mean at least rolling variance in the sweep, not right as baseline should be from a predefined epoch
        Method 3: 10%ile value from baseline epoch, Upi's suggestion, not sure how right is that to use
    '''
    # Method 1
    sweepBaseline  = np.mean(sweep[baselineWindow])
    # Method 2
    # sweepBaseline = _mean_at_least_rolling_variance(sweep[13000:20000],window=2000)
    # Method 3
    # sweepBaseline = np.percentile(sweep[baselineWindow],10) 

    if subtract_baseline:
        sweepNew = sweep - sweepBaseline
        return sweepNew, sweepBaseline
    else:
        return sweep, sweepBaseline

def _mean_at_least_rolling_variance(vector,window=2000,slide=50):
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

if __name__ == '__main__':
    abfFile = sys.argv[1]
    abf_to_data(abfFile,plot_data=True)