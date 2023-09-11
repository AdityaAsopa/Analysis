import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt

from eidynamics import pattern_index
from eidynamics import ephys_functions as ephysFunc
from eidynamics import utils

def expt2df(expt, neuron, eP):
    '''Returns dataframe for FreqSweep type of experiments'''
    numSweeps       = len(expt.stimCoords)
    numRepeats      = eP.repeats

    # create the dataframe that stores analyzed experiment results
    features        = ["CellID","ExptType","Condition","Clamp","EI","StimFreq","NumSquares","PulseWidth","PatternID","Intensity","Sweep","Repeat","Unit"]
    df              = pd.DataFrame(columns=features)
    df.astype({
                "CellID"    : "string",
                "ExptType"  : "string",
                "Condition" : "string",
                "Clamp"     : "string",
                "EI"        : "string",
                "StimFreq"  : 'int8',
                "NumSquares": "int8",
                "PulseWidth": "int8",
                "PatternID" : "string",
                "Intensity" : "int8",
                "Sweep"     : "int8",
                "Repeat"    : "int8",
                "Unit"      : "string"}
            )  # "Coords":'object',

    
    # fill in columns for experiment parameters,
    # they will serve as axes for sorting analysed data in plots
    for r,co in enumerate(expt.stimCoords): #.items():
        r = r+1
        df.loc[r,"Sweep"]      = int(r)
        df.loc[r,"NumSquares"] = int(len(co[3:]))  # numSquares
        try:
            df.loc[r,"PatternID"]  = int(pattern_index.get_patternID(co[3:]))
        except:
            # print(co[3:])
            df.loc[r,"PatternID"]  = 99 #int(pattern_index.get_patternID(co[3:]))

    repeatSeq       = (np.concatenate([np.linspace(1, 1, int(numSweeps / numRepeats)),
                                       np.linspace(2, 2, int(numSweeps / numRepeats)),
                                       np.linspace(3, 3, int(numSweeps / numRepeats))])).astype(int)

    df["CellID"]    = str(eP.cellID)
    df["ExptType"]  = str(eP.exptType)
    df["Condition"] = str(eP.condition)
    df["Clamp"]     = str(eP.clamp)
    df["EI"]        = str(eP.EorI)
    df["StimFreq"]  = eP.stimFreq  # stimulation pulse frequency
    df["PulseWidth"]= eP.pulseWidth
    df["Intensity"] = eP.intensity  # LED intensity
    df["Repeat"]    = repeatSeq[:numSweeps]    
    df["Unit"]      = str(eP.unit)

    # Add analysed data columns
    '''IR'''
    df["IR"],df["IRFlag"],IRflag = ephysFunc.IR_calc(expt.recordingData, eP.IRBaselineEpoch, eP.IRsteadystatePeriod, clamp=eP.clamp)
    expt.Flags.update({"IRFlag": IRflag})

    '''Ra'''
    df["Tau"],df["IRFlag"],tau_flag,_ = ephysFunc.tau_calc(expt.recordingData,eP.IRBaselineEpoch,eP.IRchargingPeriod,eP.IRsteadystatePeriod,clamp=eP.clamp)
    expt.Flags.update({"TauFlag": tau_flag})

    '''EPSP peaks'''
    df_peaks,APflag = ephysFunc.pulseResponseCalc(expt.recordingData,eP)
    expt.Flags.update({"APFlag": APflag})
    df = pd.concat([df, df_peaks],axis=1)

    # check if the response df already exists
    if not neuron.response.empty:
        neuron.response = pd.concat([neuron.response,df])
    else:
        neuron.response = df
    neuron.response = neuron.response.drop_duplicates()  # prevents duplicates from buildup if same experiment is run again.

    return expt

# FIXME: remove hardcoded variables, fields, and values
# FIXME: stray columns in excel file


## Code to add analysed parameters (peak, AUC, slope) columns to the dataframe
# there are two codes, one of them is useful
def add_analysed_params(df):
    df2 = df.iloc[:, :23].copy()

    # make a list of new columns
    new_columns = [
        'pulse_locs',
        'cell_response_peaks',
        'field_response_peaks',
        'cell_response_peak_norm',
        'field_response_peak_norm',
        'pulse_to_cell_response_peak_delay',
        'pulse_to_field_response_peak_delay',
        'cell_fpr',
        'field_fpr',
        'cell_ppr',
        'cell_stpr',
        'field_ppr',
        'field_stpr'
    ]

    # add new columns to df2 and set them to object type
    # for col in new_columns:
        # df2[col] = None
        # df2[col] = df2[col].astype('object')

    params = np.zeros((df.shape[0], 69))

    for i in range(df.shape[0]):
        Fs = 2e4
        row = df.iloc[i, :]
        ipi = int(Fs / row['stimFreq'])
        pw  = int(Fs * row['pulseWidth'] / 1000)
        

        # convert a vector of length 80000 to a 2D array of shape (4, 20000)
        [cell, framettl, led, field] = np.reshape(df.iloc[i,23:], (4, -1))

        # binarize the led signal where the led signal is above 3 standard deviations of the baseline (first 2000 points)
        try:
            led, peak_locs = utils.binarize_led_trace(led)
        except AssertionError:
            print('AssertionError: ', i, row['cellID'], row['exptID'], row['sweep'])
            continue
        peak_locs = peak_locs['left']

        # if there are 8 peaks add the first peak_loc value again at the beginning
        if len(peak_locs) == 8:
            peak_locs = np.insert(peak_locs, 0, peak_locs[0])
        if len(peak_locs) != 9:
            print('peak_locs not equal to 9:', i, len(peak_locs), row['cellID'], row['exptID'], row['sweep'])
            continue

        # for every row in df, there is going to be several outputs lists:
        # 1. a list of cell response for all pulses
        # 2. a list of field response for all pulses
        # 3. a list of locations of peak of cell responses w.r.t pulse start
        # 4. a list of locations of peak of field responses w.r.t pulse start
        # 5. a list of all pulse start locations

        sweep_pulse_locs = []
        sweep_pulse_to_cell_response_peak_delay = []
        sweep_pulse_to_field_response_peak_delay = []
        sweep_cell_response_peaks = []
        sweep_field_response_peaks = []

        for loc in peak_locs:
            cellslice = cell[loc:loc+ipi]
            fieldslice = field[loc:loc+ipi]

            # get max of cell slice
            cellpulsemax = utils.get_pulse_response(cell, loc, loc+ipi, 1, prop='peak')
            fieldpulsepeak = utils.get_pulse_response(field, loc, loc+ipi, 1, prop='p2p')

            # fill in lists:
            sweep_pulse_locs.append(loc)
            sweep_cell_response_peaks.append(cellpulsemax)
            sweep_field_response_peaks.append(fieldpulsepeak)
            sweep_pulse_to_cell_response_peak_delay.append(  np.argmax(cellslice)  - loc )
            sweep_pulse_to_field_response_peak_delay.append( np.argmax(fieldslice) - loc )

        # convert lists to numpy arrays
        sweep_pulse_locs = np.array(sweep_pulse_locs)
        sweep_cell_response_peaks = np.array(sweep_cell_response_peaks)     # to be stored in the df
        sweep_field_response_peaks = np.array(sweep_field_response_peaks)       # to be stored in the df
        sweep_pulse_to_cell_response_peak_delay = np.array(sweep_pulse_to_cell_response_peak_delay)
        sweep_pulse_to_field_response_peak_delay = np.array(sweep_pulse_to_field_response_peak_delay)

        # convert pulse locations to time
        sweep_pulse_locs = sweep_pulse_locs / Fs        # to be stored in the df
        sweep_pulse_to_cell_response_peak_delay = sweep_pulse_to_cell_response_peak_delay / Fs      # to be stored in the df
        sweep_pulse_to_field_response_peak_delay = sweep_pulse_to_field_response_peak_delay / Fs        # to be stored in the df

        # first pulse response
        cell_fpr  = sweep_cell_response_peaks[0]        # to be stored in the df
        field_fpr = sweep_field_response_peaks[0]       # to be stored in the df
        cell_ppr  = sweep_cell_response_peaks[1] / cell_fpr     # to be stored in the df
        field_ppr = sweep_field_response_peaks[1] / field_fpr       # to be stored in the df

        # another array to store normalized cell and field responses
        sweep_cell_response_peaks_norm = sweep_cell_response_peaks / cell_fpr       # to be stored in the df
        sweep_field_response_peaks_norm = sweep_field_response_peaks / field_fpr    # normalize field response to first pulse response  # to be stored in the df

        # STPR is the ratio of sum of last three pulse responses to the first pulse response
        cell_stpr  = np.sum(sweep_cell_response_peaks[-3:]) / cell_fpr       # to be stored in the df
        field_stpr = np.sum(sweep_field_response_peaks[-3:]) / field_fpr     # to be stored in the df

        # make a list of all these values
        paramlist = np.concatenate([sweep_pulse_locs, sweep_cell_response_peaks, sweep_field_response_peaks, sweep_cell_response_peaks_norm, sweep_field_response_peaks_norm, sweep_pulse_to_cell_response_peak_delay,sweep_pulse_to_field_response_peak_delay,
                    [cell_fpr], [field_fpr], [cell_ppr], [field_ppr], [cell_stpr], [field_stpr] ])
        
        params[i, :] = np.array(paramlist).flatten()

    # # append params array to df and df2
    # df = pd.concat([df, pd.DataFrame(params)], axis=1)
    # df2 = pd.concat([df2, pd.DataFrame(params)], axis=1)

    metadata_columns = df2.columns.to_list()
    metadata_cols = pd.MultiIndex.from_product([['metadata'],metadata_columns])
    df2.columns = metadata_cols

    param_cols1 = pd.MultiIndex.from_product( [['locs', 'peaks_cell', 'peak_field', 'peak_cell_norm', 'peak_field_norm', 'delay_cell', 'delay_field'], [0,1,2,3,4,5,6,7,8] ])
    paramsdf1 = pd.DataFrame(params[:,:63], columns=param_cols1)

    param_cols2 = pd.MultiIndex.from_product( [['analysed_params'],['cell_fpr', 'field_fpr', 'cell_ppr', 'cell_stpr', 'field_ppr', 'field_stpr']] )
    paramsdf2 = pd.DataFrame(params[:,63:69], columns=param_cols2)

    paramdf = pd.concat([df2, paramsdf1, paramsdf2], axis=1)

    return paramdf