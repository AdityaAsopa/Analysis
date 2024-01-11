import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt

from eidynamics import pattern_index
from eidynamics import ephys_functions as ephysFunc
from eidynamics import utils
import pulselocs_default
n = len(utils.metadata_parameters)

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
    df2 = df.iloc[:, :n].copy()

    numPulses = len(df['pulseTimes'][0])

    params = np.zeros((df.shape[0], 72))

    for i in range(df.shape[0]):
        Fs = 2e4
        row = df.iloc[i, :]
        ipi = int(Fs / row['stimFreq'])
        pw  = int(Fs * row['pulseWidth'] / 1000)

        cellID = int(row['cellID'])
        exptID = int(row['exptID'])
        sweepID = int(row['sweep'])

        params[i, :3] = [cellID, exptID, sweepID]

        # convert a vector of length 80000 to a 2D array of shape (4, 20000)
        [cell, framettl, led, field] = np.reshape(df.iloc[i,n:], (4, -1))

        # invert the cell signal if clampMode=='VC' and clampPotential==-70
        if row['clampMode'] == 'VC' and row['clampPotential'] == -70:
            cell = -1*cell

        # binarize the led signal where the led signal is above 3 standard deviations of the baseline (first 2000 points)
        # try:
        #     led, peak_locs = utils.binarize_trace(led)
        # except AssertionError:
        #     print('AssertionError: ', i, cellID, exptID, sweepID)
        #     params[i, 3:] = np.zeros(69)
        #     continue
        # peak_locs = peak_locs['left']

        # # if there are 8 peaks add the first peak_loc value again at the beginning
        # if len(peak_locs) == 8:
        #     peak_locs = np.insert(peak_locs, 0, peak_locs[0])
        # if len(peak_locs) != 9:
        #     print('peak_locs not equal to 9:', i, len(peak_locs), cellID, exptID, sweepID)
        #     params[i, 3:] = np.zeros(69)
        #     continue
        peak_locs = row['pulseTimes']
        numPulses = len(peak_locs)

        # for every row in df, there is going to be several outputs lists:
        # 1. a list of cell response for all pulses
        # 2. a list of field response for all pulses
        # 3. a list of locations of peak of cell responses w.r.t pulse start
        # 4. a list of locations of peak of field responses w.r.t pulse start
        # 5. a list of all pulse start locations

        sweep_pulse_locs = np.zeros(len(peak_locs))
        sweep_pulse_to_cell_response_peak_delay = np.zeros(len(peak_locs))
        sweep_pulse_to_field_response_peak_delay = np.zeros(len(peak_locs))
        sweep_cell_response_peaks = np.zeros(len(peak_locs))
        sweep_field_response_peaks = np.zeros(len(peak_locs))

        for j, loc in enumerate(peak_locs):
            cellslice = cell[loc:loc+ipi]
            fieldslice = field[loc:loc+ipi]

            # get max of cell slice
            cellpulsemax = utils.get_pulse_response(cell, loc, loc+ipi, 1, prop='peak')
            fieldpulsepeak = utils.get_pulse_response(field, loc, loc+ipi, 1, prop='p2p')

            # fill in lists:
            sweep_pulse_locs[j] = loc / Fs
            sweep_cell_response_peaks[j] = cellpulsemax
            sweep_field_response_peaks[j] = fieldpulsepeak
            sweep_pulse_to_cell_response_peak_delay[j] =   ( np.argmax(cellslice)  - loc ) / Fs
            sweep_pulse_to_field_response_peak_delay[j] =  ( np.argmax(fieldslice) - loc ) / Fs

        # first pulse response
        cell_fpr  = sweep_cell_response_peaks[0]        
        field_fpr = sweep_field_response_peaks[0]       
        cell_ppr  = sweep_cell_response_peaks[1] / cell_fpr     
        field_ppr = sweep_field_response_peaks[1] / field_fpr       

        # another array to store normalized cell and field responses
        sweep_cell_response_peaks_norm = sweep_cell_response_peaks / cell_fpr       
        sweep_field_response_peaks_norm = sweep_field_response_peaks / field_fpr    # normalize field response to first pulse response  

        # STPR is the ratio of sum of last three pulse responses to the first pulse response
        cell_stpr  = np.sum(sweep_cell_response_peaks[-3:]) / cell_fpr       
        field_stpr = np.sum(sweep_field_response_peaks[-3:]) / field_fpr     

        # make a list of all these values
        paramlist = np.concatenate([sweep_pulse_locs, sweep_cell_response_peaks, sweep_field_response_peaks, sweep_cell_response_peaks_norm, sweep_field_response_peaks_norm, sweep_pulse_to_cell_response_peak_delay,sweep_pulse_to_field_response_peak_delay,
                    [cell_fpr], [field_fpr], [cell_ppr], [field_ppr], [cell_stpr], [field_stpr] ])
        
        params[i, 3:] = np.array(paramlist).flatten()


    fields, idx = ['locs', 'peaks_cell', 'peak_field', 'peak_cell_norm', 'peak_field_norm', 'delay_cell', 'delay_field'], range(numPulses)
    col_names = []
    for f in fields:
        for id in idx:
            col_names.append(f + '_'+ str(id))
    for x in ['cell_fpr', 'field_fpr', 'cell_ppr', 'cell_stpr', 'field_ppr', 'field_stpr']:
        col_names.append(x)

    # make dataframe from analysed_params and rename first 3 columns as cellID, exptID, sweep
    analysed_params_df = pd.DataFrame(params)

    analysed_params_df.columns = ['cellID', 'exptID', 'sweep'] + col_names

    # make first three columns integers type
    analysed_params_df[['cellID', 'exptID', 'sweep']] = analysed_params_df[['cellID', 'exptID', 'sweep']].astype(int)

    df_short = pd.merge(df.iloc[:, :34], analysed_params_df, on=['cellID', 'exptID', 'sweep'])
    df_all = pd.merge(df, analysed_params_df, on=['cellID', 'exptID', 'sweep'])

    # shift last 69 columns of df_all after 34th column
    df_all = pd.concat( [df_all.iloc[:,:34], df_all.iloc[:,-69:], df_all.iloc[:,34:-69]], axis=1 ) #[x[:34], x[-69:], x[34:-69]]

    return df_short, df_all

def add_analysed_params2(df):
    df2 = df.iloc[:, :n]
    print(df.iloc[0,:50])

    # make a new analysed param df
    df3 = pd.DataFrame(columns=['cellID', 'exptID', 'sweep', 'peaks_cell', 'peaks_cell_norm', 'auc_cell', 'slope_cell', 'delay_cell','peaks_field', 'peaks_field_norm', 'cell_fpr', 'field_fpr', 'cell_ppr', 'cell_stpr', 'field_ppr', 'field_stpr'])
    df3.astype({'cellID': 'int32', 'exptID': 'int32', 'sweep': 'int32', 'peaks_cell': 'float32', 'peaks_field': 'float32', 'auc_cell': 'float32', 'slope_cell': 'float32', 'delay_cell': 'float32', 'cell_fpr': 'float32', 'field_fpr': 'float32'})


    for i in range(df.shape[0]):
        
        Fs = 2e4
        row = df.iloc[i, :]
        stimfreq = row['stimFreq']
        ipi = int(Fs / stimfreq)
        pw  = int(Fs * row['pulseWidth'] / 1000)
        protocol = row['protocol']
        numPulses = row['numPulses']
        probe_pulse_start = row['probePulseStart']
        
        # add cell ID to the row in df3
        df3.loc[i, 'cellID'] = int(row['cellID'])
        df3.loc[i, 'exptID']= int(row['exptID'])
        df3.loc[i, 'sweep']= int(row['sweep'])

        # convert a vector of length 80000 to a 2D array of shape (4, 20000)
        [cell, framettl, led, field] = np.reshape(df.iloc[i,n:], (4, -1))

        # invert the cell signal if clampMode=='VC' and clampPotential==-70
        if row['clampMode'] == 'VC' and row['clampPotential'] == -70:
            cell = -1*cell

        # binarize the led signal where the led signal is above 3 standard deviations of the baseline (first 2000 points)
        try:
            led, peak_locs = utils.binarize_trace(led)
            peak_locs = peak_locs['left']
        except AssertionError:
            print('AssertionError: ', protocol, i, row['cellID'], row['exptID'], row['sweep'])
            # print(f'peak_locs not equal to {numPulses}:', protocol, i, len(peak_locs), row['cellID'], row['exptID'], row['sweep'])
            # if peak_locs length is not correct then use the default values for each protocol
            if protocol == 'SpikeTrain':
                peak_locs = pulselocs_default.SpikeTrain
                print("Pulse locs loaded from default")
            elif (protocol == 'FreqSweep') & (probe_pulse_start == 0.2):
                peak_locs = np.concatenate([np.array([4000]),(20000*utils.get_pulse_times(8,10000/20000,stimfreq)).astype(int)])
                peak_locs = peak_locs[peak_locs <= len(cell)-ipi]
                print("Pulse locs loaded from default")
            elif (protocol == 'FreqSweep') & (probe_pulse_start > 0.22):
                peak_locs = np.concatenate([np.array([4481]),(20000*utils.get_pulse_times(8,4481/20000,stimfreq)).astype(int)])
                peak_locs = peak_locs[peak_locs <= len(cell)-ipi]
                print("Pulse locs loaded from default")
            # continue
        
        
        # if there are 8 peaks add the first peak_loc value again at the beginning
        if (protocol=='FreqSweep') & (len(peak_locs) == 8):
            peak_locs = np.insert(peak_locs, 0, peak_locs[0])

        if (protocol=='FreqSweep') & (numPulses == 8):
            numPulses = 9
        if (protocol=='Surprise') & (numPulses == 32):
            numPulses = 33
        if (protocol=='grid') & (numPulses == 8):
            numPulses = 9
        if (protocol=='grid') & (numPulses == 1):
            numPulses = 1
        if (protocol=='convergence') & (numPulses == 8):
            numPulses = 9
        if (protocol=='convergence') & (numPulses == 3):
            numPulses = 3
        if (protocol == 'SpikeTrain'):
            numPulses = 230
            
        if len(peak_locs) != numPulses:
            print(f'peak_locs not equal to {numPulses}:', protocol, i, len(peak_locs), row['cellID'], row['exptID'], row['sweep'])
            # if peak_locs length is not correct then use the default values for each protocol
            if protocol == 'SpikeTrain':
                peak_locs = pulselocs_default.SpikeTrain
                print("Pulse locs loaded from default")
            elif (protocol == 'FreqSweep') & (probe_pulse_start == 0.2):
                peak_locs = np.concatenate([np.array([4000]),(20000*utils.get_pulse_times(8,10000/20000,stimfreq)).astype(int)])
                peak_locs = peak_locs[peak_locs <= len(cell)-ipi]
                print("Pulse locs loaded from default")
            elif (protocol == 'FreqSweep') & (probe_pulse_start > 0.22):
                peak_locs = np.concatenate([np.array([4481]),(20000*utils.get_pulse_times(8,4481/20000,stimfreq)).astype(int)])
                peak_locs = peak_locs[peak_locs <= len(cell)-ipi]
                print("Pulse locs loaded from default")
            elif protocol == 'LTMRand':
                peak_locs = np.concatenate([np.array([4631]),(20000*utils.get_pulse_times(8,4631/20000,stimfreq)).astype(int)])
                print("Pulse locs loaded from default")
            elif protocol == 'surprise':
                peak_locs = np.concatenate([np.array([4007]),(20000*utils.get_pulse_times(33,4007/20000,stimfreq)).astype(int)])
                print("Pulse locs loaded from default")
            elif (protocol == 'convergence') & (numPulses == 9):
                peak_locs = np.concatenate([np.array([9585]),(20000*utils.get_pulse_times(8,9585/20000,stimfreq)).astype(int)])
                print("Pulse locs loaded from default")
            elif (protocol == 'convergence') & (numPulses == 3):
                peak_locs = np.concatenate([np.array([4631]),(20000*utils.get_pulse_times(3,4631/20000,stimfreq)).astype(int)])
                print("Pulse locs loaded from default")
            elif (protocol == 'grid') & (numPulses == 1):
                peak_locs = np.array([700,700])
                print("Pulse locs loaded from default")
            elif (protocol == 'grid') & (numPulses == 9):
                peak_locs = np.concatenate([np.array([3680]),(20000*utils.get_pulse_times(8,9680/20000,stimfreq)).astype(int)])
                print("Pulse locs loaded from default")


        sweep_cell_response_peaks = np.zeros(len(peak_locs))
        sweep_cell_response_aucs = np.zeros(len(peak_locs))
        sweep_cell_response_slopes = np.zeros(len(peak_locs))
        sweep_pulse_to_cell_response_peak_delay = np.zeros(len(peak_locs))
        sweep_field_response_peaks = np.zeros(len(peak_locs))

        # print(row['cellID'], row['exptID'], row['sweep'], len(peak_locs), peak_locs)
        for j, loc in enumerate(peak_locs):
            
            cellslice = cell[loc:loc+ipi]
            fieldslice = field[loc:loc+ipi]
            # print(j, loc, ipi, cellslice.shape, fieldslice.shape)

            # get max of cell slice
            cellpulsemax = utils.get_pulse_response(cell, loc, loc+ipi, 1, prop='peak')
            cellpulseauc = utils.get_pulse_response(cell, loc, loc+ipi, 1, prop='auc')
            # cellpulseslope = utils.get_pulse_response(cell, loc, loc+ipi, 1, prop='slope')
            fieldpulsepeak = utils.get_pulse_response(field, loc, loc+ipi, 1, prop='p2p')

            # fill in lists:
            sweep_cell_response_peaks[j] = cellpulsemax
            sweep_cell_response_aucs[j] = cellpulseauc
            # sweep_cell_response_slopes[j] = cellpulseslope
            sweep_pulse_to_cell_response_peak_delay[j] =   ( np.argmax(cellslice)  - loc ) / Fs
            sweep_field_response_peaks[j] = fieldpulsepeak

        # start adding properties to df3
        df3.at[i,'peaks_cell' ]= sweep_cell_response_peaks       
        df3.at[i,'peaks_field'] = sweep_field_response_peaks      
        df3.at[i,'auc_cell'   ]= sweep_cell_response_aucs        
        df3.at[i,'slope_cell' ]= sweep_cell_response_slopes      
        df3.at[i,'delay_cell' ]= sweep_pulse_to_cell_response_peak_delay      

        # first pulse response
        df3.at[i,'cell_fpr'  ]= sweep_cell_response_peaks[0]        
        df3.at[i,'field_fpr' ]= sweep_field_response_peaks[0]       
        df3.at[i,'cell_ppr'  ]= sweep_cell_response_peaks[1] / sweep_cell_response_peaks[0]     
        df3.at[i,'field_ppr' ]= sweep_field_response_peaks[1] / sweep_field_response_peaks[0]       

        # another array to store normalized cell and field responses
        df3.at[i,'peaks_cell_norm' ]= sweep_cell_response_peaks / sweep_cell_response_peaks[0]       
        df3.at[i,'peaks_field_norm'] = sweep_field_response_peaks / sweep_field_response_peaks[0]    # normalize field response to first pulse response  

        # STPR is the ratio of sum of last three pulse responses to the first pulse response
        df3.at[i,'cell_stpr' ] = np.sum(sweep_cell_response_peaks[-3:]) / sweep_cell_response_peaks[0]       
        df3.at[i,'field_stpr'] = np.sum(sweep_field_response_peaks[-3:]) / sweep_field_response_peaks[0]     


    # make first three columns integers type
    print("Churning through sweeps complete. merging dataframes...")
    df3[['cellID', 'exptID', 'sweep']] = df3[['cellID', 'exptID', 'sweep']].astype(int)
    df_short = pd.merge(df.iloc[:, :n], df3, on=['cellID', 'exptID', 'sweep'])
    
    # print('converting df into float32')
    # float64_cols = df.select_dtypes(include=[np.float64]).columns
    # df[float64_cols] = df[float64_cols].astype(np.float32)
    
    df_all = pd.merge(df, df3, on=['cellID', 'exptID', 'sweep'])

    # shift last 69 columns of df_all after 34th column
    print("merging complete. shifting columns...")
    df_all = pd.concat( [df_all.iloc[:,:n], df_all.iloc[:,-13:], df_all.iloc[:,n:-13]], axis=1 ) #[x[:34], x[-69:], x[34:-69]]

    return df_short, df_all