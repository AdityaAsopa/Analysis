import numpy as np
from scipy import signal
import pandas as pd

from eidynamics import pattern_index
from eidynamics import ephys_functions as ephysFunc


def expt2df(expt,neuron,eP):
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
    for r,co in expt.stimCoords.items():
        df.loc[r,"Sweep"]      = int(r)
        df.loc[r,"NumSquares"] = int(len(co))  # numSquares
        df.loc[r,"PatternID"]  = int(pattern_index.get_patternID(co))

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
    df["IR"],df["IRFlag"],IRflag = ephysFunc.IR_calc(expt.recordingData,eP.clamp,eP.IRBaselineEpoch,eP.IRsteadystatePeriod)
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
