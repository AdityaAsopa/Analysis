# Libraries
from doctest import DocFileSuite
import numpy as np
import pandas as pd
import pickle
import os
import h5py
from scipy.optimize  import curve_fit
from scipy import signal
from PIL import Image, ImageOps

# EI Dynamics module
from eidynamics                     import abf_to_data
from eidynamics.expt_to_dataframe   import expt2df
from eidynamics.spiketrain          import spiketrain_analysis
from eidynamics.ephys_functions     import IR_calc, tau_calc, spike_detect
from eidynamics.utils               import filter_data, delayed_alpha_function, PSP_start_time, get_pulse_times, _find_fpr, plot_abf_data
from eidynamics                     import pattern_index
from eidynamics                     import fit_PSC
from eidynamics                     import utils
from eidynamics.errors              import *


class Neuron:
    """
    All properties and behaviours of a recorded neuron are captured in this class.
    Methods:    Create experiment, update experiment
    Attributes: Neuron.experiment   = a dict holding experiment objects as values and exptType as keys
                Neuron.response     = a pandas dataframe container to hold the table of responses to all experiments
                Neuron.properties   = ePhys properties, derived from analyzing and stats on the response dataframe
                Neuron.animal,
                Neuron.virus
                Neuron.device
    """

    def __init__(self, exptParams):
        # Neuron attributes
        try:
            self.cell_params_parser(exptParams)
        except ParameterMismatchError as err:
            print(err)

        # derived in order: neuron.experiment -> neuron.response -> neuron.properties
        self.experiments        = {}
        self.response           = pd.DataFrame()
        self.properties         = {}

        # self.expectedResponse   = {}
        # self.spotExpected       = {}
        # self.singleSpotDataParsed = False
        # self.spotStimFreq       = 20
        self.spikeTrainSet      = []
        self.data               = {}

    def info(self):
        """
        prints properties of the cell
        """
        print("Cell ID: ", self.cellID)
        print("Experiment Details: ", self.animal)
        _ = self.summarize_experiments()
    
    def cell_params_parser(self, ep):
        """
        Stores the animal related details into Neuron attributes
        from experiment parameter file
        """
        try:
            self.cellID     = int(ep.cellID)
            self.location   = ep.location
            self.animal     = {"animalID": ep.animalID,          "sex": ep.sex,
                               "dateofBirth": ep.dateofBirth,    "dateofInjection": ep.dateofInj,
                               "dateofExpt": ep.dateofExpt}
            self.virus      = {"site": ep.site,                  "injParams": ep.injectionParams,
                               "virus": ep.virus,                "virusTitre": ep.virusTitre,
                               "injVolume": ep.volumeInj,        "ageatInj": ep.ageAtInj,
                               "ageatExpt": ep.ageAtExp,         "incubation": ep.incubation}
            self.device     = {"objMag": ep.objMag,              "polygonFrameSize": ep.frameSize,
                               "polygonGridSize": ep.gridSize,   "polygonSquareSize": ep.squareSize,
                               "DAQ": 'Digidata 1440',           "Amplifier": 'Multiclamp 700B',
                               "cellChannelUnit": ep.unit}
        except Exception as err:
            raise ParameterMismatchError(message=err)

        return self

    def show_data_column_labels(self):
        column_labels = '''
        Field ordering:
             0  cellID
             1  Expt sequence
             2  Age
             3  Age at Injection
             4  Incubation
             5  first pulse time
             6  first pulse peak time
             7  first pulse response
             8  Unit  
             9  datafile index (expt No., last 2 digits, 0-99. For ex. 32 for 2022_04_18_0032_rec.abf)
            10  sweep No.
            11  Stim Freq:    10, 20, 30, 40, 50, 100 Hz
            12  numSquares:   1, 5, 7, 15 sq
            13  intensity:    100 or 50%
            14  pulse width:  2 or 5 ms
            15  meanBaseline: mV
            16  clamp pot:    -70 or 0 mV
            17  CC or VC:     CC = 0, VC = 1
            18  Condition:    Control = 0, Gabazine = 1
            19  AP:           Action Potential flag, yes = 1, No = 0
            20  IR:           MOhm
            21  Tau:          Membrane time constant (ms, for CC) & Ra_effective (MOhm, for VC)
            22  Pattern ID:   refer to pattern ID in pattern index
                  23:37:      coords of spots [23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]
               38:20038:      Frame TTL signal
            20038:40038:      photodiode signal
            40038:60038:      Cell response
            60038:80038:      Field response
        '''
        self.data_column_labels = column_labels
        return info

    def addExperiment(self, datafile, coordfile, exptParams):
        """
        A function that takes filenames and creates an experiment object for a cell object
        """
        newExpt         = Experiment(exptParams, datafile, coordfile)
        newExpt.analyze_experiment(self, exptParams)
        self.updateExperiment(newExpt, self.experiments, exptParams.condition,
                              exptParams.exptType, exptParams.stimFreq, exptParams.EorI)
        return self

    def updateExperiment(self, exptObj, exptDict, condition, exptType, stimFreq, EorI):
        """
        Accommodates all the expriment objects into a dictionary.
        Key   : Expt file name
        Value : [ExptType, Condition, EorI, StimFreq, <Expt Object>]
        """
        exptID = exptObj.dataFile[:15]
        newDict = {exptID: [exptType, condition, EorI, stimFreq, exptObj]}
        exptDict.update(newDict)

        return exptDict

    """
    def to_dataframe(self, method='1sq'):
        # Step 2: Take the expected response dict and make a training set for the whole neuron
        # if len(self.spotExpected)>0:
        for exptID, expt in self.experiments.items():
            if 'FreqSweep' in expt or 'LTMRand' in expt or '1sq20Hz' in expt:
                exptObj = expt[-1]
                print('Adding {} to training set.'.format(exptID))
                self.add_expt_training_set_long(exptObj)

        df = pd.DataFrame(data=self.trainingSetLong)
        df.rename(columns={0:  "exptID",
                           1:  "sweep",
                           2:  "StimFreq",
                           3:  "numSq",
                           4:  "intensity",
                           5:  "pulseWidth",
                           6:  "MeanBaseline",
                           7:  "ClampingPotl",
                           8:  "Clamp",
                           9:  "Condition",
                           10: "AP",
                           11: "InputRes",
                           12: "Tau",
                           13: "patternID"
                           }, inplace=True)
        df = df.astype({"exptID": 'int32', "sweep": "int32", "StimFreq": "int32", "numSq": 'int32'}, errors='ignore')
        df = df.loc[df["StimFreq"] != 0]
        df.replace({'Clamp': {0.0:   'CC', 1.0: 'VC'}}, inplace=True)
        df.replace({'Condition': {0.0: 'CTRL', 1.0: 'GABA'}}, inplace=True)
        total_sweeps = df.shape[0]

        expt_ids = np.unique(df['exptID'])
        expt_idxs = range(len(expt_ids))

        expt_seq = np.array([0] * (total_sweeps))
        cellID   = np.array([self.cellID] * (total_sweeps))
        sigUnit  = np.array([self.device["ephysDataUnit"]] * (total_sweeps))

        age      = np.array([((self.animal["dateofExpt"]      - self.animal["dateofBirth"]).days)] * (total_sweeps))
        ageInj   = np.array([((self.animal["dateofInjection"] - self.animal["dateofBirth"]).days)] * (total_sweeps))
        inc      = np.array([((self.animal["dateofExpt"]      - self.animal["dateofInjection"]).days)] * (total_sweeps))

        # print(df.index)
        # ---------------------
        led = df.iloc[1, 29:20029]
        led = np.where(led >= 0.1 * np.max(led), np.max(led), 0)
        _, peak_props = signal.find_peaks(led, height=0.9 * np.max(led), width=30)
        first_pulse_start = (peak_props['left_ips'][0]) / 2e4
        first_pulse_start_datapoint = int(first_pulse_start * 2e4) + 20029

        fpt = np.array([first_pulse_start] * (total_sweeps))

        # _ss = _signal_sign_cf(df.iloc[:,7], df.iloc[:,8])
        # res_traces = (df.iloc[:, first_pulse_start_datapoint: first_pulse_start_datapoint+1000]).multiply(_ss, axis=0)

        res_traces = (df.iloc[:, first_pulse_start_datapoint: first_pulse_start_datapoint + 1000])
        stimFreq_array  = df.iloc[:, 2]
        clamp_pot_array = df.iloc[:, 7]
        clamp_array     = df.iloc[:, 8]

        peakres, peakres_time = _find_fpr(stimFreq_array, res_traces, clamp_pot_array, clamp_array,)
        peakres_time = peakres_time + first_pulse_start

        # Assemble dataframe
        # print(df_prop.index, peakres.index)
        df_prop = pd.DataFrame(data={"cellID": cellID, "expt_seq": expt_seq, "age": age, "ageInj": ageInj, "incubation": inc,
                               "firstpulsetime": fpt, "firstpulse_peaktime": peakres_time, "firstpeakres": peakres, "Unit": sigUnit})

        df2 = pd.concat([df_prop, df], ignore_index=False, axis=1)

        for i, j in zip(expt_ids, expt_idxs):
            df2.loc[df["exptID"] == i, "expt_seq"] = j

        # dataframe cleanup

        df2.drop_duplicates(inplace=True)
        self.data = df2
        print(self.data_column_labels)

    def add_expt_training_set_long(self, exptObj):

        tracelength    = 20000
        sos = signal.butter(N=2, Wn=1000, fs=2e4, output='sos')

        exptID         = exptObj.dataFile[:15]
        cellData       = exptObj.extract_channelwise_data(exclude_channels=[1, 2, 3, 'Time', 'Cmd'])[0]
        frameData      = exptObj.extract_channelwise_data(exclude_channels=[0, 2, 3, 'Time', 'Cmd'])[1]
        pdData         = exptObj.extract_channelwise_data(exclude_channels=[0, 1, 3, 'Time', 'Cmd'])[2]
        try:
            fieldData  = exptObj.extract_channelwise_data(exclude_channels=[0, 1, 2, 'Time', 'Cmd'])[3]
            fieldData  = signal.sosfiltfilt(sos, fieldData, axis=1)
        except:
            print('Field channel does not exist in the recording.')
            fieldData  = np.zeros((exptObj.numSweeps, tracelength))

        cellData  = signal.sosfiltfilt(sos, cellData,  axis=1)

        inputSet       = np.zeros((exptObj.numSweeps, tracelength + 29))  # photodiode trace
        outputSet1     = np.zeros((exptObj.numSweeps, tracelength))  # sweep Trace
        outputSet2     = np.zeros((exptObj.numSweeps, tracelength))  # fit Trace
        outputSet3     = np.zeros((exptObj.numSweeps, tracelength))  # field Trace
        outputSet4     = np.zeros((exptObj.numSweeps, 8))  # field Trace

        pulseStartTimes = get_pulse_times(exptObj.numPulses, exptObj.stimStart, exptObj.stimFreq)
        Fs = exptObj.Fs

        IR  = IR_calc(exptObj.recordingData, exptObj.clamp, exptObj.IRBaselineEpoch,
                      exptObj.IRsteadystatePeriod, Fs=2e4)[0]
        tau = tau_calc(exptObj.recordingData, exptObj.IRBaselineEpoch, exptObj.IRchargingPeriod,
                       exptObj.IRsteadystatePeriod, clamp=exptObj.clamp, Fs=Fs)[0]

        for sweep in range(exptObj.numSweeps):
            cellTrace  = cellData[sweep, :tracelength]
            pdTrace    = pdData[sweep, :tracelength]
            fieldTrace = fieldData[sweep, :tracelength]
            frameTrace = frameData[sweep, :tracelength]

            pstimes = (Fs * pulseStartTimes).astype(int)
            stimEnd = pstimes[-1] + int(Fs * exptObj.IPI)
            numSquares = len(exptObj.stimCoords[sweep][3:])
            sqSet = exptObj.stimCoords[sweep][3:]
            patternID = pattern_index.get_patternID(sqSet)

            # try:
            #     # fitTrace  = self.expectedResponse[exptID][patternID][5] # replaced fitted trace with trial avg trace
            #     # fitTrace  = self.expectedResponse[exptID][patternID][4]
            #     fitTrace    = self.expectedResponse[exptID][sweep][4]
            # except:
            #     fitTrace = np.zeros(len(pdTrace))

            # deconv_pulse_trend = self.expectedResponse[exptID][sweep][5]

            coordArrayTemp = np.zeros((15))
            coordArrayTemp[:numSquares] = exptObj.stimCoords[sweep][3:]

            if exptObj.clamp == 'VC' and exptObj.EorI == 'E':
                clampPotential = -70
            elif exptObj.clamp == 'VC' and exptObj.EorI == 'I':
                clampPotential = 0
            elif exptObj.clamp == 'CC':
                clampPotential = -70

            Condition = 1 if (exptObj.condition == 'Gabazine') else 0
            clamp = 0 if (exptObj.clamp == 'CC') else 1

            ap    = 0
            if exptObj.clamp == 'CC' and np.max(cellTrace [4460:stimEnd]) > 30:
                ap = 1

            tempArray  = np.array([int(exptObj.dataFile[-12:-8]),
                                   int(sweep + 1),
                                   exptObj.stimFreq,
                                   numSquares,
                                   exptObj.stimIntensity,
                                   exptObj.pulseWidth,
                                   np.round(exptObj.baselineTrend[sweep][0], 1),
                                   clampPotential,
                                   clamp,
                                   Condition,
                                   ap,
                                   np.round(IR[sweep], 1),
                                   np.round(tau[sweep], 3),
                                   patternID])
            tempArray2 = np.concatenate((tempArray, coordArrayTemp))
            inputSet[sweep, :len(tempArray2)] = tempArray2
            inputSet[sweep, len(tempArray2):] = pdTrace
            outputSet1[sweep, :] = frameTrace
            outputSet2[sweep, :] = cellTrace 
            outputSet3[sweep, :] = fieldTrace
            # outputSet4[sweep, :] = deconv_pulse_trend

        newTrainingSet = np.concatenate((inputSet, outputSet1, outputSet2, outputSet3), axis=1)
        try:
            oldTrainingSet = self.trainingSetLong
            self.trainingSetLong = np.concatenate((newTrainingSet, oldTrainingSet), axis=0)
        except AttributeError:
            self.trainingSetLong = newTrainingSet

        return self
    
    
    def generate_expected_traces(self):
                # Step 1: Generate frame expected traces and assign them to self.expectedResponse[exptID]
        if method == '1sq':
            for exptID, expt in self.experiments.items():
                if '1sq20Hz' in expt:
                    _1sqSpotProfile  = self.make_spot_profile(expt[-1])
                    _1sqExpectedDict = {exptID: [expt[1], expt[2], expt[3], _1sqSpotProfile]}
                    self.spotExpected.update(_1sqExpectedDict)
            for exptID, expt in self.experiments.items():
                if expt[0] in ['FreqSweep', 'LTMRand', '1sq20Hz']:
                    c, ei, f = expt[1:4]
                    FreqExptObj = expt[-1]
                    for k, v in self.spotExpected.items():
                        if [c, ei] == v[:2]:
                            spotExpectedDict1sq = v[-1]
                            frameExpectedDict    = self.find_frame_expected(FreqExptObj, spotExpectedDict1sq)
                            self.expectedResponse[exptID] = frameExpectedDict
                elif expt[0] in ['convergence']:
                    c, ei, f = expt[1:4]
                    exptObj  = expt[-1]
                    for k, v in self.spotExpected.items():
                        if [c, ei] == v[:2]:
                            spotExpectedDict1sq = v[-1]
                            frameExpectedDict    = self.find_frame_expected(exptObj, spotExpectedDict1sq)
                            self.expectedResponse[exptID] = frameExpectedDict
                    


        elif method == 'sweep_fit':
            for exptID, expt in self.experiments.items():
                print("Generating fits for experiment {}".format(exptID))
                if 'FreqSweep' in expt or 'LTMRand' in expt or '1sq20Hz' in expt:
                    sweepExpectedDict = self.find_sweep_expected(expt[-1])
                    self.expectedResponse[exptID] = sweepExpectedDict
    
    def make_spot_profile(self, exptObj1sq):
        if not exptObj1sq.exptType == '1sq20Hz':
            raise ParameterMismatchError(message='Experiment object has to be a 1sq experiment')

        Fs                      = exptObj1sq.Fs
        IPI                     = exptObj1sq.IPI  # 0.05 seconds
        numSweeps               = exptObj1sq.numSweeps
        condition               = exptObj1sq.condition
        EorI                    = exptObj1sq.EorI
        stimFreq                = exptObj1sq.stimFreq
        clamp                   = exptObj1sq.clamp

        # Get trial averaged stim and response traces for every spot
        pd                      = exptObj1sq.extract_trial_averaged_data(channels=[2])[2]  # 45 x 40000
        cell                    = exptObj1sq.extract_trial_averaged_data(channels=[0])[0]  # 45 x 40000
        # Get a dict of all spots
        spotCoords              = dict([(k + 1, exptObj1sq.stimCoords[k])
                                       for k in range(0, int(exptObj1sq.numSweeps / exptObj1sq.numRepeats))])

        firstPulseTime          = int(Fs * (exptObj1sq.stimStart))  # 4628 sample points
        secondPulseTime         = int(Fs * (exptObj1sq.stimStart + IPI))  # 5628 sample points

        # Get the synaptic delay from the average responses of all the spots
        avgResponseStartTime, _, _    = PSP_start_time(
            cell, clamp, EorI, stimStartTime=exptObj1sq.stimStart, Fs=Fs)   # 0.2365 seconds
        avgSecondResponseStartTime  = avgResponseStartTime + IPI  # 0.2865 seconds
        avgSynapticDelay            = 0.0055  # avgResponseStartTime-exptObj1sq.stimStart # ~0.0055 seconds
        spotExpectedDict            = {}

        for i in range(len(spotCoords)):
            # spotPD_trialAvg               = pd[i,int(Fs*avgResponseStartTime):int(Fs*avgSecondResponseStartTime)] # 1 x 1000
            spotCell_trialAVG_pulse2pulse = cell[i, firstPulseTime:secondPulseTime + 200]

            t                   = np.linspace(
                0, IPI + 0.01, len(spotCell_trialAVG_pulse2pulse))  # seconds at Fs sampling
            T                   = np.linspace(0, 0.4, int(0.4 * Fs))  # seconds at Fs sampling
            popt, _              = curve_fit(delayed_alpha_function, t, spotCell_trialAVG_pulse2pulse, p0=(
                [0.5, 0.05, 0.005]))  # p0 are initial guesses A=0.5 mV, tau=50ms,delta=5ms
            A, tau, delta         = popt
            # 400 ms = 8000 datapoints long predicted trace from the fit for the spot, not really usable
            fittedSpotRes       = delayed_alpha_function(T, *popt)
            spotExpectedDict[spotCoords[i + 1][3]] = [avgSynapticDelay, A,
                                                      tau, delta, spotCell_trialAVG_pulse2pulse, fittedSpotRes]

        all1sqAvg                     = np.mean(cell[:, firstPulseTime:secondPulseTime + 200], axis=0)
        popt, _                        = curve_fit(delayed_alpha_function, t, all1sqAvg, p0=(
            [0.5, 0.05, 0.005]))  # p0 are initial guesses A=0.5 mV, tau=50ms,delta=5ms
        A, tau, delta                   = popt
        all1sqAvg_fittedSpotRes       = delayed_alpha_function(T, *popt)
        spotExpectedDict['1sqAvg']    = [avgSynapticDelay, A, tau, delta, all1sqAvg, all1sqAvg_fittedSpotRes]

        return spotExpectedDict

    def find_frame_expected(self, exptObj, spotExpectedDict_1sq):
        '''
        generates expected response trace for a sweep by summing up the responses from
        1sq spot that make the N-sq frame corresponding to that sweep
        '''
        print("finding frame expected for: {}".format(exptObj))
        stimFreq        = exptObj.stimFreq
        IPI             = exptObj.IPI  # IPI of current freq sweep experiment
        numPulses       = exptObj.numPulses
        numSweeps       = exptObj.numSweeps
        numRepeats      = exptObj.numRepeats
        cell            = exptObj.extract_trial_averaged_data(
            channels=[0])[0][:, :20000]  # 8 x 40000 #TODO Hardcoded variable: slice length
        stimCoords      = dict([(k, exptObj.stimCoords[k - 1])
                               for k in range(1, 1 + int(numSweeps / numRepeats))])  # {8 key dict}
        stimStart       = exptObj.stimStart
        Fs              = exptObj.Fs
        frameExpected   = {}

        for k, v in stimCoords.items():
            coordsTemp = v[3:]  # nd array of spot coords
            frameID    = pattern_index.get_patternID(coordsTemp)
            numSq      = len(coordsTemp)
            firstPulseExpected = np.zeros((int(Fs * (IPI + 0.01))))
            firstPulseFitted   = np.zeros((int(Fs * (0.4))))  # added 0.01 second = 10 ms to IPI, check line 41,43,68,69

            for spot in coordsTemp:
                spotExpected        = spotExpectedDict_1sq[spot][4]  # raw traces avgd across trials, not fitted
                # all 1sq traces across all spots, all trials, avgd and then fitted
                spotFitted          = spotExpectedDict_1sq['1sqAvg'][5]
                # expected summation of all spot responses making a N-sq frame
                firstPulseExpected  += spotExpected[:len(firstPulseExpected)]
                firstPulseFitted    += spotFitted[:len(firstPulseFitted)]
            avgSynapticDelay    = spotExpectedDict_1sq[spot][0]
            expectedResToPulses = np.zeros(10000 + len(cell[0, :]))
            fittedResToPulses   = np.zeros(10000 + len(cell[0, :]))
            t1 = int(Fs * stimStart)

            for k in range(numPulses):
                t2 = t1 + int(Fs * IPI + avgSynapticDelay)
                T2 = t1 + int(0.4 * Fs)
                window1 = range(t1, t2)
                window2 = range(t1, T2)
                # print(i, frameID, avgSynapticDelay, t1,t2, T2)
                # expected trace given repeated pulses of the same frame
                expectedResToPulses[window1] += firstPulseExpected[:len(window1)]

                fittedResToPulses[window2]   += firstPulseFitted[:len(window2)]
                t1 = t1 + int(Fs * IPI)
            fittedResToPulses   = fittedResToPulses[:len(cell[0, :])]
            expectedResToPulses = expectedResToPulses[:len(cell[0, :])]
            frameExpected[frameID] = [numSq, stimFreq, exptObj.stimIntensity, exptObj.pulseWidth,
                                      expectedResToPulses, fittedResToPulses, firstPulseFitted, firstPulseExpected]

        return frameExpected

    def find_sweep_expected(self, exptObj):
        sweepExpectedDict = {}
        freq = exptObj.stimFreq
        fit_slice = utils.epoch_to_datapoints(exptObj.opticalStimEpoch, Fs=exptObj.Fs)
        for s in range(exptObj.numSweeps):
            print(s)
            time = exptObj.recordingData[s]['Time'][:20000]
            cell0 = exptObj.recordingData[s][0][:20000]
            if exptObj.exptType == '1sq20Hz':
                x = np.zeros((20000))
                y = np.zeros((8))
            else:
                fits = fit_PSC.main(time, cell0, freq, show_plots=False)
                x = fits['dfit']
                y = fits['deconv']
            sweepExpectedDict[s] = ['numSq', freq, exptObj.stimIntensity, exptObj.pulseWidth, x, y]

        return sweepExpectedDict
    """

    def add_cell_to_xl_db(self, excel_file):
        # excel_file = os.path.join(project_path_root,all_cells_response_file)
        try:
            tempDF  = pd.read_excel(excel_file)
        except FileNotFoundError:
            tempDF  = pd.DataFrame()
        outDF       = pd.concat([self.response, tempDF], ignore_index=True)
        # outDF       = pd.concat([cell.response,tempDF],axis=1)
        outDF       = outDF.drop_duplicates()
        outDF.to_excel(excel_file)  # (all_cells_response_file)
        print("Cell experiment data has been added to {}".format(excel_file))

    def save_full_dataset(self, directory):
        for protocol, protocol_data in self.data.items():
            if protocol_data is not None:
                filename = "cell" + str(self.cellID) + '_' + str(protocol) + "_dataset.h5"
                datasetFile = os.path.join(directory, filename)
                protocol_data.to_hdf(datasetFile, format='fixed', key=protocol, mode='w')
                print('Cell Data exported to {}'.format(datasetFile))

    def summarize_experiments(self):
        df = pd.DataFrame(columns=['Polygon Protocol', 'Expt Type', 'Condition', 'Stim Freq',
                          'Stim Intensity', 'Pulse Width', 'Clamp', 'Clamping Potential'])
        for exptID, expt in self.experiments.items():
            df.loc[exptID] = {
                'Polygon Protocol': expt[-1].polygonProtocolFile,
                'Expt Type': expt[-1].exptType,
                'Condition': expt[-1].condition,
                'Stim Freq': expt[-1].stimFreq,
                'Stim Intensity': expt[-1].stimIntensity,
                'Pulse Width': expt[-1].pulseWidth,
                'Clamp': expt[-1].clamp,
                'Clamping Potential': expt[-1].clampPotential
            }
        print(df)
        return df

    def make_dataframe(self):
        '''
        Make a dataframe from an experiment object
        '''

        # List of parameters
        parameters = ['cellID', 'sex','ageAtInj','ageAtExpt','incubation', 'unit',
                    'protocol','exptSeq','exptID','sweep', 'stimFreq', 'numSq', 'intensity',
                    'pulseWidth', 'clampMode', 'clampPotential', 'condition', 'AP', 'IR', 'tau',
                    'numPatterns','patternList', 'sweepBaseline']
        
        numParameters = len(parameters)

        protocol_length = {'FreqSweep': 1.0,
                            '1sq20Hz': 1.0,
                            'LTMRand': 1.0,
                            'SpikeTrain': 1.0,
                            'surprise': 1.0,
                            'convergence': 2.5}
        
        all_experiments_on_cell = list(self.experiments.keys())
        expt_seq = utils.generate_expt_sequence(all_experiments_on_cell)
        print(expt_seq)
        self.data = {
            'FreqSweep': [],
            '1sq20Hz': [],
            'LTMRand': [],
            'SpikeTrain': [],
            'surprise': [],
            'convergence': []
        }
        
        for exptID, expt in self.experiments.items():
            print(f'Adding {exptID} to cell data')
            exptObj  = expt[-1]
            Fs       = exptObj.Fs
            sweep_length = int(protocol_length[exptObj.protocol]*Fs)
            numSweeps = exptObj.numSweeps
            exptID = int(exptObj.dataFile[-12:-8])

            exptdf = pd.DataFrame(columns=parameters)

            exptdf['cellID']        = [self.cellID]             * numSweeps
            exptdf['sex']           = [self.animal["sex"]] * numSweeps
            exptdf['ageAtInj']      = [self.virus["ageatInj"]]  * numSweeps
            exptdf['ageAtExpt']     = [self.virus["ageatExpt"]] * numSweeps
            exptdf['incubation']    = [self.virus["incubation"]]* numSweeps
            exptdf['unit']          = [self.device["cellChannelUnit"]]* numSweeps
            exptdf['exptID']        = [exptID]                  * numSweeps
            exptdf['protocol']      = [exptObj.protocol]        * numSweeps
            exptdf['sweep']         = np.arange(numSweeps)      + 1
            exptdf['stimFreq']      = [exptObj.stimFreq]        * numSweeps
            exptdf['intensity']     = [exptObj.stimIntensity]   * numSweeps
            exptdf['pulseWidth']    = [exptObj.pulseWidth]      * numSweeps
            exptdf['clampMode']     = [exptObj.clamp]           * numSweeps
            exptdf['clampPotential']= [exptObj.clampPotential]  * numSweeps
            exptdf['condition']     = [exptObj.condition]       * numSweeps
            exptdf['sweepBaseline'] = exptObj.baselineTrend[:,0]

            sos     = signal.butter(N=2, Wn=1000, fs=Fs, output='sos')

            # paramData      = np.zeros((exptObj.numSweeps, numParameters))
            cellData       = exptObj.extract_channelwise_data(exclude_channels=[1, 2, 3, 'Time', 'Cmd'])[0]
            exptdf['IR']  =  IR_calc(exptObj.recordingData, exptObj.IRBaselineEpoch, exptObj.IRsteadystatePeriod, clamp=exptObj.clamp, Fs=Fs)[0]
            exptdf['tau'] = tau_calc(exptObj.recordingData, exptObj.IRBaselineEpoch, exptObj.IRchargingPeriod, exptObj.IRsteadystatePeriod, clamp=exptObj.clamp, Fs=Fs)[0]
            exptdf['AP']  = spike_detect(cellData, exptObj.opticalStimEpoch, clampingPotential=exptObj.clampPotential, clamp=exptObj.clamp, Fs=Fs)[0]

            cellData       = exptObj.extract_channelwise_data(exclude_channels=[1, 2, 3, 'Time', 'Cmd'])[0][:,:sweep_length]
            frameData      = exptObj.extract_channelwise_data(exclude_channels=[0, 2, 3, 'Time', 'Cmd'])[1][:,:sweep_length]
            pdData         = exptObj.extract_channelwise_data(exclude_channels=[0, 1, 3, 'Time', 'Cmd'])[2][:,:sweep_length]
            try:
                fieldData  = exptObj.extract_channelwise_data(exclude_channels=[0, 1, 2, 'Time', 'Cmd'])[3][:,:sweep_length]
                fieldData  = signal.sosfiltfilt(sos, fieldData, axis=1)
            except:
                print('Field channel does not exist in the recording. Writing zeros.')
                fieldData  = np.zeros((exptObj.numSweeps, sweep_length))
            cellData  = signal.sosfiltfilt(sos, cellData,  axis=1)
            fieldData = signal.sosfiltfilt(sos, fieldData, axis=1)

            datadf = pd.DataFrame(data=np.concatenate((cellData, frameData, pdData, fieldData), axis=1))        
            
            
            df = pd.DataFrame(columns=['patternList', 'numPatterns', 'numSq'])
            for i in range(numSweeps):
                x = np.array(exptObj.sweepwiseCoords[i][0], dtype='object')
                y = len(x)
                df.loc[i,'patternList'] = x
                df.loc[i,'numPatterns'] = y
                df.loc[i,'numSq']       = exptObj.sweepwiseCoords[i][1]
            exptdf['patternList'] = df['patternList']
            exptdf['numPatterns'] = df['numPatterns']
            exptdf['numSq']       = df['numSq']

            exptdf = pd.concat([exptdf, datadf], axis=1)

            self.data[exptObj.protocol].append(exptdf)

        # concatenate all dataframes within each protocol
        for protocol, protocol_data in self.data.items():
            if len(protocol_data) > 0:
                self.data[protocol] = pd.concat(protocol_data, ignore_index=True)
                self.data[protocol]['exptSeq'] = self.data[protocol]['exptID'].map(expt_seq)
            else:
                self.data[protocol] = None

        return self.data
    
    @staticmethod
    def saveCell(neuronObj, filename):
        directory       = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filename, 'wb') as fout:
            print("Neuron object saved into pickle. Use loadCell to load back.")
            pickle.dump(neuronObj, fout, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def loadCell(filename):
        try:
            with open(filename, 'rb') as fin:
                print("Cell object loaded from file")
                return pickle.load(fin)
        except FileNotFoundError:
            print("File not found.")
            raise Exception

    @staticmethod
    def load_dataframe(hdf_file):
        try:
            dataframe = pd.read_hdf(hdf_file)
        except FileNotFoundError:
            print("File not found.")
            raise Exception
        return dataframe

    def __iter__(self):
        return self.experiments.iteritems()

    """
    def make_dataframe(self):
        '''
        Make a pandas dataframe from all the recording data.
        '''

        protocol_length = {'FreqSweep': 20000,
                           '1sq20Hz': 20000,
                           'LTMRand': 20000,
                           'spikeTrain': 20000,
                           'surprise': 20000,
                           'convergence': 50000}

        for exptID, expt in self.experiments.items():
            print('Adding {} to cell data.'.format(exptID))
            exptObj  = expt[-1]
            protocol = expt[0]

            df_exptdata   = pd.DataFrame(np.zeros((1,29+4*protocol_length[protocol])) )
            df_exptdata.insert(1, 'Protocol', 'none', allow_duplicates=False)
            df_exptdata['Protocol'] = df_exptdata['Protocol'].astype(str)

            df_exptdata   = self._add_exptdata_to_dataframe(exptObj, df_exptdata, protocol=protocol)

            if df_exptdata.shape[0] > 1:
                self.data[protocol] = self._add_cell_params_to_dataframe(df_exptdata)
            else:
                self.data[protocol] = None

    
    def _add_exptdata_to_dataframe(self, exptObj, df, protocol=''):
        numParameters = 29
        exptID  = exptObj.dataFile[:15]
        Fs      = exptObj.Fs
        sos     = signal.butter(N=2, Wn=1000, fs=Fs, output='sos')

        tracelength = df.shape[1] - numParameters

        paramData      = np.zeros((exptObj.numSweeps, numParameters))
        cellData       = exptObj.extract_channelwise_data(exclude_channels=[1, 2, 3, 'Time', 'Cmd'])[0][:,:tracelength]
        frameData      = exptObj.extract_channelwise_data(exclude_channels=[0, 2, 3, 'Time', 'Cmd'])[1][:,:tracelength]
        pdData         = exptObj.extract_channelwise_data(exclude_channels=[0, 1, 3, 'Time', 'Cmd'])[2][:,:tracelength]
        try:
            fieldData  = exptObj.extract_channelwise_data(exclude_channels=[0, 1, 2, 'Time', 'Cmd'])[3][:,:tracelength]
            fieldData  = signal.sosfiltfilt(sos, fieldData, axis=1)
        except:
            print('Field channel does not exist in the recording. Writing zeros.')
            fieldData  = np.zeros((exptObj.numSweeps, tracelength))
        cellData  = signal.sosfiltfilt(sos, cellData,  axis=1)
        fieldData = signal.sosfiltfilt(sos, fieldData, axis=1)

        IR  = IR_calc(exptObj.recordingData,  exptObj.IRBaselineEpoch, exptObj.IRsteadystatePeriod, clamp=exptObj.clamp, Fs=Fs)[0]
        tau = tau_calc(exptObj.recordingData, exptObj.IRBaselineEpoch, exptObj.IRchargingPeriod, exptObj.IRsteadystatePeriod, clamp=exptObj.clamp, Fs=Fs)[0]

        pulse_start_times = get_pulse_times(exptObj.numPulses, exptObj.stimStart, exptObj.stimFreq)
        pulse_start_times = (Fs * pulse_start_times).astype(int)
        
        IPI     = int(Fs * exptObj.IPI)
        for sweep in range(exptObj.numSweeps):
            cellTrace         = cellData[sweep, :tracelength]
            frameTTL          = frameData[sweep, :tracelength]
            # numFrameChange    = np.sum(np.diff(frameTTL) > 0)
            pulse_end_times   = pulse_start_times[-1] + IPI
            sqSet             = exptObj.stimCoords[sweep][3:]
            numSquares        = len(sqSet)
            patternID         = pattern_index.get_patternID(sqSet)

            pstimes = (pulse_start_times).astype(int)
            stimEnd = pstimes[-1] + IPI

            coord_array = np.zeros((15))
            coord_array[:numSquares] = sqSet

            
            clampPotential = 0 if (exptObj.clamp == 'VC' and exptObj.EorI == 'I') else -70
                

            Condition   = 1 if ( exptObj.condition == 'Gabazine' )                            else 0
            clamp       = 0 if ( exptObj.clamp == 'CC' )                                      else 1
            ap          = 1 if ( exptObj.clamp == 'CC' and np.max(cellTrace[pstimes[0]:stimEnd])>30 ) else 0


            parameters  = np.array([int(exptObj.dataFile[-12:-8]),
                                    int(sweep + 1), # true non-pythonic sweep number
                                    exptObj.stimFreq,
                                    numSquares,
                                    exptObj.stimIntensity,
                                    exptObj.pulseWidth,
                                    np.round(exptObj.baselineTrend[sweep][0], 1),
                                    clampPotential,
                                    clamp,
                                    Condition,
                                    ap,
                                    np.round(IR[sweep], 1),
                                    np.round(tau[sweep], 3),
                                    patternID])
            
            parameters_with_coords = np.concatenate((parameters, coord_array))
            paramData[sweep, :] = parameters_with_coords

        data      = np.concatenate((paramData, pdData, frameData, cellData, fieldData), axis=1)
        df2       = pd.DataFrame(data=data)
        df2.insert(1, 'Protocol', [exptObj.exptType]*exptObj.numSweeps, allow_duplicates=False)
        df2['Protocol'] = df2['Protocol'].astype(str)

        old_dataframe = df
        dataframe = pd.concat((df, df2), axis=0)

        return dataframe

    def _add_cell_params_to_dataframe(self, df):  
        Fs = 2e4

        df.rename(columns={0:  "exptID",
                            1:  "sweep",
                            2:  "StimFreq",
                            3:  "numSq",
                            4:  "intensity",
                            5:  "pulseWidth",
                            6:  "MeanBaseline",
                            7:  "ClampingPotl",
                            8:  "Clamp",
                            9:  "Condition",
                            10: "AP",
                            11: "InputRes",
                            12: "Tau",
                            13: "patternID"
                            }, inplace=True)
        df = df.astype({"exptID": 'int32', "sweep": "int32", "StimFreq": "int32", "numSq": 'int32'}, errors='ignore')
        df.replace({'Clamp': {0.0:   'CC', 1.0: 'VC'}}, inplace=True)
        df.replace({'Condition': {0.0: 'CTRL', 1.0: 'GABA'}}, inplace=True)

        df = df[df['exptID'] != 0]
        total_sweeps = df.shape[0]

        expt_ids = np.unique(df['exptID'])
        expt_idxs= range(len(expt_ids))

        expt_seq = np.array([0] * (total_sweeps))
        cellID   = np.array([self.cellID] * (total_sweeps))
        sigUnit  = np.array([self.device["ephysDataUnit"]] * (total_sweeps))

        age      = np.array([((self.animal["dateofExpt"]      - self.animal["dateofBirth"]).days)] * (total_sweeps))
        ageInj   = np.array([((self.animal["dateofInjection"] - self.animal["dateofBirth"]).days)] * (total_sweeps))
        inc      = np.array([((self.animal["dateofExpt"]      - self.animal["dateofInjection"]).days)] * (total_sweeps))

        led = df.iloc[1, 30:20030]
        led = np.where(led >= 0.1 * np.max(led), np.max(led), 0)
        _, peak_props = signal.find_peaks(led, height=0.9 * np.max(led), width=30)
        first_pulse_start = (peak_props['left_ips'][0]) / Fs
        first_pulse_start_datapoint = int(first_pulse_start * Fs) + 20030

        fpt = np.array([first_pulse_start] * (total_sweeps))

        res_traces = (df.iloc[:, first_pulse_start_datapoint: first_pulse_start_datapoint + 1000])
        stimFreq_array  = df.iloc[:, 3]
        clamp_pot_array = df.iloc[:, 8]
        clamp_array     = df.iloc[:, 9]

        peakres, peakres_time = _find_fpr(stimFreq_array, res_traces, clamp_pot_array, clamp_array,)
        peakres_time = peakres_time + first_pulse_start

        # Assemble dataframe
        df_prop = pd.DataFrame(data={"cellID": cellID, "expt_seq": expt_seq, "age": age, "ageInj": ageInj, "incubation": inc,
                                "firstpulsetime": fpt, "firstpulse_peaktime": peakres_time, "firstpeakres": peakres, "Unit": sigUnit})
        df_prop.reset_index(inplace=True, drop=True)
        df.reset_index(inplace=True, drop=True)

        df2 = pd.concat([df_prop, df], axis=1)
        for i, j in zip(expt_ids, expt_idxs):
            df2.loc[df["exptID"] == i, "expt_seq"] = j
        
        # dataframe cleanup
        df2.drop_duplicates(inplace=True)

        return df2
    """
    
class Experiment:
    '''All different kinds of experiments conducted on a patched
    neuron are captured by this superclass.'''

    def __init__(self, exptParams, datafile, coordfile=None):
        try:
            self.exptParamsParser(exptParams)
        except ParameterMismatchError as err:
            print(err)

        self.Flags          = {"IRFlag": False, "APFlag": False, "NoisyBaselineFlag": False, "TauChangeFlag": False}
        datafile            = os.path.abspath(datafile)
        data                = abf_to_data.abf_to_data(datafile,
                                                      baseline_criterion=exptParams.baselineCriterion,
                                                      sweep_baseline_epoch=exptParams.sweepBaselineEpoch,
                                                      baseline_subtraction=exptParams.baselineSubtraction,
                                                      signal_scaling=exptParams.signalScaling,
                                                      sampling_freq=exptParams.Fs,
                                                      filter_type=exptParams.filter,
                                                      filter_cutoff=exptParams.filterHighCutoff,
                                                      plot_data=False)

        self.recordingData  = data[0]
        self.baselineTrend  = data[1]
        self.meanBaseline   = data[2]
        self.Flags["NoisyBaselineFlag"] = data[3]
        del data

        self.coordfile          = coordfile
        if coordfile:
            self.opticalStim    = Coords(coordfile, repeats=exptParams.repeats)
            self.stimCoords     = self.opticalStim.coords
            self.sweepwiseCoords= self.opticalStim.sweepwise_patterns
        else:
            self.opticalStim    = ''
            self.stimCoords     = ''
            self.sweepwiseCoords= ''

        self.numSweeps      = len(self.recordingData.keys())
        self.sweepIndex     = 0  # start of the iterator over sweeps

        # ### When Sweep class is implemented
        # self.sweep          = {}

        # f0                  = 0 # f for frames
        # frame_increament_per_sweep = len(signal.find_peaks(self.recordingData[0][1], height=1, distance=100)[0])
        # f1                  = frame_increament_per_sweep

        # for s in range(self.numSweeps):
        #     coord_file_frames = self.stimCoords[f0:f1]
        #     self.sweep[s] = Sweep(self.recordingData[s], s, coord_file_frames, exptParams)
        #     f0 = f1
        #     f1 = f1 + frame_increament_per_sweep

    def __iter__(self):
        return self

    def __next__(self):
        if self.sweepIndex >= self.numSweeps:
            raise StopIteration
        currentSweepIndex   = self.sweepIndex
        self.sweepIndex     += 1
        return self.recordingData[currentSweepIndex]

    def extract_channelwise_data(self, exclude_channels=[]):
        '''
        Returns a dictionary holding channels as keys,
        and sweeps as keys in an nxm 2-d array format where n is number of sweeps
        and m is number of datapoints in the recording per sweep.
        '''
        sweepwise_dict    = self.recordingData
        chLabels          = list(sweepwise_dict[0].keys())
        numSweeps         = len(sweepwise_dict)
        sweepLength       = len(sweepwise_dict[0][chLabels[0]])
        channelDict       = {}
        tempChannelData   = np.zeros((numSweeps, sweepLength))

        included_channels = list(set(chLabels) - set(exclude_channels))

        channelDict       = {}
        tempChannelData   = np.zeros((numSweeps, sweepLength))
        for ch in included_channels:
            for i in range(numSweeps):
                tempChannelData[i, :] = sweepwise_dict[i][ch]
            channelDict[ch] = tempChannelData
            tempChannelData = 0.0 * tempChannelData
        return channelDict

    def extract_trial_averaged_data(self, channels=[0]):
        '''
        Returns a dictionary holding channels as keys,
        and trial averaged sweeps as an nxm 2-d array where n is number of patterns
        and m is number of datapoints in the recording per sweep.
        '''
        chData = self.extract_channelwise_data(exclude_channels=[])
        chMean = {}
        for ch in channels:
            chData_temp = np.reshape(chData[ch], (self.numRepeats, int(self.numSweeps / self.numRepeats), -1))
            chMean[ch] = np.mean(chData_temp, axis=0)

        return chMean

    def analyze_experiment(self, neuron, exptParams): 

        if self.exptType == 'sealTest':
            # Call a function to calculate access resistance from recording
            return self.sealTest()
        elif self.exptType == 'IR':
            # Call a function to calculate cell input resistance from recording
            return self.inputRes(self, neuron)
        elif self.exptType in ['1sq20Hz', 'FreqSweep']:
            # Call a function to analyze the freq dependent response
            return self.FreqResponse(neuron, exptParams)
        elif self.exptType in ['LTMRand', 'LTMSeq']:
            # Call a function to analyze the freq dependent response
            return self.LTMResponse(neuron, exptParams)
        elif self.exptType in ['convergence']:
            # Call a function to analyze the freq dependent response
            return self.convergenceResponse(neuron, exptParams)
        elif self.exptType in ['SpikeTrain']:
            # Call a function to analyze the freq dependent response
            return self.SpikeTrainResponse(neuron, exptParams)

    def sealTest(self):
        # calculate access resistance from data, currently not implemented
        return self

    def inputRes(self, neuron):
        # calculate input resistance from data
        neuron.properties.update({'IR': np.mean(IR_calc(self.recordingData, np.arange[1, 200], np.arange[500, 700]))})
        return self

    # FIXME: improve feature (do away with so many nested functions)
    def FreqResponse(self, neuron, exptParams):
        # there can be multiple kinds of freq based experiments and their responses.
        # return expt2df(self, neuron, exptParams)  # this function takes expt and converts to a dataframe of responses
        return None

    def convergenceResponse(self, neuron, exptParams):
        '''code to integrate the convergence experiments in the pandas dataframe'''
        return None #expt2df(self, neuron, exptParams)

    def LTMResponse(self, neuron, exptParams):
        '''code to integrate the convergence experiments in the pandas dataframe'''
        return None

    def SpikeTrainResponse(self, neuron, exptParams):
        # Analysis of cell response to a invivo like poisson spike train
        return spiketrain_analysis(self, neuron, exptParams)

    def exptParamsParser(self, ep):
        try:
            self.dataFile           = ep.datafile
            self.cellID             = ep.cellID
            self.stimIntensity      = ep.intensity
            self.stimFreq           = ep.stimFreq
            self.pulseWidth         = ep.pulseWidth
            self.bathTemp           = ep.bathTemp
            self.location           = ep.location
            self.clamp              = ep.clamp
            self.EorI               = ep.EorI
            self.clampPotential     = ep.clampPotential
            self.polygonProtocolFile = ep.polygonProtocol
            self.numRepeats         = ep.repeats
            self.numPulses          = ep.numPulses
            self.IPI                = 1 / ep.stimFreq
            self.stimStart          = ep.opticalStimEpoch[0]
            self.Fs                 = ep.Fs
            self.exptType           = ep.exptType
            self.protocol           = ep.exptType
            self.condition          = ep.condition

            self.sweepDuration      = ep.sweepDuration
            self.MeanBaselineEpoch = ep.sweepBaselineEpoch
            self.opticalStimEpoch   = ep.opticalStimEpoch

            try:
                self.probePulseEpoch    = ep.singlePulseEpoch
                self.pulseTrainEpoch    = ep.pulseTrainEpoch
            except:
                self.probePulseEpoch    = [ep.opticalStimEpoch[0], ep.opticalStimEpoch[0] + self.IPI]
                self.pulseTrainEpoch    = ep.opticalStimEpoch

            self.IRBaselineEpoch    = ep.IRBaselineEpoch
            self.IRpulseEpoch       = ep.IRpulseEpoch
            self.IRchargingPeriod   = ep.IRchargingPeriod
            self.IRsteadystatePeriod = ep.IRsteadystatePeriod
            self.interSweepInterval = ep.interSweepInterval

            self.unit = 'pA' if self.clamp == 'VC' else 'mV' if self.clamp == 'CC' else 'a.u.'

            # merging 'FreqSweep' and '1sq20Hz' protocol labels
            if self.exptType == '1sq20Hz':
                self.exptType = 'FreqSweep'
                self.protocol = 'FreqSweep'

        except Exception as err:
            raise ParameterMismatchError(message=err)

        return self


class Coords:
    '''
    An object that stores Sweep wise record of coordinates of all the square points
    illuminated in the experiment.
    currently class "Coords" is not being used except in generating
    a dict containing sweep wise coords
    '''
    def __init__(self, coordFile, repeats=3):
        self.coordfile      = coordFile.stem
        self.gridSize       = []
        self.coords         = self.coordParser(coordFile)
        self.frames_per_sweep = pattern_index.polygon_protocol_sweep_division(self.coordfile)
        self.repeats        = repeats
        self.numSweeps      = repeats * int(len(self.coords) / self.frames_per_sweep)
        self.sweepwise_patterns = self.get_sweepwise_patterns()

    def coordParser(self, coordFile):
        coords              = []
        import csv
        with open(coordFile, 'r') as cf:
            c               = csv.reader(cf, delimiter=" ")
            for lines in c:
                intline     = []
                for i in lines:
                    intline.append(int(i))
                frameID     = intline[0]
                coords.append(intline)
        self.gridSize       = [intline[1], intline[2]]
        return coords

    def get_sweepwise_patterns(self):
        sweepdict = {}
        start = 0
        for s in range(self.numSweeps):
            end = start + self.frames_per_sweep
            c = []
            for i in range(start,end):
                c.append(pattern_index.get_patternID(self.coords[i][3:]))
            if start + self.frames_per_sweep >= len(self.coords):
                start = 0
            else:
                start = start + self.frames_per_sweep
            sweepdict[s] = [c, len(self.coords[i][3:])]
        return sweepdict


    def __iter__(self):
        return self

    def __next__(self):
        self.sweepIndex = 0  # start of the iterator over sweeps
        if self.sweepIndex >= self.numSweeps:
            raise StopIteration
        currentSweepIndex   = self.sweepIndex
        self.sweepIndex += 1
        return self.coords[currentSweepIndex]


# TODO
class Sweep:
    """
    Experimental Class: actual experimental data.
    """

    def __init__(self, sweep_data, sweep_number, sweep_coords, expt_params):
        self.unit                               = expt_params.unit
        self.EorI                               = expt_params.EorI
        self.clamp_potential                    = expt_params.clampPotential
        self.light_intensity                    = expt_params.intensity
        self.light_pulse_width                  = expt_params.pulseWidth
        self.freq                               = expt_params.stimFreq
        self.clamp                              = expt_params.clamp
        self.baseline_epoch                     = expt_params.sweepBaselineEpoch
        self.IRBaselineEpoch                    = expt_params.IRBaselineEpoch
        self.IRchargingPeriod                   = expt_params.IRchargingPeriod
        self.optical_stim_epoch                 = [0, expt_params.opticalStimEpoch[1]]
        self.IRsteadystatePeriod                = expt_params.IRsteadystatePeriod

        self.numChannels                        = len(sweep_data)
        self.Cmd                                = sweep_data['Cmd']
        self.time                               = sweep_data["Time"]
        self.cell,      self.baseline           = abf_to_data.baseline_subtractor(sweep_data[0],
                                                                                  sweep_baseline_epoch=expt_params.sweepBaselineEpoch,
                                                                                  sampling_freq=expt_params.Fs,
                                                                                  subtract_baseline=True)
        self.frameTTL,  _                       = abf_to_data.baseline_subtractor(sweep_data[1],
                                                                                  sweep_baseline_epoch=expt_params.sweepBaselineEpoch,
                                                                                  sampling_freq=expt_params.Fs,
                                                                                  subtract_baseline=True)
        self.light_stim, _                       = abf_to_data.baseline_subtractor(sweep_data[2],
                                                                                   sweep_baseline_epoch=expt_params.sweepBaselineEpoch,
                                                                                   sampling_freq=expt_params.Fs,
                                                                                   subtract_baseline=True)

        if 3 in sweep_data:
            self.field, _                       = abf_to_data.baseline_subtractor(sweep_data[3],
                                                                                  sweep_baseline_epoch=expt_params.sweepBaselineEpoch,
                                                                                  sampling_freq=expt_params.Fs,
                                                                                  subtract_baseline=True)
        else:
            self.field                          = None

        self.light_frames                       = [pattern_index.get_patternID(x[3:]) for x in sweep_coords]

        self.sweepID                            = 0

    def plot_sweep(self):
        plot_abf_data(
            {"Cmd": self.cmd,
             "Cell1": self.cell,
             "frameTTL": self.frameTTL,
             "Light": self.light_stim,
             "Field": self.field,
             "Time": self.time,
             },
            label=str(self.sweepID)
        )

    def fit(self, show_plots=False):
        fit_slice = utils.epoch_to_datapoints(self.optical_stim_epoch)
        self.fits = fit_PSC.main(self.time[fit_slice], self.cell, self.freq, show_plots=show_plots)

    def get_sweep_params(self):
        self.IR   = IR_calc(self.cell, self.clamp, self.IRBaselineEpoch, self.IRsteadystatePeriod, )
        self.tau  = tau_calc(self.cell, self.IRBaselineEpoch, self.IRchargingPeriod,
                             self.IRsteadystatePeriod, clamp=self.clamp, )
        print("IR = {}, Tau = {}".format(self.IR, self.tau))

        return self.IR, self.tau
