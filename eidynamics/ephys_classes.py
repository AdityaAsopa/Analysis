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
from eidynamics.abf_to_data         import abf_to_data
from eidynamics.expt_to_dataframe   import expt2df
from eidynamics.spiketrain          import spiketrain_analysis
from eidynamics.ephys_functions     import IR_calc, tau_calc
from eidynamics.utils               import filter_data, delayed_alpha_function, PSP_start_time, get_pulse_times, _find_fpr
from eidynamics                     import pattern_index
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
        self.expectedResponse   = {}
        self.spotExpected       = {}
        self.singleSpotDataParsed= False
        self.spotStimFreq       = 20
        self.trainingSetLong    = np.zeros((1,80029))
        self.spikeTrainSet      = []

    def cell_params_parser(self,ep):
        """
        Stores the animal related details into Neuron attributes
        from experiment parameter file
        """
        try:
            self.cellID     = int(ep.cellID)
            self.location   = ep.location
            self.animal     = {"animalID":ep.animalID,          "sex":ep.sex,
                               "dateofBirth":ep.dateofBirth,    "dateofInjection":ep.dateofInj,
                               "dateofExpt":ep.dateofExpt}
            self.virus      = {"site":ep.site,                  "injParams":ep.injectionParams,
                               "virus":ep.virus,                "virusTitre":ep.virusTitre,
                               "injVolume":ep.volumeInj,        "ageatInj":ep.ageAtInj,
                               "ageatExpt":ep.ageAtExp,         "incubation":ep.incubation}
            self.device     = {"objMag":ep.objMag,              "polygonFrameSize":ep.frameSize,
                               "polygonGridSize":ep.gridSize,   "polygonSquareSize":ep.squareSize,
                               "DAQ":'Digidata 1440',           "Amplifier":'Multiclamp 700B',
                               "ephysDataUnit": ep.unit}
        except Exception as err:
            raise ParameterMismatchError(message=err)

        return self

    def addExperiment(self,datafile,coordfile,exptParams):
        """
        A function that takes filenames and creates an experiment object for a cell object
        """
        newExpt         = Experiment(exptParams,datafile,coordfile)
        newExpt.analyze_experiment(self,exptParams)
        self.updateExperiment(newExpt,self.experiments,exptParams.condition,exptParams.exptType,exptParams.stimFreq,exptParams.EorI)
        return self

    def updateExperiment(self,exptObj,exptDict,condition,exptType,stimFreq,EorI):
        """
        Accommodates all the expriment objects into a dictionary.
        Key   : Expt file name
        Value : [ExptType, Condition, EorI, StimFreq, <Expt Object>]
        """
        exptID = exptObj.dataFile[:15]
        newDict = {exptID: [exptType, condition, EorI, stimFreq, exptObj]}
        exptDict.update(newDict)

        return exptDict

    def generate_expected_traces(self):
        for exptID,expt in self.experiments.items():
            if '1sq20Hz' in expt:
                _1sqSpotProfile  = self.make_spot_profile(expt[-1])
                _1sqExpectedDict = {exptID:[expt[1], expt[2], expt[3], _1sqSpotProfile]}
                self.spotExpected.update(_1sqExpectedDict)
        for exptID,expt in self.experiments.items():
            if 'FreqSweep' in expt or 'LTMRand' in expt or '1sq20Hz' in expt:
                c,ei,f = expt[1:4]
                FreqExptObj = expt[-1]
                print(exptID)
                for k,v in self.spotExpected.items():
                    if [c,ei] == v[:2]:
                        spotExpectedDict1sq = v[-1]
                        frameExpectedDict    = self.find_frame_expected(FreqExptObj,spotExpectedDict1sq)
                        self.expectedResponse[exptID] = frameExpectedDict
        
        # if len(self.spotExpected)>0:
        for exptID,expt in self.experiments.items():
            if 'FreqSweep' in expt or 'LTMRand' in expt or '1sq20Hz' in expt:
                exptObj = expt[-1]
                print('Adding {} to training set.'.format(exptID))
                self.add_expt_training_set_long(exptObj)

        
        df = pd.DataFrame(data=self.trainingSetLong)        
        df.rename( columns = {0:  "exptID", 
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
                              }, inplace = True )
        df = df.astype({"exptID": 'int32', "sweep":"int32", "StimFreq": "int32", "numSq": 'int32'}, errors='ignore')
        df = df.loc[df["StimFreq"] != 0]
        df.replace({'Clamp'     : { 0.0 :   'CC', 1.0 : 'VC'  }}, inplace=True)
        df.replace({'Condition' : { 0.0 : 'CTRL', 1.0 : 'GABA'}}, inplace=True)
        total_sweeps = df.shape[0]

        expt_ids = np.unique(df['exptID'])
        expt_idxs= range(len(expt_ids))

        expt_seq = np.array( [ 0                            ]*(total_sweeps) )
        cellID   = np.array( [ self.cellID                  ]*(total_sweeps) )
        sigUnit  = np.array( [ self.device["ephysDataUnit"] ]*(total_sweeps) )
        
        age      = np.array( [ ((self.animal["dateofExpt"]      - self.animal["dateofBirth"]     ).days)]*(total_sweeps) )
        ageInj   = np.array( [ ((self.animal["dateofInjection"] - self.animal["dateofBirth"]     ).days)]*(total_sweeps) )
        inc      = np.array( [ ((self.animal["dateofExpt"]      - self.animal["dateofInjection"] ).days)]*(total_sweeps) )

        
        
        # print(df.index)
        #---------------------
        led = df.iloc[1,29:20029]
        led = np.where(led>=0.1*np.max(led), np.max(led), 0)
        _,peak_props = signal.find_peaks(led, height=0.9*np.max(led), width=30)
        first_pulse_start = (peak_props['left_ips'][0]) / 2e4
        first_pulse_start_datapoint = int(first_pulse_start*2e4) + 20029


        fpt = np.array([first_pulse_start]*(total_sweeps))

        # _ss = _signal_sign_cf(df.iloc[:,7], df.iloc[:,8])
        # res_traces = (df.iloc[:, first_pulse_start_datapoint: first_pulse_start_datapoint+1000]).multiply(_ss, axis=0)

        res_traces = (df.iloc[:, first_pulse_start_datapoint: first_pulse_start_datapoint+1000])
        stimFreq_array  = df.iloc[:,2]
        clamp_pot_array = df.iloc[:,7]
        clamp_array     = df.iloc[:,8]

        peakres, peakres_time = _find_fpr(stimFreq_array, res_traces)
        # peakres, peakres_time = _find_fpr(clamp_pot_array, clamp_array, stimFreq_array, res_traces)
        peakres_time = peakres_time + first_pulse_start

        # Assemble dataframe
        # print(df_prop.index, peakres.index)
        df_prop = pd.DataFrame(data={"expt_seq":expt_seq, "cellID":cellID, "age":age, "ageInj":ageInj, "incubation":inc, "firstpulsetime":fpt, "firstpulse_peaktime": peakres_time, "firstpeakres":peakres, "Unit": sigUnit})

        df2 = pd.concat([df_prop, df], ignore_index=False, axis=1)
        
        for i,j in zip(expt_ids, expt_idxs):
            df2.loc[df["exptID"] == i, "expt_seq"] = j

        # dataframe cleanup
        
        df2.drop_duplicates(inplace=True)
        self.data = df2
        
    def add_expt_training_set_long(self, exptObj):
        '''
        # Field ordering:
            # 0  datafile index (expt No., last 2 digits, 0-99. For ex. 32 for 2022_04_18_0032_rec.abf)
            # 1  sweep No.
            # 2  Stim Freq : 10, 20, 30, 40, 50, 100 Hz
            # 3  numSquares : 1, 5, 7, 15 sq
            # 4  intensity : 100 or 50%
            # 5  pulse width : 2 or 5 ms
            # 6  meanBaseline : mV
            # 7  clamp potential: -70 or 0 mV
            # 8  CC or VC : CC = 0, VC = 1
            # 9  Gabazine : Control = 0, Gabazine = 1
            # 10 IR : MOhm
            # 11 Tau : membrane time constant (ms, for CC) & Ra_effective (MOhm, for VC)
            # 12 pattern ID : refer to pattern ID in pattern index
            # 13:28 coords of spots [12,13,14,15,16, 17,18,19,20,21, 22,23,24,25,26]
            # 28 AP : 1 if yes, 0 if no
            # 29:20029 Sample points for LED
            # 20029:40029 Sample points for ephys recording.
            # 40029:60029 1sq based Expected response
            # 60029:80029 Field response
        '''
        tracelength    = 20000
        
        exptID         = exptObj.dataFile[:15]
        cellData       = exptObj.extract_channelwise_data(exclude_channels=[1,2,3,'Time','Cmd'])[0]
        pdData         = exptObj.extract_channelwise_data(exclude_channels=[0,1,3,'Time','Cmd'])[2]
        try:
            fieldData  = exptObj.extract_channelwise_data(exclude_channels=[0,1,2,'Time','Cmd'])[3]
        except:
            print('Field channel does not exist in the recording.')
            fieldData  = np.zeros((exptObj.numSweeps,tracelength))
        
        sos = signal.butter(N=2, Wn=500, fs=2e4, output='sos')
        cellData  = signal.sosfiltfilt(sos, cellData,  axis=1)
        fieldData = signal.sosfiltfilt(sos, fieldData, axis=1)

        inputSet       = np.zeros((exptObj.numSweeps,tracelength+29)) # photodiode trace
        outputSet1     = np.zeros((exptObj.numSweeps,tracelength)) # sweep Trace
        outputSet2     = np.zeros((exptObj.numSweeps,tracelength)) # fit Trace
        outputSet3     = np.zeros((exptObj.numSweeps,tracelength)) # field Trace
        pulseStartTimes= get_pulse_times(exptObj.numPulses,exptObj.stimStart,exptObj.stimFreq)
        Fs = exptObj.Fs

        IR  =  IR_calc(exptObj.recordingData, exptObj.clamp, exptObj.IRBaselineEpoch, exptObj.IRsteadystatePeriod, Fs=2e4)[0]
        tau = tau_calc(exptObj.recordingData, exptObj.IRBaselineEpoch, exptObj.IRchargingPeriod, exptObj.IRsteadystatePeriod, clamp=exptObj.clamp, Fs=Fs)[0]

        for sweep in range(exptObj.numSweeps):
            sweepTrace = cellData[sweep,:tracelength]
            pdTrace    = pdData[sweep,:tracelength]
            fieldTrace = fieldData[sweep,:tracelength]
            # pdTrace    = np.zeros(len(pdTrace))
            
            pstimes = (Fs*pulseStartTimes).astype(int)
            stimEnd = pstimes[-1]+int(Fs*exptObj.IPI)
            numSquares = len(exptObj.stimCoords[sweep+1])
            sqSet = exptObj.stimCoords[sweep+1]
            patternID = pattern_index.get_patternID(sqSet)

            try:
                fitTrace   = self.expectedResponse[exptID][patternID][5]
            except:
                fitTrace = np.zeros(len(pdTrace))

            coordArrayTemp = np.zeros((15))            
            coordArrayTemp[:numSquares] = exptObj.stimCoords[sweep+1]

            if exptObj.clamp == 'VC' and exptObj.EorI == 'E':
                clampPotential = -70
            elif exptObj.clamp == 'VC' and exptObj.EorI == 'I':
                clampPotential = 0
            elif exptObj.clamp == 'CC':
                clampPotential = -70

            Condition = 1 if (exptObj.condition == 'Gabazine') else 0
            clamp = 0 if (exptObj.clamp == 'CC') else 1

            ap    = 0
            if exptObj.clamp == 'CC' and np.max(sweepTrace[4460:stimEnd])>30:
                ap = 1
            
            tempArray  = np.array([int(exptObj.dataFile[-12:-8]),
                                   int(sweep+1),
                                   exptObj.stimFreq,
                                   numSquares,
                                   exptObj.stimIntensity,
                                   exptObj.pulseWidth,
                                   np.round(exptObj.baselineTrend[sweep][0],1),
                                   clampPotential,
                                   clamp,
                                   Condition,
                                   ap,
                                   np.round(IR[sweep] , 1),
                                   np.round(tau[sweep], 3),
                                   patternID])
            tempArray2 = np.concatenate((tempArray,coordArrayTemp))
            inputSet  [sweep,:len(tempArray2) ] = tempArray2
            inputSet  [sweep, len(tempArray2):] = pdTrace
            outputSet1[sweep,:] = sweepTrace
            outputSet2[sweep,:] = fitTrace
            outputSet3[sweep,:] = fieldTrace

        newTrainingSet = np.concatenate((inputSet,outputSet1,outputSet2,outputSet3),axis=1)
        try:
            oldTrainingSet = self.trainingSetLong
            self.trainingSetLong = np.concatenate((newTrainingSet,oldTrainingSet),axis=0)
        except AttributeError:
            self.trainingSetLong = newTrainingSet
            
        return self

    def make_spot_profile(self, exptObj1sq):
        if not exptObj1sq.exptType == '1sq20Hz':
            raise ParameterMismatchError(message='Experiment object has to be a 1sq experiment')

        Fs                      = exptObj1sq.Fs
        IPI                     = exptObj1sq.IPI # 0.05 seconds
        numSweeps               = exptObj1sq.numSweeps
        condition               = exptObj1sq.condition
        EorI                    = exptObj1sq.EorI
        stimFreq                = exptObj1sq.stimFreq
        clamp                   = exptObj1sq.clamp

        # Get trial averaged stim and response traces for every spot
        pd                      = exptObj1sq.extract_trial_averaged_data(channels=[2])[2] # 45 x 40000
        cell                    = exptObj1sq.extract_trial_averaged_data(channels=[0])[0] # 45 x 40000
        # Get a dict of all spots
        spotCoords              = dict([(k, exptObj1sq.stimCoords[k]) for k in range(1,1+int(exptObj1sq.numSweeps/exptObj1sq.numRepeats))])
        
        firstPulseTime          = int(Fs*(exptObj1sq.stimStart)) # 4628 sample points
        secondPulseTime         = int(Fs*(exptObj1sq.stimStart + IPI)) # 5628 sample points

        # Get the synaptic delay from the average responses of all the spots
        avgResponseStartTime,_,_    = PSP_start_time(cell,clamp,EorI,stimStartTime=exptObj1sq.stimStart,Fs=Fs)   # 0.2365 seconds
        avgSecondResponseStartTime = avgResponseStartTime + IPI # 0.2865 seconds
        avgSynapticDelay        = 0.0055#avgResponseStartTime-exptObj1sq.stimStart # ~0.0055 seconds
        print(avgResponseStartTime)
        spotExpectedDict        = {}

        for i in range(len(spotCoords)):
            # spotPD_trialAvg               = pd[i,int(Fs*avgResponseStartTime):int(Fs*avgSecondResponseStartTime)] # 1 x 1000
            spotCell_trialAVG_pulse2pulse = cell[i,firstPulseTime:secondPulseTime+200]

            t                   = np.linspace(0,IPI+0.01,len(spotCell_trialAVG_pulse2pulse)) # seconds at Fs sampling
            T                   = np.linspace(0,0.4,int(0.4*Fs)) # seconds at Fs sampling
            popt,_              = curve_fit(delayed_alpha_function,t,spotCell_trialAVG_pulse2pulse,p0=([0.5,0.05,0.005])) #p0 are initial guesses A=0.5 mV, tau=50ms,delta=5ms
            A,tau,delta         = popt
            fittedSpotRes       = delayed_alpha_function(T,*popt) # 400 ms = 8000 datapoints long predicted trace from the fit for the spot, not really usable
            
            spotExpectedDict[spotCoords[i+1][0]] = [avgSynapticDelay, A, tau, delta, spotCell_trialAVG_pulse2pulse, fittedSpotRes]
        
        all1sqAvg                     = np.mean(cell[:,firstPulseTime:secondPulseTime+200],axis=0)
        popt,_                        = curve_fit(delayed_alpha_function,t,all1sqAvg,p0=([0.5,0.05,0.005])) #p0 are initial guesses A=0.5 mV, tau=50ms,delta=5ms
        A,tau,delta                   = popt
        all1sqAvg_fittedSpotRes       = delayed_alpha_function(T,*popt)
        spotExpectedDict['1sqAvg']    = [avgSynapticDelay, A, tau, delta, all1sqAvg, all1sqAvg_fittedSpotRes]

        return spotExpectedDict

    def find_frame_expected(self, exptObj, spotExpectedDict_1sq):
        stimFreq        = exptObj.stimFreq
        IPI             = exptObj.IPI # IPI of current freq sweep experiment
        numPulses       = exptObj.numPulses
        numSweeps       = exptObj.numSweeps
        numRepeats      = exptObj.numRepeats
        cell            = exptObj.extract_trial_averaged_data(channels=[0])[0][:,:20000] # 8 x 40000 #TODO Hardcoded variable: slice length
        stimCoords      = dict([(k, exptObj.stimCoords[k]) for k in range(1,1+int(numSweeps/numRepeats))]) # {8 key dict}
        stimStart       = exptObj.stimStart
        Fs              = exptObj.Fs
    
        frameExpected   = {}
        for i in range(len(stimCoords)):
            coordsTemp = stimCoords[i+1] # nd array of spot coords
            frameID    = pattern_index.get_patternID(coordsTemp)
            numSq      = len(coordsTemp)
            firstPulseExpected = np.zeros((int(Fs*(IPI+0.01))))
            firstPulseFitted   = np.zeros((int(Fs*(0.4))))# added 0.01 second = 10 ms to IPI, check line 41,43,68,69
            
            for spot in coordsTemp:            
                spotExpected        = spotExpectedDict_1sq[spot][4]
                spotFitted          = spotExpectedDict_1sq['1sqAvg'][5]
                firstPulseExpected  += spotExpected[:len(firstPulseExpected)]
                firstPulseFitted    += spotFitted[:len(firstPulseFitted)]
            avgSynapticDelay    = spotExpectedDict_1sq[spot][0]
            expectedResToPulses = np.zeros(10000+len(cell[0,:]))
            fittedResToPulses   = np.zeros(10000 + len(cell[0,:]) )
            t1 = int(Fs*stimStart)

            for k in range(numPulses):
                t2 = t1+int(Fs*IPI+avgSynapticDelay)
                T2 = t1+int(0.4*Fs)
                window1 = range(t1,t2)
                window2 = range(t1,T2)
                # print(i, frameID, avgSynapticDelay, t1,t2, T2)
                expectedResToPulses[window1] += firstPulseExpected[:len(window1)]
                
                fittedResToPulses[window2]   += firstPulseFitted[:len(window2)]
                t1 = t1+int(Fs*IPI)
            fittedResToPulses   =   fittedResToPulses[:len(cell[0,:])]
            expectedResToPulses = expectedResToPulses[:len(cell[0,:])]
            frameExpected[frameID] = [numSq, stimFreq, exptObj.stimIntensity, exptObj.pulseWidth, expectedResToPulses, fittedResToPulses, firstPulseFitted, firstPulseExpected]
        
        return frameExpected

    def find_sweep_expected(self, exptObj):
        # sweep_expected_array = []
        # cell = exptObj.extract_channelwise_data(exclude_channels=[1,2,3,'Cmd','Time'])

        # first_pulse_time = 0

        # return sweep_expected_array
        pass
    
    def add_cell_to_xl_db(self, excel_file):
        # excel_file = os.path.join(project_path_root,all_cells_response_file)
        try:
            tempDF  = pd.read_excel(excel_file)
        except FileNotFoundError:
            tempDF  = pd.DataFrame()
        outDF       = pd.concat([self.response,tempDF],ignore_index=True)
        # outDF       = pd.concat([cell.response,tempDF],axis=1)
        outDF       = outDF.drop_duplicates()
        outDF.to_excel(excel_file) #(all_cells_response_file)
        print("Cell experiment data has been added to {}".format(excel_file))

    def save_training_set(self, directory):
        # celltrainingSetLong = self.trainingSetLong
        filename = "cell"+str(self.cellID)+"_trainingSet_longest.h5"
        trainingSetFile = os.path.join(directory,filename)
        self.data.to_hdf(trainingSetFile, format='fixed', key='data', mode='w')
        
        # del self.trainingSetLong
        # with h5py.File(trainingSetFile, 'w') as f:
        #     dset = f.create_dataset("default", data = celltrainingSetLong)
        print('Cell Data exported to {}'.format(trainingSetFile))

    def summarize_experiments(self):
        df = pd.DataFrame(columns=['Polygon Protocol','Expt Type','Condition','Stim Freq','Stim Intensity','Pulse Width','Clamp','Clamping Potential'])
        for exptID,expt in self.experiments.items():
            df.loc[exptID] ={
                            'Polygon Protocol'  : expt[-1].polygonProtocolFile,
                            'Expt Type'         : expt[-1].exptType,
                            'Condition'         : expt[-1].condition,
                            'Stim Freq'         : expt[-1].stimFreq,
                            'Stim Intensity'    : expt[-1].stimIntensity,
                            'Pulse Width'       : expt[-1].pulseWidth,
                            'Clamp'             : expt[-1].clamp,
                            'Clamping Potential': expt[-1].clampPotential
                            }
        print(df)
        return df
    
    @staticmethod
    def saveCell(neuronObj,filename):
        directory       = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filename, 'wb') as fout:
            print("Neuron object saved into pickle. Use loadCell to load back.")
            pickle.dump(neuronObj, fout, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def loadCell(filename):
        try:
            with open(filename,'rb') as fin:
                return pickle.load(fin)
        except FileNotFoundError:
            print("File not found.")
            raise Exception

    def __iter__(self):
        return self.experiments.iteritems()


class Experiment:
    '''All different kinds of experiments conducted on a patched
    neuron are captured by this superclass.'''

    def __init__(self,exptParams,datafile,coordfile=None):
        try:
            self.exptParamsParser(exptParams)
        except ParameterMismatchError as err:
            print(err)

        self.Flags          = {"IRFlag":False,"APFlag":False,"NoisyBaselineFlag":False,"TauChangeFlag":False}
        datafile            = os.path.abspath(datafile)
        data                = abf_to_data(datafile,
                                          baseline_criterion=exptParams.baselineCriterion,
                                          sweep_baseline_epoch=exptParams.sweepBaselineEpoch,
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
        
        if coordfile:
            self.opticalStim    = Coords(coordfile)  
            self.stimCoords     = self.opticalStim.coords
        else:
            self.opticalStim    = ''
            self.stimCoords     = ''

        self.numSweeps      = len(self.recordingData.keys())
        self.sweepIndex     = 0  # start of the iterator over sweeps

    def __iter__(self):
        return self

    def __next__(self):
        if self.sweepIndex >= self.numSweeps:
            raise StopIteration
        currentSweepIndex   = self.sweepIndex
        self.sweepIndex     += 1
        return self.recordingData[currentSweepIndex]

    def extract_channelwise_data(self,exclude_channels=[]):
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
        tempChannelData   = np.zeros((numSweeps,sweepLength))

        included_channels = list( set(chLabels) - set(exclude_channels) )
    
        channelDict       = {}
        tempChannelData   = np.zeros((numSweeps,sweepLength))
        for ch in included_channels:
            for i in range(numSweeps):
                tempChannelData[i,:] = sweepwise_dict[i][ch]
            channelDict[ch] = tempChannelData
            tempChannelData = 0.0*tempChannelData            
        return channelDict

    def extract_trial_averaged_data(self,channels=[0]):
        '''
        Returns a dictionary holding channels as keys,
        and trial averaged sweeps as an nxm 2-d array where n is number of patterns
        and m is number of datapoints in the recording per sweep.
        '''
        chData = self.extract_channelwise_data(exclude_channels=[])
        chMean = {}
        for ch in channels:
            chData_temp = np.reshape(chData[ch],(self.numRepeats,int(self.numSweeps/self.numRepeats),-1))
            chMean[ch] = np.mean(chData_temp,axis=0)

        return chMean

    def analyze_experiment(self,neuron,exptParams):

        if self.exptType == 'sealTest':
            # Call a function to calculate access resistance from recording
            return self.sealTest()
        elif self.exptType == 'IR':
            # Call a function to calculate cell input resistance from recording
            return self.inputRes(self,neuron)
        elif self.exptType in ['1sq20Hz','FreqSweep','LTMRand','LTMSeq','convergence']:
            # Call a function to analyze the freq dependent response
            return self.FreqResponse(neuron,exptParams)
        elif self.exptType in ['SpikeTrain']:
            # Call a function to analyze the freq dependent response
            return self.SpikeTrainResponse(neuron,exptParams)

    def sealTest(self):
        # calculate access resistance from data, currently not implemented
        return self

    def inputRes(self,neuron):
        # calculate input resistance from data
        neuron.properties.update({'IR':np.mean(IR_calc(self.recordingData,np.arange[1,200],np.arange[500,700]))})
        return self

    # FIXME: improve feature (do away with so many nested functions)
    def FreqResponse(self,neuron,exptParams):
        # there can be multiple kinds of freq based experiments and their responses.
        return expt2df(self,neuron,exptParams)  # this function takes expt and converts to a dataframe of responses

    def SpikeTrainResponse(self, neuron, exptParams):
        # Analysis of cell response to a invivo like poisson spike train
        return spiketrain_analysis(self, neuron, exptParams)
    
    def exptParamsParser(self,ep):
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
            self.polygonProtocolFile= ep.polygonProtocol
            self.numRepeats         = ep.repeats
            self.numPulses          = ep.numPulses
            self.IPI                = 1 / ep.stimFreq
            self.stimStart          = ep.opticalStimEpoch[0]
            self.Fs                 = ep.Fs
            self.exptType           = ep.exptType
            self.condition          = ep.condition

            self.sweepDuration      = ep.sweepDuration      
            self.sweepBaselineEpoch = ep.sweepBaselineEpoch 
            self.opticalStimEpoch   = ep.opticalStimEpoch   
            self.IRBaselineEpoch    = ep.IRBaselineEpoch    
            self.IRpulseEpoch       = ep.IRpulseEpoch       
            self.IRchargingPeriod   = ep.IRchargingPeriod   
            self.IRsteadystatePeriod= ep.IRsteadystatePeriod
            self.interSweepInterval = ep.interSweepInterval

            self.unit = 'pA' if self.clamp == 'VC' else 'mV' if self.clamp == 'CC' else 'a.u.'
        except Exception as err:
            raise ParameterMismatchError(message=err)

        return self


class Coords:
    """
    An object that stores Sweep wise record of coordinates of all the square points
    illuminated in the experiment.
    currently class "Coords" is not being used except in generating
    a dict containing sweep wise coords
    """

    def __init__(self,coordFile):
        self.gridSize       = []
        self.numSweeps      = []
        self.coords         = self.coordParser(coordFile)

    def coordParser(self,coordFile):
        coords              = {}
        import csv
        with open(coordFile,'r') as cf:
            c               = csv.reader(cf, delimiter=" ")
            for lines in c:
                intline     = []
                for i in lines:
                    intline.append(int(i))
                frameID     = intline[0]
                coords[frameID] = (intline[3:])
        self.gridSize       = [intline[1],intline[2]]
        self.numSweeps      = len(coords)
        return coords

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
class EphysData:
    """
    Experimental Class: actual experimental data.
    """
    def __init__(self,abfFile, exptParams):
        self.datafile           = abfFile

        self.filter             = exptParams.filter
        self.filterCutoff       = exptParams.filterHighCutoff
        self.sweepBaselineEpoch = exptParams.sweepBaselineEpoch
        self.baselineSubtraction= exptParams.baselineSubtraction
        self.scaling            = exptParams.signalScaling

        self.data               = self.parseABF()
        self.numSweeps          = 0        
        self.numChannels        = 0

    def parseABF(self,exptParams):
        self.data = abf_to_data(self.datafile,
                                baseline_criterion=exptParams.baselineCriterion,
                                sweep_baseline_epoch=exptParams.sweepBaselineEpoch,
                                signal_scaling=exptParams.signalScaling,
                                sampling_freq=exptParams.Fs,
                                filter_type=exptParams.filter,
                                filter_cutoff=exptParams.filterHighCutOff,
                                plot_sample_data=False)
    
    def plotEphysData(self):
        return None