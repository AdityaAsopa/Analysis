import numpy as np
import datetime

# Animal
# Base genotype Grik4Cre C57BL/6-Tg(Grik4-cre)G32-4Stl/J (Jax Stock No. 006474)
# NCBS Strain: Grik4Cre_2018
animalID        = 'GrikAA198'
sex             = 'F'
dateofBirth     = datetime.date(2021, 8, 9)
dateofInj       = datetime.date(2021, 9, 14)
dateofExpt      = datetime.date(2021, 12, 16)
sliceThickness  = 350 											# um

# Virus Injection
site            = {'RC':1.9, 'ML':2.0, 'DV':1.5} 					# from bregma, right hemisphere in rostrocaudal, mediolateral, and dorsoventral axes
injectionParams = {'Pressure':10, 'pulseWidth':15, 'duration':30}  # picospritzer nitrogen pressure in psi, pulse width in millisecond, duration in minutes
virus           = 'ChR2'											# ChR2 (Addgene 18917) or ChETA (Addgene 35507)
virusTitre      = 6e12											# GC/ml, after dilution
dilution        = 0.5												# in PBS to make up the titre
volumeInj       = 5e-4 											# approx volume in ml
ageAtInj        = (dateofInj	- dateofBirth)
ageAtExp        = (dateofExpt	- dateofBirth)
incubation      = (ageAtExp	- ageAtInj)

# Polygon
objMag          = 40 												# magnification in x
frameSize       = np.array([13433.6, 7296.8])					    # frame size in um, with 1x magnification, 24Dec21 calibration
gridSize        = 24												# corresponds to pixel size of 13x8 µm
squareSize      = frameSize / (gridSize * objMag)

# Internal solution (is)
isBatch         = datetime.date(2022, 9, 1)
ispH            = 7.33
isOsm           = 293												# mOsm/kg H2O

# Recording solution (aCSF)
aCSFpH          = 7.39
aCSFOsm         = 303												# mOsm/kg H2O
gabaConc        = 2e-6                                            # mol/litre, if gabazine experiments were done

# Experiment
cellID			= '1981'

bathTemp		= 32												# degree celsius
location		= {'stim':'CA3',0:'CA1',3:'CA3'}                                       #usually, ch0: patch electrode, ch3: field electrode
clamp			= 'VC'
EorI			= 'I'
unit			= 'pA'
clampPotential  = 0

datafile		= '2022_12_16_0008_rec.abf'
polygonProtocol	= '7_221108_24hex_15sq_Convergence+PulseTrain_ExtFreq_1repeat_8sweeps.txt'

intensity		= 100
pulseWidth		= 5
stimFreq		= 20 												# in Hz
repeats			= 5
numPulses		= 20													# a fixed number for all frequencies

exptTypes		= ['GapFree','IR','CurrentStep','1sq20Hz','FreqSweep','LTMSeq','LTMRand','convergence']
exptType		= exptTypes[-1]

conditions		= ['Control','Gabazine']
condition		= conditions[0]

# Signal parameters
Fs						= 2e4
signalScaling			= 1											# usually 1, but sometimes the DAQ does not save current values in proper units
baselineSubtraction		= True
baselineCriterion		= 0.1										# baseline fluctuations of 10% are allowed
DAQfilterBand           = [0, 10000]

# Epochs (time in seconds)
sweepDuration           = [0  , 4.0000]
sweepBaselineEpoch      = [0  , 0.2625]								# seconds, end of baseline time
opticalStimEpoch        = [0.26275, 2.75]
singlePulseEpoch		= [0.26275, 0.76275]
pulseTrainEpoch			= [1.81275, 2.75]
frameChangeforFreqSweep = 'Auto'
IRBaselineEpoch         = [0  , 0.2625]
IRpulseEpoch            = [2.76245, 3.06245]
IRchargingPeriod        = [2.76245, 2.81245]
IRsteadystatePeriod     = [2.86245, 3.06245]

interSweepInterval      = 30                                        # seconds

# Analysis
# Filtering
filters			        = {0:'',1:'bessel',2:'butter',3:'decimate'}
filter                  = filters[0]
filterHighCutoff        = 2e3