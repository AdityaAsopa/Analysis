'''
A model of presynaptic short term plasticity dynamics from:
Interplay between Facilitation, Depression, and Residual Calcium at Three Presynaptic Terminals.
Jeremy S. Dittman, Anatol C. Kreitzer, and Wade G. Regehr
J. Neuroscience, 2000
'''

from ftplib import ftpcp
import numpy as np
from scipy.optimize import curve_fit
import scipy.integrate as spi
import matplotlib.pyplot as plt
import utils

# > EPSC = alpha * NT * F * D                                           ---(1)
'''   
alpha = average mEPSC amplitude as a scaling factor
NT = number of release sites
F  = facilitation variable, fraction of available sires activated a stimulus
D  = depression variable, fraction of sires that are release ready
'''

'''Enhancement of release depends on a calcium bound molecule CaXF with a dissociation constant KF'''
# > F(t) = F1 + (1-F1) / ( 1 + KF/CaXF(t) )                            ---(2a)

'''CaXF decays with a time constant of tauF after a jump of delF after an action potential at time=t0'''
# > d/dt (CaXF) = -CaXF(t)/tauF + delF * dd(t-t0)                       ---(3)
'''
dd = dirac delta function
'''

'''At rest, CaXF = 0 and F = F1. Immediately after an action potential, CaXF --> delF, F --> F2 where F2 is given by:'''
# > F2 = F1 + (1-F1) / (1 + KF/delF)                                    ---(4)

'''Therefore, second EPSC:'''
# > EPSC2 = alpha * Nt * F2 * (1-D1*F1)                                 ---(5)

# > rho   = EPSC2 / EPSC1
# > rho   = (alpha * NT * F2 * (1-D1*F1) ) / (alpha * NT * F1 * D1)

'''If initial availability for release is full tt.e. D1 = 1'''
# > rho   = (1-F1) * F2 / F1                                            ---(6)

'''
Table 1. Definitions of parameters used in the FD model
Symbol  Definition
CaXF    Concentration of calcium-bound siteXF
CaXD    Concentration of calcium-bound siteXD
F       Facilitation variable; fraction of available sites activated by a stimulus
D       Depression variable; fraction of sites that are release-ready
pR      Probability of release; product of the probabilities F and D
F1      Initial probability of release
KF      Affinity of CaXF for the release site
KD      Affinity of CaXD for the release site
tF      Decay time constant of CaXF after an action potential
tD      Decay time constant of CaXD after an action potential
delF    Incremental increase in CaXF after a stimulus
delD    Incremental increase in CaXD after a stimulus
ko      Baseline rate of recovery from the refractory state
kmax    Maximal recovery rate from the refractory state
rho     Facilitation ratio EPSC2/EPSC1 for closely spaced EPSCs
NT      Total number of release sites
alpha   Average mEPSC size
'''


dt     = 0.01
tstart = 0
tstop  = 2.0

def EPSC(alpha, NT, F, D):
    EPSCpeak = (alpha * NT * F * D)
    t        = np.linspace(0,1.0,int(1.0/0.01))
    EPSCform = utils.alpha_synapse(t, EPSCpeak, 0.1)
    return EPSCpeak, EPSCform

def dd(t, tspikes=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]):
    return 1 if t in tspikes else 0

def derivate_CaXF(CaXF, t, delF=0.001, tauF=100):
    CaXFdot = -CaXF/tauF + delF * dd(t)
    return CaXFdot

def STP_Dittman(time, rho, F1, tauF, tauD, kmax, k0, KD, D1=1, KF=0.2, delF=0.2):
    CaXF_t = CaXF
    F2 = 0
    return F2



'''
Example model for schaeffer collaterals as shown in Table 2:
'''
rho     = 2.2
F1      = 0.24
tauF    = 100 # ms
tauD    = 50  # ms
kmax    = 30  # s-1
k0      = 2   # s-1
KD      = 2

# --------------------------------------------
# Trial to recreate

tstop = 2
t = np.round(np.arange(0,tstop/dt)*dt,2)
stim = [0]
NT    = 20
F1    = 0.15
D1    = 1
tauF    = 100 #ms
tauD    = 50  # ms
alpha = 0.5 # mV
delF  = 0.001
delD  = 0.005
KF    = 0.2
KD    = 2
CaXF  = 0.00000001
CaXD  = 0.00000001

CaXFt = [CaXF]
CaXDt = [CaXD]
Ft    = [F1]
Dt    = [D1]
EPSCt = np.zeros(int(tstop/dt))
EPSCt[0] = EPSC(alpha,NT,F1,D1)[0]
print(EPSCt[0])

for i,tt in enumerate(t):
    spike = dd(tt)
    stim  = np.append(stim, spike)
    dcfdt = -CaXF/tauF + delF * spike
    dcddt = -CaXD/tauD + delD * spike
    dfdt  = (1-F1) / (1 + KF/CaXF)
    dddt  = (1-D1) / (1 + KD/CaXD)

    
    CaXF  = CaXF + dcfdt
    F     = F1 + dfdt
    D     = D1 + dddt
    epsc,epscform  = EPSC(alpha, NT, F, D)
    EPSCt[i:i+100] = EPSCt[i:i+100] + epscform

    plt.plot(epscform)
    

    Ft    = np.append(Ft, F)
    Dt    = np.append(Dt, D)
    EPSCt = np.append(EPSCt, epsc)
    CaXFt = np.append(CaXFt,CaXF)

plt.show()

CaXFt2 = spi.odeint(derivate_CaXF, 0, t)

plt.plot(Ft)
# plt.plot(stim)
plt.show()