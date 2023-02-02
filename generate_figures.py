'''Generate figure 1
'''
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_context('talk')
import matplotlib.pyplot as plt
import h5py
import eidynamics.utils as utils
import sys


allCellDataFile = "C:\\Users\\aditya\\OneDrive\\NCBS\\Lab\\Projects\\EI_Dynamics\\trainingSet_Long\\allCells_trainingSet_short.h5"
allCellExcelFile = "C:\\Users\\adity\\OneDrive\\NCBS\Lab\\Projects\\EI_Dynamics\\AnalysisFiles\\allCells.xlsx"
ephysStart = 28
ephysEnd   = 20028
fitStart   = 20028
fitEnd     = 40028

def main(fignum):
    fignum = int(fignum)

    if fignum ==1:
        print("making figure: ",fignum)    
        with h5py.File( allCellDataFile, "r") as f:
            print("Keys:", f.keys())
            print("data:", f['default'] )
            df = pd.DataFrame( f['default'] )
            df.rename( columns = {0:"StimFreq", 1:"numSq", 2: "intensity", 3: "pulseWidth", 4: "MeanBaseline", 5: "ClampingPotl", 6:"Clamp", 7: "GABAzineFlag", 8:"AP",9:"InputRes", 10:"Tau", 11:"patternID"}, inplace = True )
            
            # figure1(df)
            # plt.show()
            
            # ahp_figure(df)
            # plt.show()

            fit_vs_observed_trace_comparison(df)
            plt.show()
    elif fignum == 2: # VC recordings are useless
        with h5py.File( allCellDataFile, "r") as f:
            print("Keys:", f.keys())
            print("data:", f['default'] )
            df = pd.DataFrame( f['default'] )
            df.rename( columns = {0:"StimFreq", 1:"numSq", 2: "intensity", 3: "pulseWidth", 4: "MeanBaseline", 5: "ClampingPotl", 6:"Clamp", 7: "GABAzineFlag", 8:"AP",9:"InputRes", 10:"Tau", 11:"patternID"}, inplace = True )
            figure2(df)
            plt.show()
    elif fignum ==3: # VC recordings are useless
        df2 = pd.read_excel(allCellExcelFile)
        df2 = df2.iloc[:,:44]
        df2 = df2.rename(columns ={ 1:"P1", 2:"P2", 3:"P3", 4:"P4", 5:"P5", 6:"P6", 7:"P7", 8:"P8",\
                                9:"PT1",10:"PT2",11:"PT3",12:"PT4",13:"PT5",14:"PT6",15:"PT7",16:"PT8",\
                                17:"A1",18:"A2",19:"A3",20:"A4",21:"A5",22:"A6",23:"A7",24:"A8"} )
        # Useless figure 
        figure3(df2)
    elif fignum ==4:
        df2 = pd.read_excel(allCellExcelFile)
        df2 = df2.iloc[:,:44]
        df2 = df2.rename(columns ={ 1:"P1", 2:"P2", 3:"P3", 4:"P4", 5:"P5", 6:"P6", 7:"P7", 8:"P8",\
                                9:"PT1",10:"PT2",11:"PT3",12:"PT4",13:"PT5",14:"PT6",15:"PT7",16:"PT8",\
                                17:"A1",18:"A2",19:"A3",20:"A4",21:"A5",22:"A6",23:"A7",24:"A8"} )
        figure4(df2)
    
    # return df,df2

def figure1(df):
    df = df.loc[ (df["Clamp"]==0) & (df["numSq"]!=1) & (df["patternID"]<56) & (df["intensity"]==100) & (df["GABAzineFlag"] == 0 ) & (df["pulseWidth"]==2) & (df["AP"]==0) ]
    df["numSq"] = df["numSq"].astype(int)
    df["StimFreq"] = df["StimFreq"].astype(int)
    df["GABAzineFlag"] = df["GABAzineFlag"].astype(int)
    df["GABAzineFlag"] = df["GABAzineFlag"].astype(str)

    freqs = np.unique(df["StimFreq"])
    spots = np.unique(df["numSq"])
    fig,ax = plt.subplots(2,4,sharex=True,sharey=True)
    for i,sq in enumerate(spots):
        for j,freq in enumerate(freqs):
            # subplotWin = (4*(i)+j+1)
            # print(i,j,sq,freq,subplotWin)
            # ax = plt.subplot(2,4,subplotWin)
            p0 = df.loc[(df["numSq"]==sq) & (df["StimFreq"]==freq)]
            patdata = p0.iloc[:,ephysStart:ephysEnd]
            patmean = patdata.mean( axis = 0 )
            
            for k, p1 in patdata.iterrows():
                if df.loc[k,"GABAzineFlag"] == '1':
                    df.loc[k,"GABAzineFlag"] = 'Gabazine'
                elif df.loc[k,"GABAzineFlag"] == '0':
                    df.loc[k,"GABAzineFlag"] = 'Control'

                a = ax[i,j].plot( np.linspace(0,1000,20000), p1, color='#91bfdb',alpha=0.05)#,label='all')
            me = ax[i,j].plot( np.linspace(0,1000,20000), patmean, color="#4575b4")#,label='mean' )
            ax[i,j].set_xlabel( "Time (ms)" ) if i==1 else None
            ax[i,j].set_ylabel( "EPSP (mV)" ) if j==0 else None
            ax[i,j].set_ylim([-5.0,10])
            # ax[i,j].set_title(str(int(freq))+" Hz")
            # ax[i,j].legend(loc="upper right")
    # fig.legend([me],labels=['mean'],loc='right')

def figure2(df):
    df = df.loc[ (df["Clamp"]==1) & (df["numSq"]!=1)& (df["numSq"]!=7) & (df["patternID"]<56) & (df["GABAzineFlag"] == 0 ) & (df["AP"]==0) ]
    freqs = np.unique(df["StimFreq"])[:-1]
    spots = np.unique(df["numSq"])
    clamps = np.unique(df["ClampingPotl"])
    fig,ax = plt.subplots(4,4,sharex=True,sharey=True)
    for i,sq in enumerate(spots):
        for j,freq in enumerate(freqs):
            for k,ei in enumerate(clamps):
                # subplotWin = (4*(i)+j+1)

                # ax[i,j].plot(2,4,subplotWin)
                p0 = df.loc[(df["numSq"]==sq) & (df["StimFreq"]==freq) & (df["ClampingPotl"]==ei)]
                
                patdata = p0.iloc[:,ephysStart:ephysEnd]
                patmean = patdata.mean( axis = 0 )
                
                if k==0:
                    for m, p1 in patdata.iterrows():
                        epsc1Peak = -1*np.min(p1)
                        p1 = p1/epsc1Peak
                        p1 = utils.filter_data(p1, filter_type='butter',high_cutoff=1000,sampling_freq=2e4)
                        ax[2*i,j].plot( np.linspace(0,1000,20000), p1, color='#91bfdb',alpha=0.05)
                    epsc1Peak_formean = -1*np.min(patmean)
                    patmean = patmean/epsc1Peak_formean
                    patmean = utils.filter_data(patmean, filter_type='butter',high_cutoff=1000,sampling_freq=2e4)
                    E = ax[2*i,j].plot( np.linspace(0,1000,20000), patmean, color="#4575b4")
                    # ax[2*i,j].set_xlabel("Time (ms)") if j==0 else None
                    ax[2*i,j].set_ylabel( "Normalized EPSC" ) if j==0 else None
                    # ax[2*i+1,j].legend(loc="upper right")
                    # ax[2*i,j].set_title((str(int(freq))+" Hz"))

                elif k==1:
                    for m, p1 in patdata.iterrows():                        
                        ipsc1Peak = np.max(p1)
                        p1 = p1/ipsc1Peak
                        p1 = utils.filter_data(p1, filter_type='butter',high_cutoff=1000,sampling_freq=2e4)
                        ax[2*i+1,j].plot( np.linspace(0,1000,20000), p1, color='#b35806',alpha=0.05 )
                    ipsc1Peak_formean = np.max(patmean)
                    patmean = patmean/ipsc1Peak_formean
                    patmean = utils.filter_data(patmean, filter_type='butter',high_cutoff=1000,sampling_freq=2e4)
                    I = ax[2*i+1,j].plot( np.linspace(0,1000,20000), patmean, color="#b35806")
                
                    ax[3,j].set_xlabel("Time (ms)") if i==1 else None
                    ax[2*i+1,j].set_ylabel( "Normalized IPSC" ) if j==0 else None
                    # ax[i,j].legend(loc="upper right")
                    # ax[2*i+1,j].set_title((str(int(freq))+" Hz"))
    fig.legend([E,I],labels=['E','I'],loc='right')
    
def figure2_(df, include_spikes=True):
    if include_spikes == True:
        df = df.loc[ (df["Clamp"]=='VC') & (df["numSq"]!=1)& (df["numSq"]!=7) & (df["patternID"]<56) & (df["Condition"] == 'CTRL' ) ]
    else:
        df = df.loc[ (df["Clamp"]=='VC') & (df["numSq"]!=1)& (df["numSq"]!=7) & (df["patternID"]<56) & (df["Condition"] == 'CTRL' ) & (df["AP"]==0) ]
        
    freqs  = np.unique(df["StimFreq"])
    spots  = np.unique(df["numSq"])
    clamps = np.unique(df["ClampingPotl"])
    
    fig,ax = plt.subplots(4,4,sharex=True,sharey=True)
    
    ephysStart = 20038
    ephysEnd   = 40038
    
    for i,sq in enumerate(spots):
        for j,freq in enumerate(freqs):
            for k,ei in enumerate(clamps):

                p0      = df.loc[(df["numSq"]==sq) & (df["StimFreq"]==freq) & (df["ClampingPotl"]==ei)]
                
                patdata = p0.iloc[:,ephysStart:ephysEnd]
                patmean = patdata.mean( axis = 0 )
                
                if k==0:
                    for m, p1 in patdata.iterrows():
                        epsc1Peak = -1*np.min(p1)
                        p1 = p1/epsc1Peak
                        p1 = utils.filter_data(p1, filter_type='butter',high_cutoff=1000,sampling_freq=2e4)
                        ax[2*i,j].plot( np.linspace(0,1000,20000), p1, color='#91bfdb',alpha=0.05)
                    epsc1Peak_formean = -1*np.min(patmean)
                    patmean = patmean/epsc1Peak_formean
                    patmean = utils.filter_data(patmean, filter_type='butter',high_cutoff=1000,sampling_freq=2e4)
                    E = ax[2*i,j].plot( np.linspace(0,1000,20000), patmean, color="#4575b4")
                    # ax[2*i,j].set_xlabel("Time (ms)") if j==0 else None
                    ax[2*i,j].set_ylabel( "Normalized EPSC" ) if j==0 else None
                    # ax[2*i+1,j].legend(loc="upper right")
                    # ax[2*i,j].set_title((str(int(freq))+" Hz"))

                elif k==1:
                    for m, p1 in patdata.iterrows():                        
                        ipsc1Peak = np.max(p1)
                        p1 = p1/ipsc1Peak
                        p1 = utils.filter_data(p1, filter_type='butter',high_cutoff=1000,sampling_freq=2e4)
                        ax[2*i+1,j].plot( np.linspace(0,1000,20000), p1, color='#b35806',alpha=0.05 )
                    ipsc1Peak_formean = np.max(patmean)
                    patmean = patmean/ipsc1Peak_formean
                    patmean = utils.filter_data(patmean, filter_type='butter',high_cutoff=1000,sampling_freq=2e4)
                    I = ax[2*i+1,j].plot( np.linspace(0,1000,20000), patmean, color="#b35806")
                
                    ax[3,j].set_xlabel("Time (ms)") if i==1 else None
                    ax[2*i+1,j].set_ylabel( "Normalized IPSC" ) if j==0 else None
                    # ax[i,j].legend(loc="upper right")
                    # ax[2*i+1,j].set_title((str(int(freq))+" Hz"))
                    
    fig.legend([E,I],labels=['E','I'],loc='right')
    
def figure3(df2):
    '''Histogram of normalized IPSC and EPSC values
    (df["Clamp"]=='VC') & (df["NumSquares"]!=1) & (df["ExptType"] != 'LTMRand') &\
               (df["Intensity"] == 100) & (df["PulseWidth"] == 5) & (df["datafile"] != '2021_04_03_0004_rec.abf') &\
                 (df["datafile"] != '2021_04_03_0004_rec.abf')]'''
    df3 = df2.loc[ (df2["Clamp"]=='VC') & (df2["EI"]=='E') &(df2["NumSquares"]!=1) & (df2["ExptType"] != 'LTMRand') ]#& (df2["datafile"] != '2021_04_03_0004_rec.abf') ]
    p1ser = df3["P1"]
    df4 = df3.loc[:,["P1","P2","P3","P4","P5","P6","P7","P8"]].div(p1ser,axis='rows')
    allVal = df4.to_numpy()
    allVal = np.reshape(allVal,newshape=-1)
    fig = sns.histplot(data=allVal,color="#4575b4",kde=True,alpha=0.5)
    fig.set_xlim([0,10])

    df3 = df2.loc[ (df2["Clamp"]=='VC') & (df2["EI"]=='I') &(df2["NumSquares"]!=1) & (df2["ExptType"] != 'LTMRand') ]#& (df2["datafile"] != '2021_04_03_0004_rec.abf') ]
    p1ser = df3["P1"]
    df4 = df3.loc[:,["P1","P2","P3","P4","P5","P6","P7","P8"]].div(p1ser,axis='rows')
    allVal = df4.to_numpy()
    allVal = np.reshape(allVal,newshape=-1)
    fig = sns.histplot(data=allVal,color="#b35806",kde=True,alpha=0.5)
    fig.set_xlim([0,10])
    plt.show()

def figure4(df2):
    df3 = df2.loc[ (df2["Clamp"]=='CC')  & (df2["NumSquares"]!=1) & (df2["ExptType"] != 'LTMRand') & (df2["datafile"] != '2021_04_03_0004_rec.abf') & (df2["AP"]==0)]
    p1ser = df3["P1"]
    df4 = df3.copy()
    df4.loc[:,["P1","P2","P3","P4","P5","P6","P7","P8"]] = df3.loc[:,["P1","P2","P3","P4","P5","P6","P7","P8"]].div(p1ser,axis='rows')
    df4.loc[:,["A1","A2","A3","A4","A5","A6","A7","A8"]] *= (1/20000)

    _,grid1 = plot_df_slice(df3,ploty="peak",plotby='Condition')
    grid1.set_axis_labels("Pulse Index","Peak Resonse (mV)")

    _,grid2 = plot_df_slice(df3,ploty="auc",plotby="Condition")
    grid2.set_axis_labels("Pulse Index","AuC of Response (mV-s)")

    plt.show()

def plot_df_slice(df3,ploty="peak",plotby='Condition'):

    unit = df3.iloc[1,df3.columns.get_loc("Unit")]
    
    if ploty == "peak":
        vals = ["P1","P2","P3","P4","P5","P6","P7","P8"]
        valName = "Peak Response Value (" + unit + ")"
    elif ploty == "peakTime":
        vals = ["PT1","PT2","PT3","PT4","PT5","PT6","PT7","PT8"]
        valName = "Onset Delay (ms)"
    elif ploty == "auc":
        vals = ["A1","A2","A3","A4","A5","A6","A7","A8"]
        valName = "AUC (mV-s)"

    # Separate the identifier variables from value variables by melting the dataframe
    respMelt = pd.melt(df3,id_vars=["Repeat","StimFreq","EI","PatternID","NumSquares","Condition"],value_vars=vals,var_name='PulseIndex', value_name=valName)
    
    grid = sns.FacetGrid(respMelt, row="NumSquares", col="StimFreq", hue=plotby, palette="viridis", legend_out=True)
    grid.map(sns.scatterplot,"PulseIndex",valName,marker="o",alpha=0.4)
    plt.ylim(bottom=-200,top=200)
    grid.add_legend()
       
    return df3,grid

def ahp_figure(df):
    df = df.loc[ (df["Clamp"]==0) & (df["intensity"]==100) & (df["numSq"]!=1) & (df["pulseWidth"]==2)  & (df["patternID"]<56) & (df["AP"]==0) ]#&(df["GABAzineFlag"]==0)]
    df["AHPStart"] = (4460+7.5*20000/df["StimFreq"]).astype(int)
    ipi = 20000/df["StimFreq"]
    df["AHPEnd"]   = 16000
    df["numSq"] = df["numSq"].astype(int)
    df["StimFreq"] = df["StimFreq"].astype(int)
    df["GABAzineFlag"] = df["GABAzineFlag"].astype(int)
    df["GABAzineFlag"] = df["GABAzineFlag"].astype(str)

    k=0
    for i,j in df.iterrows():
        t1 = int(j.loc["AHPStart"])
        t2 = int(j.loc["AHPEnd"])
        ipi = int(20000/j.loc["StimFreq"])
        df.loc[i,"AHP"]   = np.trapz(df.iloc[k,t1:t2])/20000 #trying out auc instead of AHP peak
        df.loc[i,"Response"] = np.trapz(df.iloc[k,4460:t1])/20000
        if df.loc[i,"GABAzineFlag"] == '1':
            df.loc[i,"GABAzineFlag"] = 'Gabazine'
        elif df.loc[i,"GABAzineFlag"] == '0':
            df.loc[i,"GABAzineFlag"] = 'Control'
        k+=1
    
    vals = ["AHP"]
    valName = "AHP (mV)"
    '''
    Reference:
    columns = {0:"StimFreq", 1:"numSq", 2: "intensity", 3: "pulseWidth", 4: "MeanBaseline", 5: "ClampingPotl", 6:"Clamp", 7: "GABAzineFlag", 8:"InputRes", 9:"Tau", 10:"patternID",27:"AP"}
    '''
    # Separate the identifier variables from value variables by melting the dataframe
    vals = "AHP"
    valName = "AHP Value (mV)"
    respMelt  = pd.melt(df,id_vars=["numSq","GABAzineFlag","StimFreq","Response"],   value_vars=vals, var_name='feature', value_name=valName)
    
    # ax  = sns.relplot(kind='scatter',data=respMelt,x="TotalAUC",y="AHP Value (mV)",hue="StimFreq",col="GABAzineFlag",palette='viridis')#style='GABAzineFlag')
    # ax  = sns.catplot(data=respMelt,x="StimFreq",y="AHP Value (mV)",hue="numSq",col="GABAzineFlag",kind='box',palette='viridis')#style='GABAzineFlag')
    
    ax  = sns.lmplot(data=respMelt,x="Response",y="AHP Value (mV)",hue="numSq",row="GABAzineFlag",col="StimFreq",palette='viridis')#style='GABAzineFlag')
    # ax.set_xticks([20,30,40,50])
    # ax.legend(loc="lower left",title="Number of Spots")
    
def fit_vs_observed_trace_comparison(df):
    df = df.loc[ (df["Clamp"]==0) & (df["numSq"]!=1) & (df["patternID"]<56) & (df["intensity"]==100) & (df["GABAzineFlag"] == 0 ) & (df["pulseWidth"]==2) & (df["AP"]==0) ]
    df["numSq"]        = df["numSq"].astype(int)
    df["StimFreq"]     = df["StimFreq"].astype(int)

    freqs = np.unique(df["StimFreq"])
    spots = np.unique(df["numSq"])

    fig,ax = plt.subplots(2,4,sharex=True,sharey=True)
    for i,sq in enumerate(spots):
        for j,freq in enumerate(freqs):
            # subplotWin = (4*(i)+j+1)
            # print(i,j,sq,freq,subplotWin)
            # ax = plt.subplot(2,4,subplotWin)
            p0 = df.loc[(df["numSq"]==sq) & (df["StimFreq"]==freq)]
            patdata = p0.iloc[:,ephysStart:ephysEnd]
            fitdata = p0.iloc[:,fitStart:fitEnd]
            patmean = patdata.mean( axis = 0 )
            fitmean = fitdata.mean( axis = 0 )
            
            # for k, p1 in patdata.iterrows():
            #     a = ax[i,j].plot( np.linspace(0,1000,20000), p1, color='g',alpha=0.03)#,label='all')
            me = ax[i,j].plot( np.linspace(0,1000,20000), patmean, color="g",label='Mean Observed' )
            mefit = ax[i,j].plot( np.linspace(0,1000,20000), fitmean, color="b",alpha=0.5,label='Mean Expected')#,label='mean' )
            ax[i,j].set_xlabel( "Time (ms)" ) if i==1 else None
            ax[i,j].set_ylabel( "EPSP (mV)" ) if j==0 else None
            ax[i,j].set_xlim([200,800])
            ax[i,j].set_title(str(int(freq))+" Hz")
            # ax[i,j].legend(loc="upper right")
    fig.legend([me,mefit],labels=['Mean Observed','Mean Expected'],loc='right')


if __name__ == "__main__":
    figNum = sys.argv[1]
    main(figNum)