import numpy as np
import pandas as pd
import matplotlib.pyploy as plt


spiketraindata = pd.read_hdf(data_path / "all_cells_SpikeTrain_CC_long.h5", key='data')

def plot_spike_train_STA(df, threshold=50, squares='all', plot=False, ax=None):
    # fill in the dfsta with the mean_sta for each cell, for pattern
    cells = df['cellID'].unique()
    patterns = np.unique(df['patternList'].to_numpy())
    stadict = []
    for c in cells:
        for expt in np.unique(df[df['cellID']==c]['exptID']):
            for p in patterns:
                celltrials = df[(df['cellID']==c) & (df['exptID']==expt) & (df['patternList']==p)]
                cellID = c
                patternID = p
                # expt = celltrials['exptID'].to_numpy()[0]
                numSq = df[(df['cellID']==c) & (df['patternList']==p)]['numSq'].to_numpy()[0]
                for i in range(len(celltrials)):
                    trial_cell = celltrials.iloc[i,49:220049]
                    trial_stim = celltrials.iloc[i,440049:660049] / np.max(celltrials.iloc[i,440049:660049])
                    APlocs, _ = find_peaks(trial_cell, height=threshold, distance=1000)
                    if APlocs.shape[0] == 0:
                        print(c, p, numSq, i, 'No spikes found')
                        continue
                    print(APlocs)
                    APlocs = APlocs[APlocs>2000]
					
                    # if there are APlocs very close to each other, then pick the first one
                    isolated_peaks = np.concatenate([ [True], np.diff(APlocs)>200 ], axis=0)
                    APlocs = APlocs[isolated_peaks]
                    numSpikes = len(APlocs)
                    if numSpikes == 0:
                        continue

                    pre_event_stim = np.zeros((numSpikes,2000))
                    # remove APlocs that occur earlier than 2000 from the list
                    for n in range(numSpikes):
                        print(c, expt, p, numSpikes, numSq, i, n, APlocs[n])
                        rown = np.concatenate( [[int(c)], [int(expt)], [int(p)], [int(numSq)], [i], [n], [APlocs[n]], trial_stim[APlocs[n]-2000:APlocs[n]] ], axis=0)
                        pre_event_stim[n,:] = trial_stim[APlocs[n]-2000:APlocs[n]]
                        stadict.append(rown)

                    mean_sta = np.mean(pre_event_stim, axis=0)
                    rowmean = np.concatenate( [[int(c)], [int(expt)], [int(p)], [int(numSq)], [1000], [1000], [1000], mean_sta ], axis=0)

                    stadict.append(rowmean)

    #make df
    dfsta = pd.DataFrame(stadict)
    return dfsta
	
dfsta1 = plot_spike_train_STA(spiketraindata)
dfsta1.columns = ['cellID', 'exptID', 'patternList', 'numSq', 'trial', 'event', 'peakloc', *np.linspace(-100,0,2000)]

fig, ax = plt.subplots(8,2, figsize=(12,12))
for j,c in enumerate(dfsta1['cellID'].unique()):
    for i,p in enumerate(dfsta1['patternList'].unique()):
        dfsta1temp = dfsta1[(dfsta1['cellID']==c) & (dfsta1['patternList']==p) & (dfsta1['event']!=1000) ]
        if dfsta1temp.shape[0] == 0:
            continue
        print(dfsta1temp.shape)
        for k in range(dfsta1temp.shape[0]):
            ax[i,j].fill_between(np.linspace(-100, 0, 2000), dfsta1temp.iloc[k,7:], color='purple', alpha=0.2)
            ax[i,j].set_title(f'Cell {c}, Pattern {p}')
            ax[i,j].set_xlabel('Time before a spike event (ms)')
            ax[i,j].set_ylabel('LED intensity (a.u.)')
        sns.despine(bottom=False, left=False, ax=ax[i,j])

plt.tight_layout()