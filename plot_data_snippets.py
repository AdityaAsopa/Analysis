def draw_pulse_response_snippets(dfcell, ax, signal='cell',window=0.15, pre=0.01, patterns =[1,46,52], palette='grey', 
                                between='clampPotential', hue='numSq', hues=[1,5,15],
                                stim_scale=10, stim_offset=0.8, invert=False, filter_data=False, passband=[0.1, 1000], Fs=2e4,
                                draw_pattern=True, draw_listed_pattern=False, grid_size=24, spots_light_on_dark=False, pattern_scale=2):    
    betweens = dfcell[between].unique()
    # print(hues, betweens)
    probe_pulse_time = dfcell.iloc[0]['probePulseStart']
    shift = 0 if signal=='cell' else 60000
    t0 = int(Fs*(probe_pulse_time - pre)) 
    t1 = int(Fs*(probe_pulse_time - pre + window)) 
    tstart = [window*i+pre*i for i in range(len(hues))]
    insets = []
    if hue == 'numSq':
        color_squares = {1:viridis(0.2), 5:viridis(.4), 7:viridis(.6), 15:viridis(.8), 20:viridis(1.0)}
    elif hue == 'clampPotential':
        color_squares = {-70:flare(0.5), 0:crest(0.5)}
    elif hue == 'patternList': # values ranging from 1 to 80
        color_squares = {i:crest(i/80) for i in range(1, 81)}
    
    squares = []
    if hue=='patternList':
        squares = [len(pattern_index.patternID[i]) for i in hues]
    print(squares)
    
    for i, hu in enumerate(hues):
        for j, bet in enumerate(betweens):
            print(f'plotting {hu} and {bet}')
            dfE  = dfcell[(dfcell[hue]==hu) & (dfcell[between]==bet) & (dfcell['patternList'].isin(patterns))]
            print(dfE.shape)
            pat = dfE['patternList'].unique()[0]
            dfslice = dfE.iloc[:, shift+t0:shift+t1].to_numpy()
            if invert:
                dfslice *= -1
            if filter_data:
                dfslice = utils.filter_data(dfslice,filter_type='butter',low_cutoff=passband[0], high_cutoff=passband[1],sampling_freq=2e4)
                if signal=='field':
                    # apply a notch filter
                    dfslice = utils.filter_data(dfslice, filter_type='notch', sampling_freq=2e4)
            # plot the pulse response
            time = np.linspace(tstart[i], tstart[i]+window, int(Fs*window))
            [ax.plot(time, row , color=palette[bet][hu], alpha=0.2, linewidth=2) for row in dfslice]
            ax.plot(time, np.mean(dfslice, axis=0) , color=palette[bet][hu], alpha=1, linewidth=2)
        dfPD = dfcell[(dfcell[hue]==hu) & (dfcell[between]==bet)].iloc[0, 40000+t0:40000+t1].to_numpy()
        ax.plot(time,  stim_scale*dfPD + stim_offset, color='blue', alpha=0.8)
        if draw_pattern:
            if hue=='patternList':
                sq = squares[i]
                pat = hu
            else:
                sq = hu
            locx, locy = tstart[i]/(3*(window+pre)), 1.0
            inset_dims = pattern_scale*0.1
            axins = ax.inset_axes([locx, locy, inset_dims,inset_dims], transform=ax.transAxes )
            insets.append(axins)
            spot_locs = pattern_index.patternID[pat]
            sq_color = color_squares[sq]
            # make a colormap from sq_color
            Ncolors = 2
            clrlim1 = [1,1,1]
            clrlim2 = color_squares[sq]
            vals = np.ones((Ncolors, 4))
            if spots_light_on_dark:
                clrlim1, clrlim2 = clrlim2, clrlim1 
            vals[:, 0] = np.linspace(clrlim1[0],clrlim2[0], Ncolors)
            vals[:, 1] = np.linspace(clrlim1[1],clrlim2[1], Ncolors)
            vals[:, 2] = np.linspace(clrlim1[2],clrlim2[2], Ncolors)
            newcmp = ListedColormap(vals)
            locs = pattern_index.patternID[pat]
            print(pat, sq, locs)
            _ = plot_grid(from_pattern=draw_listed_pattern, numSq=sq, spot_locs=locs, spot_values=[1], grid=[grid_size,grid_size], ax=axins, vmin=0, vmax=1, cmap=newcmp,)
            # from_pattern=True, numSq=15, spot_locs=[], spot_values=[], grid=[24,24], ax=None, vmin=0, vmax=1, cmap='gray', locs_is_patternID=False, add_colorbar=False,)
            # draw a box around the inset
            axins.add_patch(plt.Rectangle((0,0), grid_size-0.5, grid_size-0.5, fill=False, edgecolor=sq_color, lw=2))
            axins.set_xlim([0,grid_size])
            axins.set_ylim([0,grid_size])

    return ax, insets