import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats    import kruskal, wilcoxon, mannwhitneyu, ranksums

significance_asterisks = {1.0:'n.s.', 0.05:'*', 0.01:'**', 0.001:'***', 0.0001:'****'}

def annotate_stat_stars(ax, pval, alpha=0.05, star_loc=[0.5,0.5], add_line=True, line_locs=[0,1,2,2], offset_btw_star_n_line=0.1, color='k', coord_system='axes', fontsize=12, add_n=False,**kwargs):
    annot_text = f'p={pval:.2f}'
    for k in significance_asterisks.keys():
        if pval > k:
            break
        else:
            annot_text = significance_asterisks[k]
     
    if coord_system == 'axes':    
        # add text annotation on the axis
        ax.text(star_loc[0], star_loc[1], annot_text, color=color, fontsize=fontsize, ha='center', transform=ax.transAxes, **kwargs)
    else:
        ax.text(star_loc[0], star_loc[1], annot_text, color=color, fontsize=fontsize, ha='center', **kwargs)

    ## add a line to connect the two groups for which annotation is added
    # also add two tiny lines at the end of the main line
    off = offset_btw_star_n_line
    if add_line:
        x0, x1 = line_locs[0], line_locs[1]
        y0, y1 = line_locs[2], line_locs[3]
        yaxis_extent = np.diff(ax.get_ylim())[0]
        y1a, y1b = y1-0.01*yaxis_extent, y1+0.01*yaxis_extent
        # if coord_system == 'axes', draw a line in axes coordinates else data coordinates
        if coord_system == 'axes':
            ax.plot([x0, x1], [y0, y1], transform=ax.transAxes, color=color, linewidth=1)
            ax.plot([x0, x0], [y1a, y1b], transform=ax.transAxes, color=color, linewidth=1)
            ax.plot([x1, x1], [y1a, y1b], transform=ax.transAxes, color=color, linewidth=1)
        else:
            ax.plot([x0, x1], [y0, y1], color=color, linewidth=1)
            ax.plot([x0, x0], [y1a, y1b], color=color, linewidth=1)
            ax.plot([x1, x1], [y1a, y1b], color=color, linewidth=1)


def pairwise_annotate_violin_plot(ax, df, x='', y='', stat=wilcoxon, add_line=False, offset=0.1, color='grey', coord_system='axes', fontsize=12, add_n=False, annotate='both', **kwargs ):
    '''
    This function annotates a violin plot with pairwise statistical significance values.
    The function assumes that the x-axis is a categorical variable and the y-axis is a continuous variable.
    '''
    
    unique_values = np.unique(df[x])
    labels = ax.get_xticklabels()
    num_violins = len(labels)
    violin_locs = {int(label.get_text()):label.get_position() for label in labels}
    print(violin_locs)
    counter = 0
    pvalues = {}
    nvalues = {}
    original_ylim = ax.get_ylim()[1]
    for i in unique_values:
        for j in unique_values:
            if i < j:
                xipos, xjpos = violin_locs[i][0], violin_locs[j][0]
                xpos = (xipos + xjpos) / 2
                ypos = (0.9 + counter * 0.05) * original_ylim
                print(xpos, ypos, counter)
                counter += 1
                sample1, sample2 = df[df[x] == i], df[df[x] == j]
                # count nans in sample1 and sample2
                nans1, nans2 = sample1[y].isna().sum(), sample2[y].isna().sum()
                _, pval = stat(sample1[y], sample2[y])
                if annotate == 'both': # annotate both significant and non-significant pvals
                    print(sample1.shape[0], sample2.shape[0], pval, xpos, ypos, )
                    annotate_stat_stars(ax, pval, star_loc=[xpos, ypos], add_line=True, line_locs=[xipos, xjpos, ypos, ypos], offset_btw_star_n_line=offset, color=color, coord_system=coord_system, fontsize=12, zorder=10)
                elif annotate == 'significant': # annotate only significant pvals
                    if pval < 0.05:
                        annotate_stat_stars(ax, pval, star_loc=[xpos, ypos], add_line=True, line_locs=[xipos, xjpos, ypos, ypos], offset_btw_star_n_line=offset, color=color, coord_system=coord_system, fontsize=12, zorder=10)
                elif annotate == 'non-significant': # annotate only non-significant pvals
                    if pval >= 0.05:
                        annotate_stat_stars(ax, pval, star_loc=[xpos, ypos], add_line=True, line_locs=[xipos, xjpos, ypos, ypos], offset_btw_star_n_line=offset, color=color, coord_system=coord_system, fontsize=12, zorder=10)
                pvalues[(i, j)] = pval
                n1, n2 = (len(sample1), len(sample2))
                nvalues[(i, j)] = (n1, n2)
                if add_n:
                    # add annotation of size of the groups
                    ax.text(xpos, ypos, f'(n1={n1}, n2={n2})', color=color, fontsize=fontsize-2, ha='center', coord_system=coord_system, **kwargs)

    return pvalues, nvalues
            

def pairwise_draw_and_annotate_line_plot(ax, df, x='', y='', hue='', draw=True, kind='violin', palette='viridis', split_violins=False, dodge=True, stat_across='hue', stat=kruskal, skip_first_xvalue=True, annotate_wrt_data=False, offset_btw_star_n_line=0.1, color='grey', coord_system='data', fontsize=12, zorder=10, add_n=False):
    ''' This function takes a dataframe, and makes pairwise comparisons between the groups in the hue column
    for each x value. The function then annotates the line plot with the p-values of the comparisons.'''

    if draw:
        # draw the plots
        if kind == 'violin':
            sns.violinplot(data=df, x=x, y=y, hue=hue, palette=palette, ax=ax, alpha=0.8, split=split_violins, inner='quartile', linewidth=1)
        elif kind == 'strip':
            sns.stripplot(data=df, x=x, y=y, hue=hue, palette=palette, ax=ax, alpha=0.8, dodge=dodge,)
        elif kind == 'line':
            sns.lineplot(data=df, x=x, y=y, hue=hue, palette=palette, ax=ax, alpha=0.5, errorbar=('sd', 1), err_style='bars', linewidth=3,err_kws={"elinewidth": 3, 'capsize':5})
        elif kind == 'bar':
            sns.barplot(data=df, x=x, y=y, hue=hue, palette=palette, ax=ax, alpha=0.8, dodge=dodge, ci='sd', errwidth=3, capsize=0.1)
        elif kind=='box':
            sns.boxplot(data=df, x=x, y=y, hue=hue, palette=palette, ax=ax, )
        elif kind=='point':
            sns.pointplot(data=df, x=x, y=y, hue=hue, palette=palette, ax=ax, dodge=dodge, errorbar=('ci', 95) )
        else:
            pass

    ax.legend([],[], frameon=False)
    if stat_across == 'hue':
        # get the unique hue values and x values
        hue_values  = df[hue].unique() # group labels for each x-axis categorical value
        x_values    = df[x].unique() # x-axis categorical value labels
        # set the xticks and xticklabels according to the x-values
        ax.set_xticks(range(len(x_values)), labels=x_values)
        xticks = ax.get_xticks()
        xticklabels = ax.get_xticklabels()
        # get the max value of data across all x and all hue groups
        max_ydata = df[y].max()
        # set ypos to be 0.9*ylim
        ypos = 0.9*ax.get_ylim()[1]

        # for each x-value, get the ygroup values for hue1 and hue2
        for ix, x_val in enumerate(x_values):
            if (ix==0) & (skip_first_xvalue):
                continue
            grouped_df = df[(df[x]==x_val)].groupby(hue)
            group_data = grouped_df[y].apply(list)
            # convert all the group data into a list of lists
            group_data = group_data.values.tolist()
            kruskal_statistic, kruskal_pval = kruskal(*group_data)

            # get the location of x_val on the x-axis of ax
            # get x-ticks and x-tick-labels
            xpos = xticks[ix]

            # get the maximum value of y for the given x_val across all the groups, add the offset to get the ypos for annotation
            if annotate_wrt_data:
                ypos = 1.1* np.max(group_data)

            # convert xpos and ypos into axes coordinate system if coord_system=='axes'
            if coord_system=='axes':
                xpos = ax.transAxes.inverted().transform(ax.transData.transform([xpos, ypos]))[0]
                ypos = ax.transAxes.inverted().transform(ax.transData.transform([xpos, ypos]))[1]
            

            
            annotate_stat_stars(ax, kruskal_pval, star_loc=[xpos, ypos], add_line=False, color=color, coord_system=coord_system, fontsize=12, zorder=10)
            if add_n:
                # add annotation of size of the groups
                ax.text(xpos, ypos, f'(n1={len(df[(df[x] == x_val) & (df[hue] == hue_values[0])])}, n2={len(df[(df[x] == x_val) & (df[hue] == hue_values[1])])})', color=color, fontsize=fontsize-2, ha='center', zorder=10)


        # print(ix, x_val, kruskal_statistic, kruskal_pval, xpos, ypos)


    elif (stat_across == 'x') or (stat_across == x):
        # get the unique hue values and x values
        unique_values = np.unique(df[x])
        xticks = ax.set_xticks(range(len(unique_values)), labels=unique_values)
        xticks = ax.get_xticks()
        labels = ax.get_xticklabels()
        num_violins = len(labels)
        # violin_locs = {int(label.get_text()):label.get_position() for label in labels}
        original_ylim = ax.get_ylim()[1]

        counter = 0
        for ix, i in enumerate(unique_values):
            for jx, j in enumerate(unique_values):
                if ix > jx:
                    # xipos, xjpos = violin_locs[i][0], violin_locs[j][0]
                    xipos, xjpos = xticks[ix], xticks[jx]
                    xpos = (xipos + xjpos) / 2
                    ypos = (1.0 + counter * 0.05) * original_ylim
                    counter += 1
                    _, pval = stat(df[df[x] == i][y], df[df[x] == j][y])
                    annotate_stat_stars(ax, pval, star_loc=[xpos, ypos], add_line=True, line_locs=[xipos, xjpos, ypos, ypos], color=color, coord_system=coord_system, fontsize=12, zorder=10)
                    if add_n:
                        # add annotation of size of the groups
                        ax.text(xpos, ypos, f'(n1={len(df[df[x] == i])}, n2={len(df[df[x] == j])})', color=color, fontsize=fontsize-2, ha='center', zorder=10, **kwargs)

    
    elif stat_across == None:
        pass



