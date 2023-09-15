import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
df = pd.read_hdf( "model_params.h5" )


print( df.shape )
#print( df )
#print( df.sort_values( by= ['exc', 'numSq'] ) )
colNames = df.columns
print( colNames[4:] )
'''
colNames = list( colNames[4:] )
# df.boxplot( column = colNames, by=['exc', 'numSq'], layout = ( 4, 2 ), autorange = True )
for nn in colNames:
    #df2 = df.loc[(df["cell"] == 7492) & (df["exc"] == 1) && (df['numSq'] == 15)]
    df2 = df.loc[(df["exc"] == 0) & (df['numSq'] == 15)]
    df2.boxplot( column = nn, by=['cell', 'pattern'] )

#for nn in colNames:
    #df.boxplot( column = nn, by=['exc'] )

plt.show()
'''
plt.rcParams.update( {"font.size": 18} )
params = ['Ca_bind_RR.Kd', 'Ca_bind_RR.tau', 'docking.Kf', 
        'vesicle_release.Kf', 'remove.Kf', 'replenish_vesicle.tau', 
        'vesicle_pool.concInit', 'ligand_binding.tau', 'ligand_binding.Kd']
fig1 = plt.figure( figsize = ( 12, 12 ), layout = "tight" )
fig1.suptitle( "Parameter Distributions" )
fig2 = plt.figure( figsize = ( 12, 12 ), layout = "tight" )
fig2.suptitle( "Dependence on synaptic strength" )
for idx, pp in enumerate( params ):
    #sb.violinplot( data = df.loc[df[''] == foo], x = 'cell', y = pp, inner = 'point' )
    #sb.violinplot( data = df, x = 'cell', y = pp, inner = 'point' )
    plt.figure( fig1 )
    plt.subplot( 5, 2, idx+1 )
    ax = sb.violinplot( data = df, x = 'cell', y = pp, hue='exc', inner = 'point', legend = None )
    ax.legend_.remove()
    if idx > 4:
        ax.get_xaxis().set_ticklabels( ["1", "2", "3", "4", "5", "6"] )
    else:
        ax.get_xaxis().set_visible( False )
    #ax0 = sb.violinplot( data = df.loc[df['exc']==0], x = 'cell', y = pp, inner = 'point' )
    #ax1 = sb.violinplot( data = df.loc[df['exc']==1], x = 'cell', y = pp, inner = 'point' )
    plt.figure( fig2 )
    plt.subplot( 5, 2, idx+1 )
    #ax = sb.scatterplot( data = df, x = 'refPk', y = pp, hue = 'exc', legend = False )
    if idx <= 5:
        ax.get_xaxis().set_visible( False )

plt.show()




#['initScore', 'finalScore', 'Ca_bind_RR.Kd', 'Ca_bind_RR.tau', 'docking.Kf', 'vesicle_release.Kf', 'remove.Kf', 'replenish_vesicle.tau', 'vesicle_pool.concInit', 'refPk', 'firstPk']
