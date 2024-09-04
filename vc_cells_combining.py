from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import psutil

df1 = pd.read_hdf('screened_cells_FreqSweep_VC_channelr_long_separated_part1.h5', key='data')
df2 = pd.read_hdf('screened_cells_FreqSweep_VC_channelr_long_separated_part2.h5', key='data')

df3 = pd.concat([df1,df2], axis=0, ignore_index=True).reset_index()

df3.to_hdf('screened_cells_FreqSweep_VC_channelr_long_separated_combined.h5', key='data')