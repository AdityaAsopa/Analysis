import os
import sys
import pathlib
import analysis
from eidynamics.plot_maker  import dataframe_to_plots
from eidynamics             import ephys_classes
from all_cells              import *


def batch_analysis(cellDirectory, add_cell_to_database=False, export_training_set=False, save_plots=False):
    _,savedCellFile = analysis.analyse_cell(cellDirectory,
                                        load_cell=True,
                                        save_pickle=True,
                                        add_cell_to_database = add_cell_to_database,
                                        export_training_set = export_training_set,
                                        save_plots = save_plots)

    return savedCellFile

def batch_plot(cellFile):
    try:
        dataframe_to_plots(cellFile, ploty="PeakRes",  gridRow="NumSquares", plotby="EI",        clipSpikes=True)
        dataframe_to_plots(cellFile, ploty="PeakRes",  gridRow="NumSquares", plotby="PatternID", clipSpikes=True)
        dataframe_to_plots(cellFile, ploty="PeakRes",  gridRow="PatternID",  plotby="Repeat",    clipSpikes=True)

        dataframe_to_plots(cellFile, ploty="PeakTime", gridRow="NumSquares", plotby="EI",        clipSpikes=True)
        dataframe_to_plots(cellFile, ploty="PeakTime", gridRow="NumSquares", plotby="PatternID", clipSpikes=True)
        dataframe_to_plots(cellFile, ploty="PeakTime", gridRow="PatternID",  plotby="Repeat",    clipSpikes=True)

        dataframe_to_plots(cellFile, ploty="AUC",      gridRow="NumSquares", plotby="EI",        clipSpikes=True)
        dataframe_to_plots(cellFile, ploty="AUC",      gridRow="NumSquares", plotby="PatternID", clipSpikes=True)
        dataframe_to_plots(cellFile, ploty="AUC",      gridRow="PatternID",  plotby="Repeat",    clipSpikes=True)
    except:
        pass

def meta_analysis(cellFile):
    pass

def meta_plot(allCellsFile):
    pass

if __name__ == "__main__":
    if "analyse" in sys.argv:
        print("Analysing all catalogued cells recordings...")

        for cellDirectory in all_cells:
            savedCellFile = batch_analysis((project_path_root / cellDirectory),add_cell_to_database=True, export_training_set=True, save_plots=True)
            print("Data saved in cell file: ",savedCellFile)

    elif "codetest" in sys.argv:
        print("Checking if analysis pipline is working...")
        for cellDirectory in test_cells:
            savedCellFile = batch_analysis(cellDirectory,add_cell_to_database=False, export_training_set=True, save_plots=True)
            print(savedCellFile)
        print('All Tests Passed!')
        
    else:
        for cellDirectory in all_cells:
            try:
                print("Looking for analysed cell pickles to plot directly from...")
                cf = [os.path.join(cellDirectory, pickleFile) for pickleFile in os.listdir(cellDirectory) if pickleFile.endswith("cell.pkl")]
                print("Plotting from: ",cf[0])
                batch_plot(cf[0])
            except FileNotFoundError:
                print("Cell pickle not found. Beginning analysis.")
                savedCellFile = batch_analysis((project_path_root / cellDirectory),add_cell_to_database=True, export_training_set=True, save_plots=True)
                print("Data saved in cell file: ",savedCellFile)
                batch_plot(savedCellFile)
