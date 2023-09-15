import os
import sys
import datetime
import pathlib
import argparse
import importlib
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull

import parse_data
from eidynamics.plot_maker  import dataframe_to_plots
from eidynamics             import ephys_classes
from eidynamics.utils       import *
import generate_screening_param_figures
import collate_dataset

def batch_analysis(cellDirectory, add_cell_to_database=False, all_cell_response_db='', export_training_set=False, save_plots=False, user='Adi'):
    _,savedCellFile = parse_data.parse_cell(cellDirectory,
                                        load_cell=True,
                                        save_pickle=True,
                                        add_cell_to_database = add_cell_to_database,
                                        all_cell_response_db = all_cell_response_db,
                                        export_training_set = export_training_set,
                                        save_plots = save_plots,
                                        user=user)

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

def main(args):    
    all_cells_filename = pathlib.Path(args.file).stem
    all_cells_filepath = pathlib.Path(args.file)
    cell_list = importlib.import_module(all_cells_filename, all_cells_filepath) # type: ignore
    project_path_root = cell_list.project_path_root
    all_cell_response_file = project_path_root / cell_list.all_cells_response_file

    make_plots = True if args.plot else False
    print("Make Plots = ", make_plots)
        
    # user
    user = ''
    if args.adi:
        user = 'Adi'
    elif args.sulu:
        user = 'Sulu'


    if args.analyse:
        all_cells = cell_list.all_cells
        failed_cells = []
        parsed_cells = []
        project_path_root = cell_list.project_path_root
        print("Analysing all catalogued cells recordings...")
        for i, cellDirectory in enumerate(all_cells):
            msg = 'Analysing cell from: ' + cellDirectory
            # reset_and_print(i, len(all_cells), clear=True, message=msg)
            try:
                savedCellFile = batch_analysis((project_path_root / cellDirectory),add_cell_to_database=True, all_cell_response_db=all_cell_response_file, export_training_set=True, save_plots=False, user=user)
                print("Data saved in cell file: ",savedCellFile)
                # Batch plot
                #batch_plot(savedCellFile)
                parsed_cells.append(cellDirectory)
            except Exception as err:
                print(err)
                print("@@@@@@  Some error with this cell. Moving on to the next one.")
                failed_cells.append(cellDirectory)
        
        print('&'*180)
        print("failed to process following cells: ")
        print(failed_cells)

        # print parsed cells from the list one by one, with each cell in a new line
        print('$'*180)
        print("successfully processed following cells: ")
        print(parsed_cells)

        # collate all the data from the parsed cells into a single dataframe
        protocols = ['FreqSweep','LTMRand','SpikeTrain','surprise','convergence', 'grid']
        collate_dataset.main(cell_set = parsed_cells, protocols = protocols)

        # make data quality plots for all_cells data
        #generate_screening_param_figures.main()

        

    elif args.test:
        test_cells = cell_list.test_cells
        print("Checking if analysis pipeline is working...")
        for cellDirectory in test_cells:
                savedCellFile = batch_analysis(cellDirectory,add_cell_to_database=False, export_training_set=True, save_plots=make_plots, user=user)
                print(savedCellFile)

        # make data quality plots for all_cells data
        print("generating test plots")
        generate_screening_param_figures.test()
        print('All Tests Passed!')

    else:
        all_cells = cell_list.all_cells
        for cellDirectory in all_cells:
            try:
                print("Looking for analysed cell pickles to plot directly from...")
                cf = [os.path.join(cellDirectory, pickleFile) for pickleFile in os.listdir(cellDirectory) if pickleFile.endswith("cell.pkl")]
                print("Plotting from: ",cf[0])
                batch_plot(cf[0])
            except FileNotFoundError:
                print("Cell pickle not found. Beginning analysis.")
                savedCellFile = batch_analysis((project_path_root / cellDirectory),add_cell_to_database=True, all_cell_response_db=all_cell_response_file, export_training_set=True, save_plots=True, user=user)
                print("Data saved in cell file: ",savedCellFile)
                batch_plot(savedCellFile)

"""Argument Parser"""

parser = argparse.ArgumentParser(description="Run the main analysis program on a list of cells.")

parser.add_argument("file",   help="Required, python file that has list of selected cells to run batch analysis")

parser.add_argument( "-q", "--quiet", action="store_true", help="Flag to turn off printout to the terminal")

process_group  = parser.add_mutually_exclusive_group()
process_group.add_argument("-t", "--test",    action="store_true", help="Flag to run a code test")
process_group.add_argument("-a", "--analyse", action="store_true", help="Flag to run batch analysis")

user_group  = parser.add_mutually_exclusive_group()
user_group.add_argument("-aa", "--adi",    action="store_true", help="Flag to run Aditya's parsing code")
user_group.add_argument("-sm", "--sulu", action="store_true", help="Flag to run Sulu's parsing code")

parser.add_argument("-p", "--plot",   action="store_true", help="to display plots")

args = parser.parse_args()

# to suppress the print statements, if quiet flag == True                
@contextmanager 
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

start_time = datetime.datetime.now()
print("Start:", str(start_time))

if not args.quiet:
    main(args)
else:
    with suppress_stdout_stderr():
        main(args)

print('All Done!')

stop_time = datetime.datetime.now()
print("Start:", str(start_time))
print("Stop:", str(stop_time))
print("Batch analysis took {} to run".format(stop_time-start_time) )