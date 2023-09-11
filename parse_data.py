"""
Created on Friday 12th March 2021

@author: Aditya Asopa, Bhalla Lab, NCBS
"""

import sys
import os
import pathlib
import importlib

from eidynamics             import ephys_classes
from eidynamics             import data_quality_checks
from eidynamics.errors      import *
from eidynamics.plot_maker  import dataframe_to_plots
from eidynamics.utils       import show_experiment_table, reset_and_print, parse_other_experiment_param_file


def parse_cell(cell_directory, load_cell=True, save_pickle=True, add_cell_to_database=False, all_cell_response_db='', export_training_set=True, save_plots=True, user='Adi'):
    cell_directory = pathlib.Path(cell_directory)
    cell_response_file, cell_pickle_file, cellFile = "", "", ""
    cellID = cell_directory.stem
    cell = None
    # _ = show_experiment_table(cell_directory)

    if load_cell:
        cell_pickle_file = cell_directory / str(cellID + ".pkl")
        try:
            print("#___ Loading cell from: ", cell_pickle_file)
            cell = ephys_classes.Neuron.loadCell(cell_pickle_file)
        except:
            print(f"#___ Error in loading cell from {cell_pickle_file}. Creating new cell.")
            cell_pickle_file = ''
            cell = None

    try:
        file_ext = "rec.abf" if user=='Adi' else '.abf'
        rec_files = list(cell_directory.glob('*' + file_ext))
        # print all rec files one by one
        print("Following are the recordings in the directory: ")
        print(rec_files)

        for i, rec_file in enumerate(rec_files):
            msg = "Now parsing: " + rec_file.name
            reset_and_print(i, len(rec_files), clear=False, message=msg)
            cell, cell_pickle_file, cell_response_file = parse_recording(rec_file, cell_file=cell_pickle_file, user=user)
        print(f"Following is the summary of all experiments on the cell {cellID} ")
        _ = cell.summarize_experiments()

        print('##__ Making Dataframe.')
        cell.make_dataframe() # type: ignore

        print('###_ Running cell stability checks')
        data_quality_checks.run_qc(cell, cell_directory)
        '''
        if add_cell_to_database:
            if all_cell_response_db == '':
                # create a new database file
                all_cell_response_db = cell_directory / "all_cells_response.xlsx"
                
            print("#### Adding cell to database file: ", all_cell_response_db)
            cell.add_cell_to_xl_db(all_cell_response_db) # type: ignore
        '''
        if export_training_set:
            print("#### Saving traces for training for each protocol")
            cell.save_full_dataset(cell_directory) # type: ignore

        if save_pickle:
            print("#### Saving pickle and excel files")
            cellFile            = cell_directory / str(str(cell.cellID) + ".pkl") # type: ignore
            cellFile_csv        = cell_directory / str(str(cell.cellID) + ".xlsx") # type: ignore
            ephys_classes.Neuron.saveCell(cell, cellFile)
            cell.response.to_excel(cellFile_csv) # type: ignore
        else:
            cellFile = ''

    except UnboundLocalError as err:
        print(err)
        print("Check if there are '_rec' labeled .abf files in the directory.")

    if save_plots:
        dataframe_to_plots(cell_pickle_file, ploty="PeakRes",  gridRow="NumSquares",
                                             plotby="EI",        clipSpikes=True)
        dataframe_to_plots(cell_pickle_file, ploty="PeakRes",  gridRow="NumSquares",
                                             plotby="PatternID", clipSpikes=True)
        dataframe_to_plots(cell_pickle_file, ploty="PeakRes",  gridRow="PatternID",
                                             plotby="Repeat",    clipSpikes=True)

        dataframe_to_plots(cell_pickle_file, ploty="PeakTime", gridRow="NumSquares",
                                             plotby="EI",        clipSpikes=True)
        dataframe_to_plots(cell_pickle_file, ploty="PeakTime", gridRow="NumSquares",
                                             plotby="PatternID", clipSpikes=True)
        dataframe_to_plots(cell_pickle_file, ploty="PeakTime", gridRow="PatternID",
                                             plotby="Repeat",    clipSpikes=True)

        dataframe_to_plots(cell_pickle_file, ploty="AUC",      gridRow="NumSquares",
                                             plotby="EI",        clipSpikes=True)
        dataframe_to_plots(cell_pickle_file, ploty="AUC",      gridRow="NumSquares",
                                             plotby="PatternID", clipSpikes=True)
        dataframe_to_plots(cell_pickle_file, ploty="AUC",      gridRow="PatternID",
                                             plotby="Repeat",    clipSpikes=True)

    os.remove(cell_pickle_file)  # remove temporary pickle and excel file
    os.remove(cell_response_file)
    return cell, cellFile


def parse_recording(recording_file, load_cell=True, cell_file=None, user='Adi'):
    datafile      = pathlib.Path(recording_file)
    exptDir       = datafile.parent
    exptFile      = datafile.name
    fileID        = exptFile[:15]

    parameterFilePath = ''
    paramfileName = ''
    if user=='Adi':
        parameterFilePath = exptDir / str(fileID + "_experiment_parameters.py")
        paramfileName = parameterFilePath.stem
        parameterFilePath = str(parameterFilePath.parent) # str() to convert pathlib path --> plain string
    elif user=='Sulu':
        parameterFilePath = exptDir / str("PPFExpt_protocol_params_March06_23_" + datafile.stem + ".py")
    
    print("ParameterFilePath: ", parameterFilePath)    
    
    # Import Experiment Variables
    try:
        print("Looking for experiment parameters locally")
        sys.path.append(parameterFilePath)
        
        if user=='Adi':
            exptParams = importlib.import_module(paramfileName, parameterFilePath)
        elif user=='Sulu':
            exptParams = parse_other_experiment_param_file(parameterFilePath, user=user)

        datafile_according_to_epfile = exptParams.datafile
        if not datafile_according_to_epfile == exptFile:
            raise FileMismatchError()
        print('Experiment parameters loaded from: ', parameterFilePath)

    except (FileMismatchError, FileNotFoundError) as err:
        print(err)
        print("No special instructions, using default variables.")
        import eidynamics.experiment_parameters_default as exptParams
        save_trial  = False
        print('Default Experiment Parameters loaded.\n'
              'Experiment will not be added to the cell pickle file,\n'
              'only excel file of cell response will be created.')
    except Exception as err:
        print(err)
        print("Experiment Parameters error. Quitting!")
        sys.exit()

    # Import stimulation coordinates
    try:
        coordfileName   = exptParams.polygonProtocol
        if not coordfileName:
            raise FileNotFoundError
        coordfile       = pathlib.Path.cwd() / "polygonProtocols" / coordfileName
        print('Local coord file loaded from: ', coordfile)
    except FileNotFoundError:
        print('No coord file found, probably there isn\'t one')
        coordfile       = ''
    except Exception as err:
        print(err)
        coordfile       = ''

    # Load or initialize a Neuron object
    if (cell_file != '') & (load_cell == True):
        try:
            print('Loading local cell data')
            cell = ephys_classes.Neuron.loadCell(cell_file)
        except:
            print('Error in loading from file. Creating new cell.')
            cell = ephys_classes.Neuron(exptParams)
    else:
        print('Creating new cell.')
        cell = ephys_classes.Neuron(exptParams)

    # Add current experiment to the cell object
    print(f'Adding experiment {fileID} to the cell.')
    cell.addExperiment(datafile=datafile, coordfile=coordfile, exptParams=exptParams)

    # Create temporary file to save
    cell_file_pickle     = exptDir / str(str(exptParams.cellID) + "_temp.pkl")
    cell_file_csv        = exptDir / str(str(exptParams.cellID) + "_temp.xlsx")

    # Save the neuron object
    ephys_classes.Neuron.saveCell(cell, cell_file_pickle)
    cell.response.to_excel(cell_file_csv)

    return cell, cell_file_pickle, cell_file_csv


if __name__ == "__main__":
    input_address = pathlib.Path(sys.argv[1])
    
    user = sys.argv[2]
    print(user, "Input address: ", input_address)
    if input_address.is_dir():
        parse_cell(input_address, load_cell=True, save_pickle=True,
                                    add_cell_to_database=True, export_training_set=True,
                                    save_plots=False, user=user)
    else:
        parse_recording(input_address, load_cell=True, cell_file='', user=user)
else:
    print("Data parsing program imported")
