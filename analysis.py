"""
Created on Friday 12th March 2021

@author: Aditya Asopa, Bhalla Lab, NCBS
"""

import sys
import os
import pathlib
import importlib # FIXME: deprecated module, replace with importlib

from eidynamics             import ephys_classes
from eidynamics.errors      import *
from eidynamics.plot_maker   import dataframe_to_plots

def analyse_cell(cell_directory, load_cell=True, save_pickle=True, add_cell_to_database=False, export_training_set=False, save_plots=True):
    cell_directory = pathlib.Path(cell_directory)
    print(120*"-","\nAnalyzing New Cell from: ",cell_directory)
    
    if load_cell:
        cellID = cell_directory.stem
        cell_pickle_file = cell_directory / str(cellID+".pkl")
    else:
        cell_pickle_file = ''
    
    try:
        fileExt = "rec.abf"
        recFiles = list( cell_directory.glob('*'+fileExt) )
        for recFile in recFiles:
            print("Now analysing: ", recFile.name)
            cell,cell_pickle_file,cell_response_file = analyse_recording(recFile, cell_file=cell_pickle_file)

        print('Now generating expected traces.')
        cell.generate_expected_traces()

        if add_cell_to_database:
            cell.add_cell_to_xl_db()

        if export_training_set:
            print("Saving traces for training")
            cell.save_training_set(cell_directory)

        if save_pickle:
            cellFile            = cell_directory / str(str(cell.cellID) + ".pkl" )
            cellFile_csv        = cell_directory / str(str(cell.cellID) + ".xlsx")
            ephys_classes.Neuron.saveCell(cell, cellFile)
            cell.response.to_excel(cellFile_csv)
        else:
            cellFile = ''
        
    except UnboundLocalError as err:
        print("Check if there are '_rec' labeled .abf files in the directory.")

    if save_plots:        
        dataframe_to_plots(cell_pickle_file, ploty="PeakRes",  gridRow="NumSquares", plotby="EI",        clipSpikes=True)
        dataframe_to_plots(cell_pickle_file, ploty="PeakRes",  gridRow="NumSquares", plotby="PatternID", clipSpikes=True)
        dataframe_to_plots(cell_pickle_file, ploty="PeakRes",  gridRow="PatternID",  plotby="Repeat",    clipSpikes=True)

        dataframe_to_plots(cell_pickle_file, ploty="PeakTime", gridRow="NumSquares", plotby="EI",        clipSpikes=True)
        dataframe_to_plots(cell_pickle_file, ploty="PeakTime", gridRow="NumSquares", plotby="PatternID", clipSpikes=True)
        dataframe_to_plots(cell_pickle_file, ploty="PeakTime", gridRow="PatternID",  plotby="Repeat",    clipSpikes=True)

        dataframe_to_plots(cell_pickle_file, ploty="AUC",      gridRow="NumSquares", plotby="EI",        clipSpikes=True)
        dataframe_to_plots(cell_pickle_file, ploty="AUC",      gridRow="NumSquares", plotby="PatternID", clipSpikes=True)
        dataframe_to_plots(cell_pickle_file, ploty="AUC",      gridRow="PatternID",  plotby="Repeat",    clipSpikes=True)

    os.remove(cell_pickle_file) # remove temporary pickle and excel file
    os.remove(cell_response_file)    
    return cell, cellFile

def analyse_recording(recording_file, load_cell=True, cell_file=''):
    datafile      = pathlib.Path(recording_file)
    exptDir       = datafile.parent
    exptFile      = datafile.name
    fileID        = exptFile[:15]
    parameterFilePath = exptDir / str(fileID + "_experiment_parameters.py")
    paramfileName = parameterFilePath.stem
    parameterFilePath = str(parameterFilePath.parent) # putting a str() converts it from a pathlib windowspath object to a palin string
    
    # Import Experiment Variables
    try:
        print("Looking for experiment parameters locally")
        sys.path.append(parameterFilePath)
        exptParams = importlib.import_module(paramfileName, parameterFilePath)
        if not exptParams.datafile == exptFile:
            raise FileMismatchError()
        print('Experiment parameters loaded from: ', parameterFilePath)
    except (FileMismatchError,FileNotFoundError) as err:
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
        coordfile       = os.path.join(os.getcwd(), "polygonProtocols", coordfileName)
        coordfile       = pathlib.Path.cwd() / "polygonProtocols" / coordfileName
        # os.path.isfile(coordfile)
        print('Local coord file loaded from: ', coordfile)
    except FileNotFoundError:
        print('No coord file found, probably there isn\'t one')
        coordfile       = ''
    except Exception as err:
        print(err)
        coordfile       = ''


    if (cell_file is not '') & (load_cell is True):
        try:
            print('Loading local cell data')
            cell = ephys_classes.Neuron.loadCell(cell_file)
        except:
            print('Error in loading from file. Creating new cell.')
            cell = ephys_classes.Neuron(exptParams)
    else:
        print('Creating new cell.')
        cell = ephys_classes.Neuron(exptParams)

    cell.addExperiment(datafile=datafile, coordfile=coordfile, exptParams=exptParams)

    cell_file_pickle     = exptDir / str(str(exptParams.cellID) + "_temp.pkl")
    cell_file_csv        = exptDir / str(str(exptParams.cellID) + "_temp.xlsx")

    ephys_classes.Neuron.saveCell(cell, cell_file_pickle)
    cell.response.to_excel(cell_file_csv)

    return cell, cell_file_pickle, cell_file_csv

if __name__ == "__main__":
    input_address = pathlib.Path(sys.argv[1]) 
    if input_address.is_dir():
        analyse_cell(input_address, load_cell=True, save_pickle=False, add_cell_to_database=False, export_training_set=False, save_plots=True)
    else:
        analyse_recording(input_address, load_cell=True, cell_file='')
else:
    print("Programme accessed from outside")