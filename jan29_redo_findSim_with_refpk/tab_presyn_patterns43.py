import sys
import os
import pandas
import argparse
import multiprocessing
import hoss

sampleRate = 20000.0
fracVesicleRelease = 1.0  # frac of vesicle pool released in first pulse.
presynModelNumber = 75
transmitterDecayTime = 5e-3 # 5 ms for sure.
transmitterBaseline = 0.0   # Approaches it
hossParams = '"Ca_bind_RR.Kd", "Ca_bind_RR.tau", "docking.Kf", "vesicle_release.Kf", "remove.Kf", "replenish_vesicle.tau", "vesicle_pool.concInit"'
numHossParams = len( hossParams.split(",") ) + 1

class EvalFunc:
    def __init__( self, cell, celldf, exc, numSq, pattern, args ):
        self.cell = cell
        self.celldf = celldf
        self.exc = exc
        self.numSq = numSq
        self.pattern = pattern
        self.pkDelay = args.peakDelay
        self.presynModelNumber = args.presynModelNumber
        #self.matchMinima = args.matchMinima
        self.matchMinima = True
        self.algorithm = args.algorithm
        #self.doPairedPulse = args.pairedPulse
        self.doPairedPulse = True

    def fitCell( self ):
        curr_proc=multiprocessing.current_process()
        # uncomment following line to get this to work
        curr_proc.daemon=False
        #stpNames = ['p0', 'p1', 'p2','p3', 'p4','p5', 'p6','p7']
        stpNames = ['p{}'.format(ii) for ii in range(9)]
        tpkNames = ['tpk{}'.format(ii) for ii in range(9)]
        freqList = self.celldf['freq'].unique()
        for ff in freqList:
            patdf = self.celldf.loc[
                    (self.celldf['freq']==ff) & 
                    (self.celldf['exc']==self.exc) & 
                    (self.celldf['numSq']==self.numSq) &
                    (self.celldf['pattern']==self.pattern)
                ]
            meanSTP = patdf[stpNames].mean(axis=0)
            meanTpk = patdf[tpkNames].mean(axis=0)
            self.dumpFindSim( meanSTP, meanTpk, ff, True, False ) # for the peaks
            if self.doPairedPulse:
                self.dumpFindSim( meanSTP, meanTpk, ff, True, True ) # for the paired pulse
            if ( self.matchMinima ):
                self.dumpFindSim( meanSTP, meanTpk, ff, False, False ) # for the minima
        self.dumpHoss( freqList, self.matchMinima )
        excLabel = "Exc" if self.exc else "Inh"
        fileSig = "{}_{}_{}_{}".format( self.cell, self.exc, self.numSq, self.pattern )
        fname = "Configs/run{}.json".format( fileSig )
        hoss.main( [fname, "--algorithm", self.algorithm ] )
        df = pandas.read_table( "Results/_opt{}result.txt".format( fileSig ), sep='\s+', skiprows=3, nrows=numHossParams, engine="python", header = None )
        assert(os.path.exists( "Expts/fs_{}_20_pk.json".format(fileSig) ) )
        for ff in freqList:
            os.remove( "Expts/fs_{}_{}_pk.json".format( fileSig, ff ) )
            os.remove( "Expts/fs_{}_{}_pp.json".format( fileSig, ff ) )
            os.remove( "Expts/fs_{}_{}_min.json".format( fileSig, ff ) )
            #print( "Removing Expts/fs_{}_{}_min.json".format( fileSig, ff))
        paramNames = ["cell", "exc", "numSq", "pattern", "initScore", "finalScore"] + list(df.iloc[:,0])
        initScore = 0.0
        finalScore = 0.0
        with open( "Results/_opt{}result.txt".format(fileSig) , "r") as fp:
            last_line = fp.readlines()[-1]
            sp = last_line.replace( ",", " ").split( " " )
            initScore = float(sp[3])
            finalScore = float( sp[-1])
        return paramNames, [self.cell, self.exc, self.numSq, self.pattern, initScore, finalScore] + list( df.iloc[:, 2] )
        '''
        df = pandas.read_table( "Results/_opt{}results.txt".format( fileSig ), sep='\s+', skiprows=3, nrows=8, engine="python", header = None )
        paramNames = ["cell", "exc", "numSq", "pattern"] + list(df.iloc[:,0])
        return paramNames, [self.cell, self.exc, self.numSq, pattern] + list( df.iloc[:, 2] )
        '''

    def ticker( self, arg ):
        # arg should be the return value from the fitcell func
        #print( "{}_{}_{}    ".format( self.cell, self.exc, self.numSq ), end = "" )
        print( ".", end = '' )
        sys.stdout.flush()



    def dumpFindSim( self, meanSTP, meanTpk, freq, isPk, isPP ):
        fileSig = "{}_{}_{}_{}".format( self.cell, self.exc, self.numSq, self.pattern )
        stimAmpl = 50.0
        Ca_basal = 0.08
        stimWidth = 0.002
        pkDelay = self.pkDelay * 0.001
        windowStartt = -0.005 if isPk else 0.005
        windowEndt = 0.005 if isPk else 0.015
        pd = [pkDelay] * 8
        label = "Exc" if self.exc else "Inh"
        pkLabel = "pk" if isPk else "min"
        if isPP:
            pkLabel = "pp"

        transmitter = "glu" if self.exc else "GABA"
        fsStartStr  = '''{{
    "FileType": "FindSim",
    "Version": "1.0",
    "Metadata": {{
        "transcriber": "Upi Bhalla",
        "organization": "NCBS",
        "email": "bhalla@ncbs.res.in",
        "source": {{
            "sourceType": "other",
            "PMID": 0,
            "year": 2022,
            "figure": ""
        }},
        "testModel": "./Models/{}Presyn.g",
        "testMap": "./Maps/mapPresyn.json"
    }},  
    "Experiment": {{
        "design": "TimeSeries",
        "species": "mouse",
        "cellType": "CA1 pyramidal neuron",
        "notes": "Data is from recording of IPSC to 5 or 15 sq optical stim by Aditya Asopa"
    }},  
    "Stimuli": [
        {{
            "timeUnits": "sec",
            "quantityUnits": "uM",
            "entity": "CaInput",
            "field": "conc",
            "data": [
                [ 0, 0.08]'''

        fsMidStr = '''
            ]
        }}
    ],  
    "Readouts": {{
        "timeUnits": "sec",
        "quantityUnits": "ratio",
        "entities": [
            "L_R"
        ],
        "field": "conc",
        "window": {{"startt": {1}, "endt": {2}, "dt": 0.001,
            "operation": "{3}"
        }},
        "normalization": {{
            "entities": [
                "{5}"
            ],
            "sampling": "{4}"
        }},
        "data": ['''.format( transmitter, windowStartt, windowEndt, "max" if isPk else "min", "start", "L_R" if isPk else "vesicle_pool")

        fsEndStr = '\n        ]\n  }\n}\n'

        fname = "Expts/fs_{}_{}_{}.json".format( fileSig, freq, pkLabel )
        settleTime = 1.0
        with open( fname, "w" ) as fp:
            fp.write( fsStartStr.format( label ) )
            for idx in range( len( meanSTP ) ):
                if isPP and idx > 2 :   # Only do first 3 points: ref, p0,p1
                    continue
                t = settleTime if idx == 0 else settleTime + 0.3 + (idx-1)/freq
                fp.write( ",\n                [{:.4f}, {:.3f}],\n".format( t, stimAmpl ) )
                fp.write( "                [{:.4f}, {:.3f}]".format( stimWidth + t, Ca_basal ) )
            fp.write( fsMidStr )
            comma = ""
            #for idx, dd in enumerate( deconv ):
            #print( "LENS = ",  len( meanSTP ), len( meanTpk ) )
            #print( meanSTP )
            for idx, (stp, tpk) in enumerate( zip( meanSTP, meanTpk ) ):
                if isPP and idx > 2:   # Only do first 2 points.
                    continue
                if isPk:
                    fp.write( "{}\n            [{:.4f}, {:.4f}, 0]".format( 
                        comma, tpk + 0.8, 
                        fracVesicleRelease * stp ) )
                else:
                    fp.write( "{}\n            [{:.4f}, {:.4f}, 0]".format( 
                        comma, transmitterDecayTime + tpk + 0.8, 
                        transmitterBaseline ) )
                comma = ","
            fp.write( fsEndStr )

    def dumpHoss( self, freqList, hasMin ):
        excLabel = "Exc" if self.exc else "Inh"
        fileSig = "{}_{}_{}_{}".format( self.cell, self.exc, self.numSq, self.pattern )
        fname = "Configs/run{}.json".format( fileSig )
        exptStr = ''
        comma = "\n                    "
        for ff in freqList:
            exptStr += '{}"fs_{}_{}_pk.json": {{"weight": {} }}'.format(
                comma, fileSig, ff, 200 if ff in [20, 50] else 100 )
            comma = ",\n                    "
            if self.doPairedPulse:
                exptStr += '{}"fs_{}_{}_pp.json": {{"weight": {} }}'.format(
                    comma, fileSig, ff, 200 if ff in [20, 50] else 100 )
            if hasMin:
                exptStr +='{}"fs_{}_{}_min.json": {{"weight": {} }}'.format(
                    comma, fileSig, ff, 50 if ff in [20, 50] else 25 )
    
        startStr = '''{{
    "FileType":"Hoss",
    "Version":"1.0",
    "author": "TabPresyn program",
    "model": "Models/{0}Presyn{1}.g",
    "map":"Maps/mapPresyn.json",
    "exptDir": "./Expts",
    "scoreFunc": "NRMS",
    "tolerance": 0.0001,
    "algorithm": "SLSQP",
    "comment": "Program-generated HOSS config for presynaptic release.",
        "HOSS": [
            {{
            "name": "IndividualPathways", 
            "hierarchyLevel": 1,
            "presyn": {{
                "comment": "This is the only block at present",
                "expt": {{{2}
                }},
                "params": [{3}],
                "paramBounds": {{ "vesicle_pool.concInit":[0.3e-4,10.0e-3,0],
                "docking.Kd":[1e-14, 0.00001,0]
                }},

                "resultFile": "./Results/_opt{4}result.txt",
                "optModelFile": "./Results/opt{4}.g"
            }}
        }}
    ]\n}}'''.format( excLabel, self.presynModelNumber, exptStr, hossParams, fileSig)

        with open( fname, "w" ) as fp:
            fp.write( startStr )

def main():
    global fracVesicleRelease
    parser = argparse.ArgumentParser( description = "Fits presynaptic release model to STP curves from pandas file. Required directories: Configs, Models, Expts, Results, Maps" )
    parser.add_argument( "-o", "--output", type = str, help = "Optional: output pandas hdf file for model params, default = model_params.h5", default = "model_params.h5")
    parser.add_argument( "-f", "--file", type = str, 
            help = "Optional: Name of tabulated STP file in pandas hdf5 format. Default = 'STP_pks_and_refs.h5'", 
            default = "STP_pks_and_refs.h5" )
    parser.add_argument( "-pd", "--peakDelay", type = float, 
            help = "Optional: Delay for time of glu peak in ms. Default = 2", default = 2.0)
    parser.add_argument( '-n', '--numProcesses', type = int, default = 0, help='Optional: Number of processes to spawn' )
    parser.add_argument( '-pmn', '--presynModelNumber', type = int, default = 75, help='Optional: Presynaptic model number' )
    parser.add_argument( '-fvr', '--fracVesicleRelease', type = float, default = 1.0, help='Optional: Fraction of total vesicle count released on first pulse. default 1.0' )
    parser.add_argument( '-m', '--matchMinima', action = "store_true", help='Flag: Turns on dumping of experiments to match minima of trace' )
    parser.add_argument( '-pp', '--pairedPulse', action = "store_true", help='Flag: Turns on dumping of experiments to do paired pulse matching' )
    parser.add_argument( '-a', '--algorithm', type = str, help='Optional: Algorithm name to use, from the set available to scipy.optimize.minimize. Options are CG, Nelder-Mead, Powell, BFGS, COBYLA, SLSQP, TNC. The library has other algorithms but they either require Jacobians or they fail outright. There is also L-BFGS-B which handles bounded solutions, but this is not needed here because we already take care of bounds. COBYLA works well and is the default.', default = "COBYLA" )

    args = parser.parse_args()
    dat = pandas.read_hdf( args.file )
    cellList = dat['cell'].unique()
    print( "Cell list = ", cellList, len( cellList ) )

    numProcesses = args.numProcesses

    if numProcesses == 0:
        numProcesses = multiprocessing.cpu_count() // 4
    if numProcesses == 0:
        numProcesses = 1
    numProcesses = min( numProcesses, len( cellList ) * 4 )
    fracVesicleRelease = args.fracVesicleRelease

    print( "NUM PROCESSES = ", numProcesses )
    pool = multiprocessing.Pool( processes = numProcesses )
    params = []
    ret = []
    for cell in cellList:
        for exc in [ 0, 1 ]:
            for numSq in [ 5, 15 ]:
                print( "cell={} exc={} numSq={}".format( cell, exc, numSq ))
                celldf = dat.loc[dat['cell']==cell]
                patterns = celldf.loc[celldf['numSq']==numSq]['pattern'].unique()
                for pp in patterns:
                    ev = EvalFunc( cell, celldf, exc, numSq, pp, args )
                    ret.append( pool.apply_async( ev.fitCell, callback=ev.ticker ))
    ans = [ rr.get() for rr in ret ]

    for aa in ans:
        [paramNames, paramVals] = aa
        params.append( paramVals )
    df = pandas.DataFrame( params, columns = paramNames )
    df.info()
    df.to_hdf( args.output, "w" )

    #pool.join()
    pool.close()

if __name__ == "__main__":
    main()
