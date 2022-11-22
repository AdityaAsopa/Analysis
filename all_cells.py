from pathlib import Path
import sys

# These cells have been passed through the data parsing pipeline by 6 Jan 2022
all_cells = ["Lab\\Projects\\EI_Dynamics\\Data\\22-06-03_GrikAA014\\141\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\22-06-01_GrikAA011\\111\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\22-04-18_G749\\7492\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\22-04-18_G749\\7491\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-12-31_G620\\6201\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-12-28_G630\\6301\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-12-16_G550\\5501\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-12-16_G550\\5502\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-12-15_G562\\5621\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-11-10_G561\\5611\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-09-23_G529\\5291\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-09-07_G521\\5212\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-09-07_G521\\5211\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-07-29_G388\\3882\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-07-30_G387\\3872\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-07-30_G387\\3871\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-08-11_G379\\3791\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-08-10_G378\\3781\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-07-09_G353\\3531\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-06-04_G340\\3402\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-06-04_G340\\3401\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-06-09_G337\\3373\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-05-18_G320\\3201\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-05-12_G319\\3192\\",            
            "Lab\\Projects\\EI_Dynamics\\Data\\21-04-16_G294\\2941\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-04-01_G251\\2511\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-04-02_G250\\2501\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-03-06_G234\\2341\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-03-06_G234\\2342\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-03-04_G233\\2331\\"]


test_cells = [".\\testExamples\\testCells\\5211\\",
              ".\\testExamples\\testCells\\3882\\"]
# ".\\testExamples\\testCells\\111\\",
              
# convergence expt, not yet explored
other_cells = ["Lab\\Projects\\EI_Dynamics\\Data\\21-07-29_G388\\3881\\"]

all_cells_response_file = "Lab\\Projects\\EI_Dynamics\\AnalysisFiles\\allCells.xlsx"

##############################################################################
NCBS_data_path  = Path( r"\\storage.ncbs.res.in\adityaa\\" )
cloud_data_path = Path( "C:\\Users\\adity\\OneDrive\\NCBS\\" )
rig_data_path   = Path( "C:\\Users\\aditya\\OneDrive\\NCBS\\" )
laddu_data_path = Path( "C:\\Users\\user\\OneDrive\\NCBS\\")
test_data_path  = Path( '.\\' )

if NCBS_data_path.exists():
    project_path_root = NCBS_data_path
elif laddu_data_path.exists():
    project_path_root = laddu_data_path
elif rig_data_path.exists():
    project_path_root = rig_data_path
elif cloud_data_path.exists():
    project_path_root = cloud_data_path
else:
    print('Data path error. Testing code on test cells.')
    raise FileNotFoundError