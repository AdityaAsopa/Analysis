# cell directories
import os

NCBSDataPath =   "\\storage.ncbs.res.in\\adityaa\\"
cloudDataPath = "C:\\Users\\adity\\OneDrive\\NCBS\\"
rigDataPath =   "C:\\Users\\aditya\\OneDrive\\NCBS\\"
testDataPath = '.\\'

if os.path.exists(NCBSDataPath):
    projectPathRoot = NCBSDataPath
elif os.path.exists(cloudDataPath):
    projectPathRoot = cloudDataPath
elif os.path.exists(rigDataPath):
    projectPathRoot = rigDataPath
elif os.path.exists('.\\'):
    projectPathRoot = '.\\'
# These cells have been passed through the data parsing pipeline by 6 Jan 2022

allCells = ["Lab\\Projects\\EI_Dynamics\\Data\\21-12-28_G630\\6301\\",
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
            "Lab\\Projects\\EI_Dynamics\\Data\\21-03-04_G233\\2331\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-12-28_G630\\6301\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-12-16_G550\\5501\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-12-16_G550\\5502\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-12-15_G562\\5621\\",
            "Lab\\Projects\\EI_Dynamics\\Data\\21-11-10_G561\\5611\\"]

testCells = [".\\testExamples\\testCells\\5211\\",
             ".\\testExamples\\testCells\\3882\\"]

# convergence expt, not yet explored
otherCells = ["Lab\\Projects\\EI_Dynamics\\Data\\21-07-29_G388\\3881\\"]

allCellsResponseFile = "Lab\\Projects\\EI_Dynamics\\AnalysisFiles\\allCells.xlsx"
