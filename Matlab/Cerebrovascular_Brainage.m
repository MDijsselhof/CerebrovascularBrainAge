%% ExploreASL Cerebrovascular Brain-age
% This wrapper loads ExploreASL imaging data, creates datastructures for input into 
% Python ML, executes the ML and obtains output.

%% admin

addpath('/scratch/radv/mdijsselhof/ExploreASL') % add ExploreASL path to working directory
ExploreASL_Initialize

addpath('/scratch/radv/mdijsselhof/CerebrovascularBrainAge/Matlab') % add Cerebrovascular Brain-age path to working directory
%% Settings
Settings.DataFolder = "/home/radv/mdijsselhof/my-scratch/CerebrovascularBrainAge/Data/"; %% Add studyfolder
Settings.PythonEnvironment = "/home/radv/mdijsselhof/my-scratch/CerebrovascularBrainAge/Python/Scripts"; % Python3 scripts used for ML 
Settings.CondaEnvironmentPath = '/home/radv/mdijsselhof/.conda/envs/CBA/';  % location of Conda environment containing required packages
Settings.MLAlgorithms = ["All"]; % Select Machine Learning algorithms. Options are: ["All", "RandomForest", "DecisionTree", "XGBoost", "BayesianRidge", 
% "LinearReg", "SVR", "Lasso", "GPR", "ElasticNetCV", "ExtraTrees", "GradBoost", "AdaBoost", "KNN", 
% "LassoLarsCV", "LinearSVR", "RidgeCV", "SGDReg", "Ridge", "LassoLars", "ElasticNet", "RVM", "RVR"]
Settings.CBFAtlasType = ["TotalGM","Tatu_ACA_MCA_PCA","DeepWM"]; % Select Atlas used for feature creation. Options are:["TotalGM","DeepWM","ATTbasedFlowTerritories","Tatu_ACA_MCA_PCA","Desikan_Killiany_MNI_SPM12","Hammers",H0cort_CONN"]
Settings.FeatureType = ["T1w","FLAIR","CBF","CoV"]; % Select feature types. Options are: ["T1w", "FLAIR", "CBF", "CoV", "ATT", "Tex", or all combinations in format ["T1W",FLAIR"]]. 
Settings.HemisphereType = ["Both"]; % Use ExploreASL values for both hemispheres ["Both"] or single ["Single"]
Settings.ValidationMethod = ['K-fold']; % Set validation method to preferred method. Options are: ['Permutation'],['K-fold'],['Stratified K-fold']
Settings.PermutationSplitSize = 0.2; % Set split size of validation set for permutations, between 0 and 1
Settings.ValidationMethodRepeats = 5; % Set number of K-folds, or number of permutations.
Settings.FeatureImportance = [1];  % Turn SHAP feature importance estimation method on or off;
Settings.FeatureImportanceForAlgorithm = ["ExtraTrees"]; % If set to specific algorithms, this will make sure feature importance is only performed for this algorithm to speed up processing. Default = [], example = ["ExtraTrees"]

% avoid double quotes ! 
% get CBF Atlas type from datapar (?)
% add script that could add extra dataset via some settings
% bilateral or unilateral rename
% perhaps remove ID column
% generate .json for every .tsv/.csv to understand purpose of every file
% xASL_io_writejson

% !! add new features here if necessary !!
Settings.FeatureSets.T1w = ["GM_vol","WM_vol","CSF_vol","GM_ICVRatio","GMWM_ICVRatio"];
Settings.FeatureSets.FLAIR = ["WMHvol_WMvol","WMH_count"];
Settings.FeatureSets.CBF = ["CBF"];
Settings.FeatureSets.CoV = ["CoV"];
Settings.FeatureSets.ASL = ["CBF", "CoV"];
Settings.FeatureSets.ATT = ["ATT"];
Settings.FeatureSets.Tex = ["Tex"];
% !! add new features here if necessary !!

% Subjects to be removed
% provide string of subject ID's
Settings.RemoveTrainingSubjectsList = ['sub-5908001_1','sub-599401_1','sub-5909601_1','sub- 5910801_1','sub-5912001_1','sub-5912002_1','sub-5913501_1','sub-5915802_1','sub-5917601_1','sub-5922602_1','sub-5926501_1','sub-5926502_1','sub-5940701_1','sub-5941901_1','sub-0055_1','sub-0056_1','sub-0734_1','sub-1038_1'];
Settings.RemoveValidationSubjectsList = [];
Settings.RemoveTestingSubjectsList = ['sub-ALZH0333801946_1','sub-ALZH0420802298_1','sub-ALZH0451401874_1','sub-ALZH0535702107_1','sub-ALZH0556800763_1','sub-ALZH0571700412_1','sub-ALZH0596301452_1','sub-ALZH0596301590_1','sub-ALZH0596301870_1','sub-ALZH0596302038_1','sub-ALZH0664100000_1','sub-ALZH0665200000_1','sub-ALZH0668900000_1','sub-ALZH0682900000_1','sub-ALZH0693200000_1','sub-ALZH0697100000_1','sub-ALZH0719500000_1','sub-ALZH0796101058_1','sub-ALZH0904600252_1','sub-ALZH0915401506_1','sub-ALZH0976801196_1','sub-ALZH1020200393_1','sub-ALZH1020200589_1','sub-ALZH0668900000_1'];
%% Admin
% Data paths %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Settings.Paths.TrainingSetPath = fullfile(Settings.DataFolder,'Training/');
Settings.Paths.ValidationSetPath = fullfile(Settings.DataFolder,'Validation/');
Settings.Paths.TestingSetPath = fullfile(Settings.DataFolder,'Testing/');
Settings.Paths.ResultsPath = fullfile(Settings.DataFolder,'Results/');
Settings.Paths.MLPath = fullfile(Settings.DataFolder,'ML/');

if  exist(Settings.Paths.TrainingSetPath,'dir') ~= 7 % xASL_adm_createdir
    mkdir(Settings.Paths.TrainingSetPath)
end
if  exist(Settings.Paths.ValidationSetPath,'dir') ~= 7
    mkdir(Settings.Paths.ValidationSetPath)
end
if  exist(Settings.Paths.TestingSetPath,'dir') ~= 7
    mkdir(Settings.Paths.TestingSetPath)
end

if  exist(Settings.Paths.ResultsPath,'dir') ~= 7
    mkdir(Settings.Paths.ResultsPath)
end

Settings.Paths.Results.CBA_validation = fullfile(Settings.Paths.ResultsPath,'CBA_estimation_validation.tsv');
Settings.Paths.Results.CBA_test = fullfile(Settings.Paths.ResultsPath ,'CBA_estimation_test.csv');
Settings.Paths.Results.CBA_test_cor = fullfile(Settings.Paths.ResultsPath,'CBA_estimation_test_cor.csv');


%% Data structure creation

Settings = xASL_CBA_ConfigureData(Settings);
disp('Data structure created')

%% Feature selection

Settings = xASL_CBA_SelectFeatureData(Settings, 1);
if  Settings.ValidateInTraining ~= 1
    Settings = xASL_CBA_SelectFeatureData(Settings, 2);
end

if  Settings.TestInTraining ~= 1
    Settings = xASL_CBA_SelectFeatureData(Settings, 3);
end

%% Python ML
xASL_CBA_ML(Settings);

%% Prediction output

if Settings.ValidateInTraining
xASL_CBA_ShowResults(Settings, Settings.Paths.Results.CBA_test)
xASL_CBA_ShowResults(Settings, Settings.Paths.Results.CBA_test_cor)
else
xASL_CBA_ShowResults(Settings, Settings.Paths.Results.CBA_test)
xASL_CBA_ShowResults(Settings, Settings.Paths.Results.CBA_validation)
xASL_CBA_ShowResults(Settings, Settings.Paths.Results.CBA_test_cor)
end

