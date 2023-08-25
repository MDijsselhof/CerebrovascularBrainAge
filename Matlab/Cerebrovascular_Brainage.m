%% ExploreASL Cerebrovascular Brain-age
% This wrapper loads ExploreASL imaging data, creates datastructures for input into 
% Python ML, executes the ML and obtains output.

%% admin

addpath('/scratch/mdijsselhof/ExploreASL/') % add ExploreASL path to working directory
ExploreASL_Initialize

addpath('/scratch/mdijsselhof/Cerebrovascular-Brain-age/Matlab') % add Cerebrovascular Brain-age path to working directory
%% Settings
Settings.DataFolder = "/scratch/mdijsselhof/Cerebrovascular-Brain-age/Data/"; %% Add studyfolder
Settings.PythonEnvironment = "/scratch/mdijsselhof/Cerebrovascular-Brain-age/Python/Scripts"; % Python3 environment used for ML 
Settings.MLAlgorithms = ["All"]; % Select Machine Learning algorithms. Options are: ["All", "RandomForest", "DecisionTree", "XGBoost", "BayesianRidge", 
% "LinearReg", "SVR", "Lasso", "GPR", "ElasticNetCV", "ExtraTrees", "GradBoost", "AdaBoost", "KNN", 
% "LassoLarsCV", "LinearSVR", "RidgeCV", "SGDReg", "Ridge", "LassoLars", "ElasticNet", "RVM", "RVR"]
Settings.CBFAtlasType = ["GM","Tatu_ACA_MCA_PCA","DeepWM"]; % Select Atlas used for feature creation. Options are:["TotalGM","DeepWM","ATTbasedFlowTerritories","Tatu_ACA_MCA_PCA","Desikan_Killiany_MNI_SPM12","Hammers",H0cort_CONN"]
Settings.FeatureType = ["T1w","CBF","CoV"]; % Select feature types. Options are: ["T1w", "FLAIR", "CBF", "CoV", "ATT", "Tex", or all combinations in format ["T1W",FLAIR"]]. 
Settings.HemisphereType = ["Both"]; % Use ExploreASL values for both hemispheres ["Both"] or single ["Single"]
Settings.ValidationMethod = ['K-fold']; % Set validation method to preferred method. Options are: ['Permutation'],['K-fold'],['Stratified K-fold']
Settings.PermutationSplitSize = 0.2; % Set split size of validation set for permutations, between 0 and 1
Settings.ValidationMethodRepeats = 5; % Set number of K-folds, or number of permutations.
Settings.FeatureImportance = ['Permutation'];  % Set feature importance estimation method to preferred method. Options are: ['Permutation'],['SHAP'];


% !! add new features here if necessary !!
Settings.FeatureSets.T1w = ["GM_vol","WM_vol","CSF_vol","GM_ICVRatio","GMWM_ICVRatio"];
Settings.FeatureSets.FLAIR = ["WMHvol_WMvol", "WMH_count"];
Settings.FeatureSets.CBF = ["CBF"];
Settings.FeatureSets.CoV = ["CoV"];
Settings.FeatureSets.ASL = ["CBF", "CoV"];
Settings.FeatureSets.ATT = ["ATT"];
Settings.FeatureSets.Tex = ["Tex"];
% !! add new features here if necessary !!

% Subjects to be removed
% provide string of subject ID's
Settings.RemoveTrainingSubjectsList = [];
Settings.RemoveTestingSubjectsList = [];

%% Admin
% Data paths %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Settings.Paths.TrainingSetPath = fullfile(Settings.DataFolder,'Training/');
Settings.Paths.ValidationSetPath = fullfile(Settings.DataFolder,'Validation/');
Settings.Paths.TestingSetPath = fullfile(Settings.DataFolder,'Testing/');
Settings.Paths.ResultsPath = fullfile(Settings.DataFolder,'Results/');
Settings.Paths.MLPath = fullfile(Settings.DataFolder,'ML/');

if ~exist(Settings.Paths.TrainingSetPath,'dir') == 7
    mkdir(Settings.Paths.TrainingSetPath)
end
if ~exist(Settings.Paths.ValidationSetPath,'dir') == 7
    mkdir(Settings.Paths.ValidationSetPath)
end
if ~exist(Settings.Paths.TestingSetPath,'dir') == 7
    mkdir(Settings.Paths.TestingSetPath)
end

if ~exist(Settings.Paths.ResultsPath,'dir') == 7
    mkdir(Settings.Paths.ResultsPath)
end

Settings.Paths.Results.CBA_validation = fullfile(Settings.Paths.ResultsPath,'CBA_estimation_validation.csv');
Settings.Paths.Results.CBA_test = fullfile(Settings.Paths.ResultsPath ,'CBA_estimation_test.csv');
Settings.Paths.Results.CBA_test_cor = fullfile(Settings.Paths.ResultsPath,'CBA_estimation_test_cor.csv');


%% Data structure creation

Settings = xASL_CBA_ConfigureData(Settings);
disp('Data structure created')

%% Feature selection

Settings = xASL_CBA_SelectFeatureData(Settings, 1);
if ~Settings.ValidateInTraining == 1
    Settings = xASL_CBA_SelectFeatureData(ValidationDataSetPath, Settings, 2);
end

if ~Settings.TestInTraining == 1
    Settings = xASL_CBA_SelectFeatureData(TestingDataSetPath, Settings, 3);
end

%% Python ML
xASL_CBA_ML(Settings);

%% Prediction output

if ~Settings.ValidationFraction == 1
xASL_CBA_ShowResults(Settings, Settings.Paths.Results.CBA_test)
else
xASL_CBA_ShowResults(Settings, Settings.Paths.Results.CBA_test)
xASL_CBA_ShowResults(Settings, Settings.Paths.Results.CBA_validation)
xASL_CBA_ShowResults(Settings, Settings.Paths.Results.CBA_test_cor)
end
