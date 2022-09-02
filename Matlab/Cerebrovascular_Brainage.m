%% ExploreASL Cerebrovascular Brain-age
% This wrapper loads ExploreASL imaging data, creates datastructures for input into 
% Python ML, executes the ML and obtains output.

%% admin

addpath('/scratch/mdijsselhof/ExploreASL/') % add ExploreASL path to working directory
ExploreASL_Initialize

addpath('/scratch/mdijsselhof/Cerebrovascular-Brain-age/Matlab') % add Cerebrovascular Brain-age path to working directory
%% Settings
Settings.DataFolder = "/scratch/mdijsselhof/Cerebrovascular-Brain-age/Data/"; %% Add studyfolder
Settings.PythonEnvironment = "/scratch/mdijsselhof/Cerebrovascular-Brain-age/Python/CBA/"; % Python3 environment used for ML 
Settings.MLAlgorithms = ["All"]; % Select Machine Learning algorithms. Options are: ["All", RandomForest", "DecisionTree", "XGBoost", "BayesianRidge", 
% "LinearReg", "SVR", "Lasso", "GPR", "ElasticNetCV", "ExtraTrees", "GradBoost", "AdaBoost", "KNN", 
% "LassoLarsCV", "LinearSVR", "RidgeCV", "SGDReg", "Ridge", "LassoLars", "ElasticNet", "RVM", "RVR"]
Settings.CBFAtlasType = ["GM","CortVascTerritoriesTatu","DeepWM"]; % Select Atlas used for feature creation. Options are:["TotalGM","DeepWM","ATTbasedFlowTerritories","CortVascTerritoriesTatu","Desikan_Killiany_MNI_SPM12","Hammers",H0cort_CONN"]
Settings.FeatureType = ["All"]; % Select feature types. Options are: ["All", "T1w", "FLAIR", "ASL", "T1w+FLAIR", "T1w+ASL", "FLAIR+ASL", "T1w+FLAIR+ASL"]
Settings.HemisphereType = ["Both"]; % Use ExploreASL values for both hemispheres ["Both"] or single ["Single"]
Settings.TestInTraining = 0; % Boolean. If testing also using part of training data, set to 1;
Settings.TestFraction = 0; % Set testing fraction to preferred number. Only if TestInTraining is True, otherwise will be ignored
Settings.ValidationFraction = 0.2; % Set validation fraction to preferred number. 

%% Admin
% Data paths %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Settings.Paths.TrainingSetPath = fullfile(Settings.DataFolder,'Training/');
Settings.Paths.ValidationSetPath = fullfile(Settings.DataFolder,'Validation/');
Settings.Paths.TestingSetPath = fullfile(Settings.DataFolder,'Testing/');
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

%% Subjects to be removed
% provide string of subject ID's

Settings.RemoveTrainingSubjectsList = ["sub-59080_1", "sub-59094_1", "sub-59096_1", "sub-59108_1", "sub-59120_1", "sub-59120_2", "sub-59135_1", "sub-59158_1","sub-59176_1", "sub-59226_2", "sub-59265_1", "sub-59265_2", "sub-59407_1", "sub-59419_1"];
Settings.RemoveTestingSubjectsList = [];
disp('Subjects to be removed added')

%% Data structure creation

[TrainingDataSetPath, ValidationDataPath, TestingDataSetPath] = xASL_CBA_ConfigureData(Settings);
disp('Data structure created')
%% Feature selection

[TrainingFeatureSetsFolder, TrainingFeatureSetsNames] = xASL_CBA_SelectFeatureData(TrainingDataSetPath, Settings, 1);
[ValidationFeatureSetsFolder, ValidationFeatureSetsNames] = xASL_CBA_SelectFeatureData(ValidationDataPath, Settings, 2);
[TestingFeatureSetsFolder,TestingFeatureSetsNames] = xASL_CBA_SelectFeatureData(TestingDataSetPath, Settings, 3);

%% Python ML
TrainingFeatureSetsNamesString = string(TrainingFeatureSetsNames);
TestingFeatureSetsNamesString = string(TestingFeatureSetsNames);
xASL_CBA_ML(TrainingFeatureSetsFolder, ValidationFeatureSetsFolder, TestingFeatureSetsFolder, Settings);

%% Prediction output


