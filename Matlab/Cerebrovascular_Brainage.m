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
Settings.FeatureType = ["T1w","FLAIR","CBF","CoV"]; % Select feature types. Options are: ["T1w", "FLAIR", "CBF", "CoV", "ATT", "Tex", or all combinations in format ["T1W",FLAIR"]]. 
Settings.HemisphereType = ["Both"]; % Use ExploreASL values for both hemispheres ["Both"] or single ["Single"]
Settings.TestInTraining = 1; % Boolean. If testing also using part of training data, set to 1;
Settings.TestFraction = 0.2; % Set testing fraction to preferred number. Only if TestInTraining is True, otherwise will be ignored
Settings.ValidationFraction = 0; % Set validation fraction to preferred number. 

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

[TrainingDataSetPath, ValidationDataSetPath, TestingDataSetPath] = xASL_CBA_ConfigureData(Settings);
disp('Data structure created')

%% Feature selection

[TrainingFeatureSetsFolder, TrainingFeatureSetsNames,SelectedFeaturesList] = xASL_CBA_SelectFeatureData(TrainingDataSetPath, Settings, 1);
if Settings.ValidationFraction > 0
[ValidationFeatureSetsFolder, ValidationFeatureSetsNames] = xASL_CBA_SelectFeatureData(ValidationDataSetPath, Settings, 2);
end
[TestingFeatureSetsFolder,TestingFeatureSetsNames] = xASL_CBA_SelectFeatureData(TestingDataSetPath, Settings, 3);

%% Python ML
if Settings.ValidationFraction > 0
xASL_CBA_ML(TrainingFeatureSetsFolder, ValidationFeatureSetsFolder, TestingFeatureSetsFolder, Settings, SelectedFeaturesList);
else
xASL_CBA_ML(TrainingFeatureSetsFolder, [], TestingFeatureSetsFolder, Settings, SelectedFeaturesList);
end
%% Prediction output

xASL_CBA_ShowResults(Settings, Settings.Paths.Results.CBA_validation)
xASL_CBA_ShowResults(Settings, Settings.Paths.Results.CBA_test)
xASL_CBA_ShowResults(Settings, Settings.Paths.Results.CBA_test_cor)
