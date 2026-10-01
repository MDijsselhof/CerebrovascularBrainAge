%% ExploreASL Cerebrovascular Brain-age
% This wrapper loads ExploreASL imaging data, creates datastructures for input into 
% Python ML, executes the ML and obtains output.

%% admin

addpath('/home/P074668/Work/ExploreASL/Main/ExploreASL/') % add ExploreASL path to working directory
ExploreASL_Initialize

addpath('/home/P074668/Work/CerebrovascularBrainAge/CerebrovascularBrainAge/Matlab/') % add Cerebrovascular Brain-age path to working directory
%% Settings
Settings.DataFolder = "/home/P074668/Work/CerebrovascularBrainAge/CerebrovascularBrainAge/Data/"; %% Add folder containing Cerebrovascular Brain Age data used for training-validation-testing
Settings.PythonEnvironment = "/home/P074668/Work/CerebrovascularBrainAge/CerebrovascularBrainAge/Python/Scripts"; % Python3 scripts used for ML 
Settings.CondaEnvironmentPath = '/home/P074668/.conda/envs/CBA/';  % location of Conda environment containing required packages
Settings.CondaVersionName = 'anaconda3/2024.10-1'; % name of conda version to load
Settings.MLAlgorithms = ["ExtraTrees"]; % Select Machine Learning algorithms. Options are: ["All", "RandomForest", "DecisionTree", "XGBoost", "BayesianRidge", 
% "LinearReg", "SVR", "Lasso", "GPR", "ElasticNetCV", "ExtraTrees", "GradBoost", "AdaBoost", "KNN", 
% "LassoLarsCV", "LinearSVR", "RidgeCV", "SGDReg", "Ridge", "LassoLars", "ElasticNet", "RVM", "RVR"]
Settings.CBFAtlasType = ["TotalGM","Tatu_ACA_MCA_PCA"]; % Select Atlas used for feature creation. Options are:["TotalGM","DeepWM","Tatu_ICA_PCA","Tatu_ACA_MCA_PCA","Tatu_ACA_MCA_PCA_Prox_Med_Dist","Desikan_Killiany_MNI_SPM12","Hammers",H0cort_CONN"]
Settings.FeatureType = ["T1w","CBF"]; % Select feature types. Options are: ["T1w", "FLAIR", "CBF", "CoV", "ATT", "Tex", or all combinations in format ["T1W",FLAIR"]]. 
Settings.HemisphereType = ["Both"]; % Use ExploreASL values for both hemispheres ["Both"] or single ["Single"]
Settings.ValidationMethod = ['K-fold']; % Set validation method to preferred method. Options are: ['Permutation'],['K-fold'],['Stratified K-fold']
Settings.PermutationSplitSize = []; % Set split size of validation set for permutations, between 0 and 1
Settings.ValidationMethodRepeats = 5; % Set number of K-folds, or number of permutations.
Settings.FeatureImportance = 1;  % Turn SHAP feature importance estimation method on or off;
Settings.FeatureImportanceForAlgorithm = ["RVR"]; % If set to specific algorithms, this will make sure feature importance is only performed for this algorithm to speed up processing. Default = [], example = ["ExtraTrees"]

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
Settings.RemoveTrainingSubjectsList = [];
Settings.RemoveValidationSubjectsList = [];
Settings.RemoveTestingSubjectsList = [];
Settings.RemoveTestingSubjectsList = ["sub-PD007_1", "sub-PROB015_1", "sub-PROB006_1", "sub-PD020_1", "sub-PROB002_1", "sub-PD018_1", "sub-PD028_1", "sub-PROB016_1", "sub-PD017_1", "sub-PD015_1", "sub-PD027_1", "sub-PD013_1", "sub-PD026_1", "sub-PD010_1", "sub-PROB014_1", "sub-PD023_1", "sub-PD024_1"];
%Settings.RemoveTrainingSubjectsList = ["sub-5908001_1","sub-5994_1","sub-59096_1","sub-59108_1","sub-59120_1","sub-59120_2","sub-59135_1","sub-59158_2","sub-59176_1","sub-59226_2","sub-59265_1","sub-59265_2","sub-59407_1","sub-59419_1","sub-0055_1","sub-0056_1","sub-0734_1","sub-1038_1"];
%Settings.RemoveTestingSubjectsList = ["sub-501102_1","sub-11563_1","sub-122261_1","sub-133876_1","sub-140746_1","sub-141136_1","sub-142816_1","sub-144760_1","sub-15286_1","sub-159311_1","sub-184186_1","sub-188230_1","sub-194657_1","sub-216498_1","sub-223818_1","sub-225494_1","sub-226969_1","sub-233895_1","sub-234917_1","sub-235783_1","sub-241962_1","sub-242299_1","sub-243281_1","sub-256966_1","sub-258791_1","sub-258917_1","sub-26653_1","sub-273374_1","sub-273619_1","sub-306526_1","sub-327345_1","sub-328939_1","sub-329567_1","sub-332509_1","sub-341657_1","sub-34184_1","sub-346276_1","sub-357795_1","sub-358873_1","sub-361064_1","sub-365862_1","sub-378796_1","sub-40125_1","sub-41957_1","sub-48077_1","sub-500590_1","sub-500629_1","sub-501315_1","sub-501420_1","sub-501474_1","sub-501793_1","sub-502099_1","sub-502213_1","sub-502260_1","sub-502391_1","sub-502636_1","sub-600062_1","sub-600074_1","sub-600113_1","sub-600134_1","sub-600145_1","sub-600148_1","sub-81191_1","sub-82150_1","sub-84888_1","sub-93497_1","sub-97819_1"];
%Settings.RemoveTestingSubjectsList = ["sub-ALZH0333801946_1","sub-ALZH0420802298_1","sub-ALZH0451401874_1","sub-ALZH0535702107_1","sub-ALZH0556800763_1","sub-ALZH0571700412_1","sub-ALZH0596301452_1","sub-ALZH0596301590_1","sub-ALZH0596301870_1","sub-ALZH0596302038_1","sub-ALZH0664100000_1","sub-ALZH0665200000_1","sub-ALZH0668900000_1","sub-ALZH0682900000_1","sub-ALZH0693200000_1","sub-ALZH0697100000_1","sub-ALZH0719500000_1","sub-ALZH0796101058_1","sub-ALZH0904600252_1","sub-ALZH0915401506_1","sub-ALZH0976801196_1","sub-ALZH1020200393_1","sub-ALZH1020200589_1","sub-ALZH0668900000_1"];
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

