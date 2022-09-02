function MLoutput = xASL_CBA_ML(TrainingSetPath, ValidationSetPath, TestingSetPath, Settings)
%% Brain Age Prediction
% This wrapper loads ML configured datasets, calls Python for brain age predictions, and stores output

% admin
MLscriptPath = char(fullfile(Settings.PythonEnvironment,'MachineLearning.py')); % Python3 ML script
FeatureSetsTrainingList = xASL_adm_GetFileList(TrainingSetPath,'^.+$','List',[],false); % get feature sets
FeatureSetsValidationList = xASL_adm_GetFileList(ValidationSetPath,'^.+$','List',[],false); % get feature sets
FeatureSetsTestingList = xASL_adm_GetFileList(TestingSetPath,'^.+$','List',[],false); % get feature sets

% create results directory
Settings.Paths.ResultsPath = fullfile(Settings.DataFolder,'Results/');
if exist(Settings.Paths.ResultsPath,'dir') == 7
    disp('Results folder already exists')
else
    mkdir(Settings.Paths.ResultsPath )
end

% create selected algorithm list
if isequal(Settings.MLAlgorithms,"All") == 1 % use all algorithms
    MLAlgorithms = ["RandomForest", "DecisionTree", "XGBoost", "BayesianRidge", "LinearReg", "SVR", "Lasso",...
    "GPR", "ElasticNetCV", "ExtraTrees", "GradBoost", "AdaBoost", "KNN", "LassoLarsCV",...
    "LinearSVR", "RidgeCV", "SGDReg", "Ridge", "LassoLars", "ElasticNet", "RVM", "RVR"];
else
    MLAlgorithms = Settings.MLAlgorithms; % selection of ML algorithms    
end
NMLAlgorithms = numel(MLAlgorithms); % number of algorithms used for Cerebrovascular Brain-age prediction

for nMLAlgorithm = 1 : NMLAlgorithms
    MLAlgorithmName = char(MLAlgorithms(1,nMLAlgorithm));
    if nMLAlgorithm == 1
        MLAlgorithmsList = MLAlgorithmName;
    else
        MLAlgorithmsList = [MLAlgorithmsList ' ' MLAlgorithmName];
    end
end
% create feature set list
NFeatureSets = numel(FeatureSetsTrainingList);

for nFeatureSet = 1 : NFeatureSets
    FeatureSet = FeatureSetsTrainingList{nFeatureSet,1}; % get feature set
    FeatureSetName = char(FeatureSet(1,13:end-4)); % get name of feature set
    if nFeatureSet == 1
        FeatureSetsList = FeatureSetName;
    else
        FeatureSetsList = [FeatureSetsList ' ' FeatureSetName];
    end
end

% Call Python environment

% Call Machine Learning script with provided input

PythonCommand = ['python3 ' MLscriptPath ' --TrainingDataDir ' char(Settings.Paths.TrainingSetPath) ' --ValidationDataDir ' char(Settings.Paths.ValidationSetPath) ' --TestingDataDir ' char(Settings.Paths.TestingSetPath) ' --ResultsDataDir ' char(Settings.Paths.ResultsPath) ' --FeatureSetsList ' FeatureSetsList ' --AlgorithmsList '  MLAlgorithmsList ];
xASL_system(PythonCommand,1)

end