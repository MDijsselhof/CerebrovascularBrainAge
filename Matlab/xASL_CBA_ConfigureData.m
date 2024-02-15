function [Settings]  = xASL_CBA_ConfigureData(Settings)
%% Brain Age Prediction configure data
% Load and configure data for ML processing

TrainingSetList = xASL_adm_GetFileList(Settings.Paths.TrainingSetPath,'^.+$','List',[],true); % all scans
ValidationSetList = xASL_adm_GetFileList(Settings.Paths.ValidationSetPath,'^.+$','List',[],true); % all scans
TestingSetList = xASL_adm_GetFileList(Settings.Paths.TestingSetPath,'^.+$','List',[],true); % all scans

NTrainingSets = numel(TrainingSetList);
NValidationSets = numel(ValidationSetList);
NTestingSets = numel(TestingSetList);


% Algorithms %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if contains(Settings.MLAlgorithms,"All") == 1
    MLAlgorithmsList = ["RandomForest", "DecisionTree", "XGBoost", "BayesianRidge", "LinearReg", "SVR", "Lasso", "GPR", "Naive_median", "Naive_mean", "ElasticNetCV", "ExtraTrees", "GradBoost", "AdaBoost", "KNN", "LassoLarsCV", "LinearSVR", "RidgeCV", "SGDReg", "Ridge", "LassoLars", "ElasticNet", "RVM", "RVR"];
    NML_algorithmsList = numel(MLAlgorithmsList);
else
    MLAlgorithmsList = Settings.MLAlgorithms;
    NML_algorithmsList = numel(MLAlgorithmsList);
end

%% Load data

% get training data paths and load data
[TrainingData, TrainingDataPaths] = LoadInputData(TrainingSetList, Settings.Paths.TrainingSetPath, Settings.CBFAtlasType); % load Training datasets

% get validation data paths and load data if available
if isempty(ValidationSetList)
    disp('Validation data folder empty, assuming validation will be done in training set')
    Settings.ValidateInTraining = 1;
else
    [ValidationData, ValidationDataPaths] = LoadInputData(ValidationSetList, Settings.Paths.ValidationSetPath, Settings.CBFAtlasType); % load Testing datasets
end

% get testing data paths and load data if available
if isempty(TestingSetList)
    disp('Testing data folder empty, assuming testing will be done in training set')
    Settings.TestInTraining = 1;
else
    [TestingData, TestingDataPaths] = LoadInputData(TestingSetList, Settings.Paths.TestingSetPath, Settings.CBFAtlasType); % load Testing datasets
    Settings.TestInTraining = 0;
end

%% Extract correct subject data
% If size of the datasets differs within training/validation/testing, extract subjects from Age_Sex.csv for further use
TrainingData = xASL_CBA_ExtractSubjects(Settings, TrainingData, TrainingDataPaths);
if ~Settings.ValidateInTraining
    ValidationData = xASL_CBA_ExtractSubjects(Settings, ValidationData, ValidationDataPaths);
end
if ~Settings.TestInTraining
    TestingData = xASL_CBA_ExtractSubjects(Settings, TestingData, TestingDataPaths);
end

%% Merge data
% merge all datasets for training and testing, constructs features
% also removes selected subjects

NTrainingSets = numel(TrainingData); % amount of training datasets
[MLTrainingData, RemovedSubjectListTraining]= xASL_CBA_CreateMLdataset(TrainingData, NTrainingSets, Settings.HemisphereType, Settings.RemoveTrainingSubjectsList);
% save removed subjects
if ~isempty(Settings.RemoveTrainingSubjectsList) == 1
    RemovedSubjectListTraining(1:size(Settings.RemoveTrainingSubjectsList,2),end+1) = cellstr(Settings.RemoveTrainingSubjectsList)';
    xASL_tsvWrite(RemovedSubjectListTraining, char(fullfile(Settings.Paths.TrainingSetPath,'TrainingDataRemovedSubjects.tsv')),1,0);
end

% get validation data paths and load data if available
if ~Settings.ValidateInTraining == 1
    NValidationSets = numel(ValidationData); % amount of training datasets
    [MLValidationData, RemovedSubjectListValidation] = xASL_CBA_CreateMLdataset(ValidationData, NValidationSets, Settings.HemisphereType, Settings.RemoveValidationSubjectsList);
    % save removed subjects
    if ~isempty(Settings.RemoveValidationSubjectsList) == 1
        RemovedSubjectListTesting(1:size(Settings.RemoveValidationSubjectsList,2),end+1) = cellstr(Settings.RemoveValidationSubjectsList)';
        xASL_tsvWrite(RemovedSubjectListTesting, char(fullfile(Settings.Paths.ValidationSetPath,'ValidationDataRemovedSubjects.tsv')),1,0);
    end
end

% get testing data paths and load data if available
if ~Settings.TestInTraining == 1
    NTestingSets = numel(TestingData); % amount of training datasets
    [MLTestingData, RemovedSubjectListTesting] = xASL_CBA_CreateMLdataset(TestingData, NTestingSets, Settings.HemisphereType, Settings.RemoveTestingSubjectsList);
    % save removed subjects
    if ~isempty(Settings.RemoveTestingSubjectsList) == 1
        RemovedSubjectListTesting(1:size(Settings.RemoveTestingSubjectsList,2),end+1) = cellstr(Settings.RemoveTestingSubjectsList)';
        xASL_tsvWrite(RemovedSubjectListTesting, char(fullfile(Settings.Paths.TestingSetPath,'TestingDataRemovedSubjects.tsv')),1,0);
    end
end

%% clean data   
% several issues with data loading and tsv/csv readers may results in added
% strings in the headers. This part will remove the added strings and
% return the data to its original format

% BasicHeaderString = ["participant_id","ID","Age","Sex","Site"];
% 
% MLFeatureHeaders= fields(Settings.FeatureSets(1));
% NMLFeatureHeaders= numel(fields(Settings.FeatureSets(1)));
% 
% for iFeatureSet = 1 : NMLFeatureHeaderString
%     getfield(Settings.FeatureSets,MLFeatureHeaders{1}
%     NSubFeatures = numel(fields(Settings.FeatureSets(1)));
% end

%% Save data
% save data for use in ML
% Training data is either divided into training and validation, or divided into training and testing based the TestInTraining boolean

TrainingDataPath = char(fullfile(Settings.Paths.TrainingSetPath,'TrainingDataComplete.csv'));
ValidationDataPath = char(fullfile(Settings.Paths.ValidationSetPath,'ValidationDataComplete.csv'));
TestingDataPath = char(fullfile(Settings.Paths.TestingSetPath,'TestingDataComplete.csv'));

if Settings.TestInTraining == 1 && Settings.ValidateInTraining == 1 % only training data available
        
    MLTrainingDataNoHeader = MLTrainingData(2:end,:);
    MLTrainingDataShuffled = MLTrainingData(1,:);
    MLTrainingDataShuffled(2:numel(MLTrainingData(2:end,1))+1,:) = MLTrainingDataNoHeader(randperm(numel(MLTrainingDataNoHeader(:,1))),:);
    
    xASL_csvWrite(MLTrainingDataShuffled,TrainingDataPath,1);
    disp('Training data saved, testing will be done via validation steps in the training data set')

elseif Settings.TestInTraining == 1 && ~Settings.ValidateInTraining == 1 % training and validation data available
    MLTrainingDataNoHeader = MLTrainingData(2:end,:);
    MLTrainingDataShuffled = MLTrainingData(1,:);
    MLTrainingDataShuffled(2:numel(MLTrainingData(2:end,1))+1,:) = MLTrainingDataNoHeader(randperm(numel(MLTrainingDataNoHeader(:,1))),:);
    
    MLValidationDataNoHeader = MLValidationData(2:end,:);
    MLValidationDataShuffled = MLValidationData(1,:);
    MLValidationDataShuffled(2:numel(MLValidationData(2:end,1))+1,:) = MLValidationDataNoHeader(randperm(numel(MLValidationDataNoHeader(:,1))),:);
    
    xASL_csvWrite(MLTrainingDataShuffled,TrainingDataPath,1);
    xASL_csvWrite(MLValidationDataShuffled,ValidationDataPath,1);
    disp('Training data and validation data saved, testing will be done via validation steps in the training data set')
    
elseif  ~Settings.TestInTraining == 1 && Settings.ValidateInTraining == 1 % training and testing data available
    
    MLTrainingDataNoHeader = MLTrainingData(2:end,:);
    MLTrainingDataShuffled = MLTrainingData(1,:);
    MLTrainingDataShuffled(2:numel(MLTrainingData(2:end,1))+1,:) = MLTrainingDataNoHeader(randperm(numel(MLTrainingDataNoHeader(:,1))),:);
    
    MLTestingDataNoHeader = MLTestingData(2:end,:);
    MLTestingDataShuffled = MLTestingData(1,:);
    MLTestingDataShuffled(2:numel(MLTestingData(2:end,1))+1,:) = MLTestingDataNoHeader(randperm(numel(MLTestingDataNoHeader(:,1))),:);
    
    xASL_csvWrite(MLTrainingDataShuffled,TrainingDataPath,1);
    xASL_csvWrite(MLTestingDataShuffled,TestingDataPath,1);
    disp('Training data and testing data saved, validation will be performed in the training data set')
    
else % training, validation and testing data available
    MLTrainingDataNoHeader = MLTrainingData(2:end,:);
    MLTrainingDataShuffled = MLTrainingData(1,:);
    MLTrainingDataShuffled(2:numel(MLTrainingData(2:end,1))+1,:) = MLTrainingDataNoHeader(randperm(numel(MLTrainingDataNoHeader(:,1))),:);

    MLValidationDataNoHeader = MLValidationData(2:end,:);
    MLValidationDataShuffled = MLValidationData(1,:);
    MLValidationDataShuffled(2:numel(MLValidationData(2:end,1))+1,:) = MLValidationDataNoHeader(randperm(numel(MLValidationDataNoHeader(:,1))),:);
    
    MLTestingDataNoHeader = MLTestingData(2:end,:);
    MLTestingDataShuffled = MLTestingData(1,:);
    MLTestingDataShuffled(2:numel(MLTestingData(2:end,1))+1,:) = MLTestingDataNoHeader(randperm(numel(MLTestingDataNoHeader(:,1))),:);
    
    xASL_csvWrite(MLTrainingDataShuffled,TrainingDataPath,1);
    xASL_csvWrite(MLValidationDataShuffled,ValidationDataPath,1);
    xASL_csvWrite(MLTestingDataShuffled,TestingDataPath,1);
    disp('Training data, validation and testing data saved')
end

Settings.TrainingDataPath = TrainingDataPath;
Settings.ValidationDataPath = ValidationDataPath;
Settings.TestingDataPath = TestingDataPath;

disp('Dataset constructed')
end
