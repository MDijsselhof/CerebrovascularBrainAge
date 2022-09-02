function [TrainingDataPath, ValidationDataPath, TestingDataPath]  = xASL_CBA_ConfigureData(Settings)
%% Brain Age Prediction configure data
% Load and configure data for ML processing

TrainingSetList = xASL_adm_GetFileList(Settings.Paths.TrainingSetPath,'^.+$','List',[],true); % all scans
TestingSetList = xASL_adm_GetFileList(Settings.Paths.TestingSetPath,'^.+$','List',[],true); % all scans

NTrainingSets = numel(TrainingSetList);
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
TrainingData = LoadInputData(TrainingSetList, Settings.Paths.TrainingSetPath, Settings.CBFAtlasType); % load Testing datasets

% get testing data paths and load data if available
if isempty(TestingSetList)
    disp('Testing data folder empty, assuming testing will be done in training set')
    TestInTraining = 1;
else
    TestingData = LoadInputData(TestingSetList, Settings.Paths.TestingSetPath, Settings.CBFAtlasType); % load Testing datasets
end
    
%% Merge data
% merge all datasets for training and testing, constructs features
% also removes selected subjects

NTrainingSets = numel(TrainingData); % amount of training datasets
[MLTrainingData, RemovedSubjectListTraining]= xASL_CBA_CreateMLdataset(TrainingData, NTrainingSets, Settings.HemisphereType, Settings.RemoveTrainingSubjectsList);
% save removed subjects
RemovedSubjectListTraining(1:size(Settings.RemoveTrainingSubjectsList,2),end+1) = cellstr(Settings.RemoveTrainingSubjectsList)';
xASL_tsvWrite(RemovedSubjectListTraining, char(fullfile(Settings.Paths.TrainingSetPath,'TrainingDataRemovedSubjects.tsv')),1,0);

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

%% Save data
% save data for use in ML
% Training data is either divided into training and validation, or divided into training and testing based the TestInTraining boolean

    TrainingDataPath = char(fullfile(Settings.Paths.TrainingSetPath,'TrainingDataComplete.tsv'));
    ValidationDataPath = char(fullfile(Settings.Paths.ValidationSetPath,'ValidationDataComplete.tsv'));
    TestingDataPath = char(fullfile(Settings.Paths.TestingSetPath,'TestingDataComplete.tsv'));
        
if ~Settings.TestInTraining == 1

    
    MLTrainingDataNoHeader = MLTrainingData(2:end,:);
    MLTrainingDataShuffled = MLTrainingData(1,:);
    MLTrainingDataShuffled(2:numel(MLTrainingData(2:end,1))+1,:) = MLTrainingDataNoHeader(randperm(numel(MLTrainingDataNoHeader(:,1))),:);
    
    NTraining = round(numel(MLTrainingDataShuffled(2:end,1))*(1-Settings.ValidationFraction),0);
    NValidation = numel(MLTrainingDataShuffled(2:end,1)) - NTraining;
    MLTrainingDataShuffledTraining = MLTrainingDataShuffled(1:NTraining+1,:);
    MLTrainingDataShuffledValidation = MLTrainingDataShuffled(1,:);
    MLTrainingDataShuffledValidation(2:NValidation+2,:) = MLTrainingDataShuffled(NTraining+1:end,:);
 
    xASL_tsvWrite(MLTrainingDataShuffledTraining,TrainingDataPath,1,0);
    xASL_tsvWrite(MLTrainingDataShuffledValidation,ValidationDataPath,1,0);
    disp(['Training data shuffeld into ' num2str(Settings.ValidationFraction) ' validation partition and ' num2str((1-Settings.ValidationFraction)) ' training partition, and saved'])
    
    xASL_tsvWrite(MLTestingData,TestingDataPath,1,0);
    disp('Testing data saved')
else % training data is randomly sorted to training or testing based on partition coefficient
    MLTrainingDataNoHeader = MLTrainingData(2:end,:);
    MLTrainingDataShuffled = MLTrainingData(1,:);
    MLTrainingDataShuffled(2:numel(MLTrainingData(2:end,1))+1,:) = MLTrainingDataNoHeader(randperm(numel(MLTrainingDataNoHeader(:,1))),:);
    
    NTraining = round(numel(MLTrainingDataShuffled(2:end,1))*(1-Settings.TestFraction),0);
    NTesting = numel(MLTrainingDataShuffled(2:end,1)) - NTraining;
    MLTrainingDataShuffledTraining = MLTrainingDataShuffled(1:NTraining+1,:);
    MLTrainingDataShuffledTesting = MLTrainingDataShuffled(1,:);
    MLTrainingDataShuffledTesting(2:NTesting+2,:) = MLTrainingDataShuffled(NTraining+1:end,:);
 
    xASL_tsvWrite(MLTrainingDataShuffledTraining,TrainingDataPath,1,0);
    xASL_tsvWrite(MLTrainingDataShuffledTesting,TestingDataPath,1,0);
    disp(['Training data shuffeld into ' num2str(Settings.TestFraction) ' testing partition and ' num2str((1-Settings.TestFraction)) ' training partition, and saved'])
    
end
disp('Dataset constructed')
end
