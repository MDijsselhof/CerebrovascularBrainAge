function MLoutput = xASL_CBA_ML(Settings)
%% Brain Age Prediction
% This wrapper loads ML configured datasets, calls Python for brain age predictions, and stores output

% admin
MLscriptPath = char(fullfile(Settings.PythonEnvironment,'MachineLearning.py')); % Python3 ML script
FeatureSets = xASL_adm_GetFileList(Settings.FeatureSetsFolder,'^.+$','List',[],false); % get feature sets

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
        MLAlgorithmsList = [MLAlgorithmsList ',' MLAlgorithmName];
    end
end
% create feature set list
NFeatureSets = numel(FeatureSets);

for nFeatureSet = 1 : NFeatureSets
    FeatureSet = FeatureSets{nFeatureSet,1}; % get feature set
    FeatureSetName = char(FeatureSet(1,13:end-4)); % get name of feature set
    if nFeatureSet == 1
        FeatureSetsList = FeatureSetName;
    else
        FeatureSetsList = [FeatureSetsList ',' FeatureSetName];
    end
end

Settings.FeatureSetsList = FeatureSetsList;
Settings.MLAlgorithmsList = MLAlgorithmsList;

disp(['Features selected are : ' FeatureSetsList]);
disp(['Algorithms selected are : ' MLAlgorithmsList]);

MLInputJSONpath = fullfile(Settings.DataFolder,'MLInputSettings.json');
MLInputJSON = jsonencode(Settings);

fid = fopen(MLInputJSONpath,'w');
fprintf(fid,'%s',MLInputJSON);
fclose(fid);

% Call Machine Learning script with provided input
PythonCommand = ['python3 ' MLscriptPath ' --MLInputJSON ' char(MLInputJSONpath)];
xASL_system(PythonCommand,1)

end