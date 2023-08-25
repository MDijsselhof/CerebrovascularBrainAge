function [Settings] = xASL_CBA_SelectFeatureData(Settings, TrainValTest)
% this function selects feature data from the large data structure,
% and prints new training, validation and testing data containing only selected features
% TrainingBoolean = 1; Set dataset to save in Training or Testing Folder
% admin
SelectedFeatureSet = Settings.FeatureType;

if TrainValTest == 1 % training folder
    DatasetPath = Settings.TrainingDataPath;
    Settings.FeatureSetsFolder = char(fullfile(Settings.Paths.TrainingSetPath,'FeatureSets'));
    FeatureSetsSavePath = char(fullfile(Settings.FeatureSetsFolder,'TrainingSet_'));
elseif TrainValTest == 2 % validation folder
    DatasetPath = Settings.ValidationDataPath;
    Settings.FeatureSetsFolder = char(fullfile(Settings.Paths.ValidationSetPath,'FeatureSets'));
    FeatureSetsSavePath = char(fullfile(Settings.FeatureSetsFolder,'ValidationSet_'));
else % testing folder
    DatasetPath = Settings.TestingDataPath;
    Settings.FeatureSetsFolder = char(fullfile(Settings.Paths.TestingSetPath,'FeatureSets'));
    FeatureSetsSavePath = char(fullfile(Settings.FeatureSetsFolder,'TestingSet_'));
end

if  exist(Settings.FeatureSetsFolder) == 7
    disp('Feature sets folder already present');
else
    mkdir(Settings.FeatureSetsFolder)
end


% build all features
Settings.FinalFeatureSets = FeatureConstruction(Settings.FeatureSets);

for iSelectedFeature = 1 : numel(SelectedFeatureSet)
    FeatureSetNameCombinations = combnk(SelectedFeatureSet,iSelectedFeature); % combinations of feature names
    if isequal(iSelectedFeature,1)
        Settings.SelectedFeaturesFinal(1:size(FeatureSetNameCombinations,1),1) = FeatureSetNameCombinations;
    else % here we concatenate the names where needed
        FeatureSetNameCombinationsNew = [];
        FeatureSetColumnNameCombinationsNew = [];
        for iSetCombinations = 1 : size(FeatureSetNameCombinations,1)
            FeatureSetNameCombinationsNew{iSetCombinations,1} = strjoin(FeatureSetNameCombinations(iSetCombinations,:),'');
        end
        Settings.FinalFeatureSetsSize = size(Settings.SelectedFeaturesFinal,1);
        Settings.SelectedFeaturesFinal(Settings.FinalFeatureSetsSize+1:Settings.FinalFeatureSetsSize+size(FeatureSetNameCombinationsNew,1),1) = FeatureSetNameCombinationsNew;
    end
end

NFeatureSets = numel(Settings.FinalFeatureSets);

% load data
DataSet = xASL_tsvRead(DatasetPath);

% select columns of features
FeatureLoc = find(ismember(Settings.FinalFeatureSets(:,1),Settings.SelectedFeaturesFinal));
for nFeatureSet = 1 : size(FeatureLoc,1) 
    FeatureNames = Settings.FinalFeatureSets(FeatureLoc(nFeatureSet),2);
    % select basic data
    FeatureSet = DataSet(:,1:5); % select subject, ID, Age, Sex, Site columns
    
    % select feature data
    for iCell = 1 : size(Settings.FinalFeatureSets(nFeatureSet,2))
        FeatureNamesLoc = [];
        if iscell(FeatureNames{iCell})
            for iCell2 = 1 : numel(FeatureNames{iCell})
                FeatureNamesLoc = [];
                FeatureNamesCell = FeatureNames{iCell};
                [~, FeatureNamesLoc] = find(contains(DataSet(1,:),FeatureNamesCell{iCell2}));
                FeatureSet(:,end+1:end+size(FeatureNamesLoc,2)) = DataSet(:,FeatureNamesLoc);
            end
        else
            [~, FeatureNamesLoc] = find(contains(DataSet(1,:),FeatureNames{iCell}));
            FeatureSet(:,end+1:end+size(FeatureNamesLoc,2)) = DataSet(:,FeatureNamesLoc);
        end
    end
    
    % save
    FeatureSetPath = char(fullfile([FeatureSetsSavePath Settings.FinalFeatureSets{FeatureLoc(nFeatureSet),1} '.tsv']));
    xASL_tsvWrite(FeatureSet,FeatureSetPath,1,0)
end
disp('Feature sets constructed')
end

function [OutputFeatures] = FeatureConstruction(FeatureSets) % creates a list of combinations of feature set names, and column names in ExploreASL output, 
FeatureSetsNames = fieldnames(FeatureSets);
FeatureSetsColumnNames= struct2cell(FeatureSets);
OutputFeatures = {};

for iCombinationSize = 1 : numel(FeatureSetsNames)
    FeatureSetNameCombinations = combnk(FeatureSetsNames,iCombinationSize); % combinations of feature names
    FeatureSetColumnNameCombinations = combnk(FeatureSetsColumnNames,iCombinationSize);  % combinations of feature column names
    if isequal(iCombinationSize,1)
        OutputFeatures(1:size(FeatureSetNameCombinations,1),1) = FeatureSetNameCombinations;
        OutputFeatures(1:size(FeatureSetNameCombinations,1),2) = FeatureSetColumnNameCombinations;
    else % here we concatenate the names where needed
        FeatureSetNameCombinationsNew = [];
        FeatureSetColumnNameCombinationsNew = [];
        for iSetCombinations = 1 : size(FeatureSetNameCombinations,1)
            FeatureSetNameCombinationsNew{iSetCombinations,1} = strjoin(FeatureSetNameCombinations(iSetCombinations,:),'');
            FeatureSetColumnNameCombinationsNew{iSetCombinations,1} = FeatureSetColumnNameCombinations(iSetCombinations,:);
        end
        Settings.FinalFeatureSetsSize = size(OutputFeatures,1);
        OutputFeatures(Settings.FinalFeatureSetsSize+1:Settings.FinalFeatureSetsSize+size(FeatureSetNameCombinationsNew,1),1) = FeatureSetNameCombinationsNew;
        OutputFeatures(Settings.FinalFeatureSetsSize+1:Settings.FinalFeatureSetsSize+size(FeatureSetNameCombinationsNew,1),2) = FeatureSetColumnNameCombinationsNew;
    end
end
end