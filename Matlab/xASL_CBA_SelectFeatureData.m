function [FeatureSetsFolder, FinalFeatureSets, SelectedFeaturesFinal] = xASL_CBA_SelectFeatureData(DatasetPath, Settings, TrainValTest)
% this function selects feature data from the large data structure,
% and prints new training and testing data containing only selected features
% TrainingBoolean = 1; Set dataset to save in Training or Testing Folder
% admin
SelectedFeatureSet = Settings.FeatureType;

if TrainValTest == 1 % training folder
    FeatureSetsFolder = char(fullfile(Settings.Paths.TrainingSetPath,'FeatureSets'));
    FeatureSetsSavePath = char(fullfile(FeatureSetsFolder,'TrainingSet_'));
elseif TrainValTest == 2 % validation folder
    FeatureSetsFolder = char(fullfile(Settings.Paths.ValidationSetPath,'FeatureSets'));
    FeatureSetsSavePath = char(fullfile(FeatureSetsFolder,'ValidationSet_'));
else % testing folder
    FeatureSetsFolder = char(fullfile(Settings.Paths.TestingSetPath,'FeatureSets'));
    FeatureSetsSavePath = char(fullfile(FeatureSetsFolder,'TestingSet_'));
end

if  exist(FeatureSetsFolder) == 7
    disp('Feature sets folder already present');
else
    mkdir(FeatureSetsFolder)
end

% !! add new features here if necessary !!
FeatureSets.T1w = ["GM_vol","WM_vol","CSF_vol","GM_ICVRatio","GMWM_ICVRatio"];
FeatureSets.FLAIR = ["WMHvol_WMvol", "WMH_count"];
FeatureSets.CBF = ["CBF"];
FeatureSets.CoV = ["CoV"];
FeatureSets.ASL = ["CBF", "CoV"];
FeatureSets.ATT = ["ATT"];
FeatureSets.Tex = ["Tex"];
% !! add new features here if necessary !!

% build all features
FinalFeatureSets = FeatureConstruction(FeatureSets);

for iSelectedFeature = 1 : numel(SelectedFeatureSet)
    FeatureSetNameCombinations = combnk(SelectedFeatureSet,iSelectedFeature); % combinations of feature names
    if isequal(iSelectedFeature,1)
        SelectedFeaturesFinal(1:size(FeatureSetNameCombinations,1),1) = FeatureSetNameCombinations;
    else % here we concatenate the names where needed
        FeatureSetNameCombinationsNew = [];
        FeatureSetColumnNameCombinationsNew = [];
        for iSetCombinations = 1 : size(FeatureSetNameCombinations,1)
            FeatureSetNameCombinationsNew{iSetCombinations,1} = strjoin(FeatureSetNameCombinations(iSetCombinations,:),'');
        end
        FinalFeatureSetsSize = size(SelectedFeaturesFinal,1);
        SelectedFeaturesFinal(FinalFeatureSetsSize+1:FinalFeatureSetsSize+size(FeatureSetNameCombinationsNew,1),1) = FeatureSetNameCombinationsNew;
    end
end

NFeatureSets = numel(FinalFeatureSets);

% load data
DataSet = xASL_tsvRead(DatasetPath);

% select columns of features
FeatureLoc = find(ismember(FinalFeatureSets(:,1),SelectedFeaturesFinal));
for nFeatureSet = 1 : size(FeatureLoc,1) 
    FeatureNames = FinalFeatureSets(FeatureLoc(nFeatureSet),2);
    % select basic data
    FeatureSet = DataSet(:,1:4); % select subject, ID, Age, Sex columns
    
    % select feature data
    for iCell = 1 : size(FinalFeatureSets(nFeatureSet,2))
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
    FeatureSetPath = char(fullfile([FeatureSetsSavePath FinalFeatureSets{FeatureLoc(nFeatureSet),1} '.tsv']));
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
        FinalFeatureSetsSize = size(OutputFeatures,1);
        OutputFeatures(FinalFeatureSetsSize+1:FinalFeatureSetsSize+size(FeatureSetNameCombinationsNew,1),1) = FeatureSetNameCombinationsNew;
        OutputFeatures(FinalFeatureSetsSize+1:FinalFeatureSetsSize+size(FeatureSetNameCombinationsNew,1),2) = FeatureSetColumnNameCombinationsNew;
    end
end
end