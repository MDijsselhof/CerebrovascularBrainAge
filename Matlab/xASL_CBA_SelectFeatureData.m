function [FeatureSetsFolder, FeatureSetsNames, SelectedFeatures] = xASL_CBA_SelectFeatureData(DatasetPath, Settings, TrainValTest)
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

% build features
FeatureSets.T1w = ["GM_vol","WM_vol","CSF_vol","GM_ICVRatio","GMWM_ICVRatio"];
FeatureSets.FLAIR = ["WMHvol_WMvol", "WMH_count"];
FeatureSets.ASL = ["CoV", "CBF"];
FeatureSets.T1wFLAIR = ["GM_vol","WM_vol","CSF_vol","GM_ICVRatio","GMWM_ICVRatio","WMHvol_WMvol", "WMH_count"];
FeatureSets.T1wASL = ["GM_vol","WM_vol","CSF_vol","GM_ICVRatio","GMWM_ICVRatio", "CoV", "CBF"];
FeatureSets.FLAIRASL = ["WMHvol_WMvol", "WMH_count", "CoV", "CBF"];
FeatureSets.T1wFLAIRASL = ["GM_vol","WM_vol","CSF_vol","GM_ICVRatio","GMWM_ICVRatio", "WMHvol_WMvol", "WMH_count", "CoV", "CBF"];

FeatureSetsNames = fieldnames(FeatureSets);
NFeatureSets = numel(FeatureSetsNames);

% load data
DataSet = xASL_tsvRead(DatasetPath);

% select columns of features
if isequal(SelectedFeatureSet,"All")  % select all feature sets
    for nFeatureSet = 1 : NFeatureSets
        FeatureNames = FeatureSets.(FeatureSetsNames{nFeatureSet,1});
        % select basic data
        FeatureSet = DataSet(:,1:4); % select subject, ID, Age, Sex columns
        
        % select feature data
        [~, FeatureNamesLoc] = find(contains(DataSet(1,:),FeatureNames));
        FeatureSet(:,end+1:end+size(FeatureNamesLoc,2)) = DataSet(:,FeatureNamesLoc);
        
        % save
        FeatureSetPath = char(fullfile([FeatureSetsSavePath FeatureSetsNames{nFeatureSet,1} '.tsv']));
        xASL_tsvWrite(FeatureSet,FeatureSetPath,1,0)
    end
else % select featuresets provided in settings
    [FeatureLoc, ~] = find(contains(FeatureSetsNames,SelectedFeatureSet));
    for nFeatureSet = 1 : size(FeatureLoc,1)
        FeatureNames = FeatureSets.(FeatureSetsNames{FeatureLoc(nFeatureSet),1});
        % select basic data
        FeatureSet = DataSet(:,1:4); % select subject, ID, Age, Sex columns
        
        % select feature data
        [~, FeatureNamesLoc] = find(contains(DataSet(1,:),FeatureNames));
        FeatureSet(:,end+1:end+size(FeatureNamesLoc,2)) = DataSet(:,FeatureNamesLoc);
        
        % save
        FeatureSetPath = char(fullfile([FeatureSetsSavePath FeatureSetsNames{FeatureLoc(nFeatureSet),1} '.tsv']));
        xASL_tsvWrite(FeatureSet,FeatureSetPath,1,0)   
    end
end
disp('Feature sets constructed')
end