function [MLData, RemovedSubjectList] = xASL_CBA_CreateMLdataset(ImagingData, NDataSets, FeatureType, RemoveSubjectsList)
% ML Data contains subjects, age, sex, imaging data (structural and ASL)
% merged row-wise for all datasets

% create data structure
for nDataset = 1 : NDataSets
    
    % Demographics
    DataSetData = ImagingData{nDataset,1}; % all imaging data
    DataSetSubjects = DataSetData{1,1}(:,1); % obtain subject list
    DataSetSubjectsAge = DataSetData{end,1}(:,4); % last cell contains ages
    DataSetSubjectsSex = DataSetData{end,1}(:,3); % last cell contains sex, 1 being male
    DataSetIDs{1,1} = 'ID';
    if nDataset == 1
        DataSetIDs(2:size(DataSetData{1,1}(2:end,1),1)+1,1) = num2cell((1:1:(size(DataSetData{1,1}(2:end,1),1))))';
        PreviousnDatasetSubjects = size(DataSetData{1,1}(2:end,1),1);
    else % add number of ID's of previous dataset to current dataset ID's
        DataSetIDs(2:end,:) = []; % first clear cell array
        DataSetIDs(2:size(DataSetData{1,1}(2:end,1),1)+1,1) = num2cell((1:1:(size(DataSetData{1,1}(2:end,1),1))) + PreviousnDatasetSubjects )' ;
        PreviousnDatasetSubjects = DataSetIDs{end,1};
    end
    
    % Structural imaging data
    DataSetStructural = DataSetData{1,1}(:,6:12); % get structural data from first imaging datasubset (as its the same for all)
    
    % ASL imaging data
    NASLImagingDataSubset = numel(DataSetData) - 1; % amount of ASL datasubsets
    
    for nASLImagingDataSubset = 1 : NASLImagingDataSubset
        
        nDataSubSetASL = DataSetData{nASLImagingDataSubset,1}(:,13:end); % get ASL data columns
        nDataSubSetASLHeaders = nDataSubSetASL(1,:);
        
        if nASLImagingDataSubset == 1
            DataSetASL = nDataSubSetASL; % set as first columns
        else
            DataSetASL = [DataSetASL nDataSubSetASL]; % add to existing columns
        end
    end
    
    % ASL imaging data - feature construction
    % hemishpere selection
    if FeatureType == 'Both' % use both hemispheres, remove individual ones
        ASLDataLeftHemispereLocation = find(contains(DataSetASL(1,:),'_L_'));
        DataSetASL(:,ASLDataLeftHemispereLocation) = []; % remove left
        ASLDataRightHemispereLocation = find(contains(DataSetASL(1,:),'_R_'));
        DataSetASL(:,ASLDataRightHemispereLocation) = []; % remove right
    else % Use Left and Right results, remove Both hemispheres
        ASLDataBothHemispereLocation = find(contains(DataSetASL(1,:),'_B_'));
        DataSetASL(:,ASLDataBothHemispereLocation) = []; % remove left
    end
    
    MLnDataSet = [DataSetSubjects DataSetIDs DataSetSubjectsAge DataSetSubjectsSex DataSetStructural DataSetASL];
    
    % set NaN WMH to 0
    [~, WMHscolumn] = find(contains(MLnDataSet(1,:),'WMH')); % find n/a in WMH columns
    [WMNaNRowLoc, WMNaNColumnLoc] = find(contains(MLnDataSet(1:end,WMHscolumn),'n/a')); % find n/a in WMH columns
    MLnDataSet(WMNaNRowLoc, WMHscolumn) = cellstr('0');
    
    % construct new features
    % WMHvol/WM/vol
    [~, WMcolumn] = find(contains(MLnDataSet(1,:),'WM_vol')); % find n/a in WMH columns
    [~, WMHvolcolumn] = find(contains(MLnDataSet(1,:),'WMH_vol')); % find n/a in WMH columns
    WMColumns = str2double(MLnDataSet(2:end,WMcolumn));
    WMHColumns = str2double(MLnDataSet(2:end,WMHvolcolumn));
    WMHvolWMvol= (WMHColumns./1000)./(WMColumns); % divide WMH vol by 1000 to get to L
    MLnDataSet(2:end,WMHvolcolumn) = num2cell(WMHvolWMvol);
    MLnDataSet{1,WMHvolcolumn} = 'WMHvol_WMvol';
    
    % remove NaN subjects
    [NaNlocRow, NanLocColumn] = find(contains(DataSetASL(:,:),'n/a'));
    UniqueNanLocRow = unique(NaNlocRow);
    RemovedSubjectList(:,nDataset) = MLnDataSet(UniqueNanLocRow,1);
    % get subject name for list
    MLnDataSet(UniqueNanLocRow,:) = [];
    
    % remove selected subjects
    
    nDataSetRemoveSubjectsList = RemoveSubjectsList;
    if ~isempty(nDataSetRemoveSubjectsList) == 1
        [nDataSetRemoveSubjectsLocRow, nDataSetRemoveSubjectsLocColumn] = find(contains(MLnDataSet(:,1),nDataSetRemoveSubjectsList)); % find loc of subjects to be removed
        MLnDataSet(nDataSetRemoveSubjectsLocRow,:) = [];
    end
    
    if nDataset == 1
        MLData = MLnDataSet;
    else
        MLData(end+1:(end+(size(MLnDataSet(:,1),1)-1)),:) = MLnDataSet(2:end,:);
    end
end