function [MLData, RemovedSubjectsList] = xASL_CBA_CreateMLdataset(ImagingData, NDataSets, FeatureType, RemoveSubjectsList)
% ML Data contains subjects, age, sex, imaging data (structural and ASL)
% merged row-wise for all datasets

% create data structure
for nDataset = 1 : NDataSets

    % Demographics
    DataSetData = ImagingData{nDataset,1}; % all imaging data
    DataSetSubjects = DataSetData{1,1}(:,find(contains(DataSetData{end,1}(1,:),"ID"),1)); % obtain subject list
    DataSetSubjectsAge = DataSetData{end,1}(:,find(contains(DataSetData{end,1}(1,:),"Age"),1)); % contains ages
    DataSetSubjectsSex = DataSetData{end,1}(:,find(contains(DataSetData{end,1}(1,:),"Sex"),1)); % contains sex, 1 being male
    DataSetSubjectsSite = DataSetData{end,1}(:,find(contains(DataSetData{end,1}(1,:),"Site"),1)); % contains site
    DataSetIDs{1,1} = 'ID';
    if nDataset == 1
        DataSetIDs(2:size(DataSetData{1,1}(2:end,1),1)+1,1) = num2cell((1:1:(size(DataSetData{1,1}(2:end,1),1))))';
        PreviousnDatasetSubjects = size(DataSetData{1,1}(2:end,1),1);
    else % add number of ID's of previous dataset to current dataset ID's
        DataSetIDs(2:end,:) = []; % first clear cell array
        DataSetIDs(2:size(DataSetData{1,1}(2:end,1),1)+1,1) = num2cell((1:1:(size(DataSetData{1,1}(2:end,1),1))) + PreviousnDatasetSubjects )' ;
        PreviousnDatasetSubjects = DataSetIDs{end,1};
    end

    % calculate start of non-structural and motion data columns, ignore motion
    NonStructDataStart = find(contains(DataSetData{1,1}(1,:),"GMWM_ICVRatio"),1) + 1; % always the same
    if contains(DataSetData{1,1}(1,NonStructDataStart),'Motion') == 1
        StructuralEnd = NonStructDataStart - 1;
        DataStart = NonStructDataStart + 1;
        MotionPresent = 1;
        WMHpresent = 0;
    elseif contains(DataSetData{1,1}(1,NonStructDataStart),'WMH') == 1
        DataStart = NonStructDataStart + 2;
        StructuralEnd = NonStructDataStart + 1;
        WMHpresent = 1 ;
        if contains(DataSetData{1,1}(1,StructuralEnd + 1),'Motion') == 1
            DataStart = NonStructDataStart + 3;
            StructuralEnd = NonStructDataStart + 1;
            MotionPresent = 1;
        end
    else % no WMH and no motion
        DataStart = NonStructDataStart;
        StructuralEnd = NonStructDataStart - 1 ;
        WMHpresent = 0;
    end

    % Structural imaging data
    DataSetStructural = DataSetData{1,1}(:,6:StructuralEnd); % get structural data from first imaging datasubset (as its the same for all)

    % ASL imaging data
    NASLImagingDataSubset = numel(DataSetData) - 1; % amount of ASL datasubsets

    for nASLImagingDataSubset = 1 : NASLImagingDataSubset

        nDataSubSetASL = DataSetData{nASLImagingDataSubset,1}(:,DataStart:end); % get ASL data columns
        nDataSubSetASLHeaders = nDataSubSetASL(1,:);

        if nASLImagingDataSubset == 1
            DataSetASL = nDataSubSetASL; % set as first columns
        else
            DataSetASL = [DataSetASL nDataSubSetASL]; % add to existing columns
        end
    end

    % ASL imaging data - feature construction
    % hemishpere selection
    if FeatureType == 'Both' % use both hemispheres, remove individual left and right hemispheres
        ASLDataLeftHemispereLocation = find(contains(DataSetASL(1,:),'_L_'));
        DataSetASL(:,ASLDataLeftHemispereLocation) = []; % remove left
        ASLDataRightHemispereLocation = find(contains(DataSetASL(1,:),'_R_'));
        DataSetASL(:,ASLDataRightHemispereLocation) = []; % remove right
    else % Use Left and Right results, remove Both hemispheres
        ASLDataBothHemispereLocation = find(contains(DataSetASL(1,:),'_B_'));
        DataSetASL(:,ASLDataBothHemispereLocation) = []; % remove both
    end

    MLnDataSet = [DataSetSubjects DataSetIDs DataSetSubjectsAge DataSetSubjectsSex DataSetSubjectsSite DataSetStructural DataSetASL];

    if WMHpresent == 1 % turn NaNs to 0 and create ratio
        % set NaN WMH to 0
        [~, WMHscolumn] = find(contains(MLnDataSet(1,:),'WMH')); % find n/a in WMH columns
        [WMNaNRowLoc, WMNaNColumnLoc] = find(contains(MLnDataSet(1:end,WMHscolumn),'n/a')); % find n/a in WMH columns
        MLnDataSet(WMNaNRowLoc, WMHscolumn) = cellstr('0');

        % construct new features
        % WMHvol/WMvol
        [~, WMcolumn] = find(contains(MLnDataSet(1,:),'WM_vol')); % find n/a in WMH columns
        [~, WMHvolcolumn] = find(contains(MLnDataSet(1,:),'WMH_vol')); % find n/a in WMH columns
        WMColumns = str2double(MLnDataSet(2:end,WMcolumn));
        WMHColumns = str2double(MLnDataSet(2:end,WMHvolcolumn));
        WMHvolWMvol= (WMHColumns./1000)./(WMColumns); % divide WMH vol by 1000 to get to L
        MLnDataSet(2:end,WMHvolcolumn) = num2cell(WMHvolWMvol);
        MLnDataSet{1,WMHvolcolumn} = 'WMHvol_WMvol';
    end

    % remove selected subjects
    nDataSetRemoveSubjectsList = RemoveSubjectsList;
    if ~isempty(nDataSetRemoveSubjectsList) == 1
        [nDataSetRemoveSubjectsLocRow, nDataSetRemoveSubjectsLocColumn] = find(contains(MLnDataSet(:,1),nDataSetRemoveSubjectsList)); % find loc of subjects to be removed
        MLnDataSet(nDataSetRemoveSubjectsLocRow,:) = [];
    end

    if nDataset == 1
        MLData = MLnDataSet;
    else
        %         if size(MLnDataSet(2:end,:),2) == size(MLData,2)
        %             MLData(end+1:(end+(size(MLnDataSet(:,1),1)-1)),:) = MLnDataSet(2:end,:);
        %         elseif  size(MLData,2) < size(MLnDataSet,2)
        %             warning('WARNING: Features do not match, using smallest amount of features available ')
        %             LocFeaturesAvailable = find(contains(MLnDataSet(1,:),MLData(1,:)));
        %             LocFeaturesMissing = find(~contains(MLnDataSet(1,:),MLData(1,:)));
        %             disp(['Features missing are: ' MLnDataSet{1,LocFeaturesMissing}] )
        %             MLData_tailored = MLData(:,LocFeaturesAvailable); % extract columns
        %             MLData_tailored(end+1:(end+(size(MLnDataSet(:,1),1)-1)),:) = MLnDataSet(2:end,LocFeaturesAvailable);
        %             MLData = MLData_tailored;
        %         elseif size(MLData,2) > size(MLnDataSet(2:end,:),2)
        %             warning('WARNING: Features do not match, using smallest amount of features available ')
        %             LocFeaturesAvailable = find(contains(MLData(1,:),MLnDataSet(1,:)));
        %             MLData(end+1:(end+(size(MLnDataSet(:,1),1)-1)),:) = MLnDataSet(2:end,LocFeaturesAvailable); % extract columns
        %end
        MLData(end+1:(end+(size(MLnDataSet(:,1),1)-1)),:) = MLnDataSet(2:end,:);


    end
end

% remove duplicate columns if present
FeatureHeaders = MLData(1,:);
[~, uniqueIdx] =unique(FeatureHeaders); % Find the indices of the unique strings
duplicates = FeatureHeaders; % Copy the original into a duplicate array
duplicates(uniqueIdx) = []; % remove the unique strings, anything left is a duplicate
DuplicatedHeaders = unique(duplicates); % find the unique duplicates.

if numel(DuplicatedHeaders) ~= 0
    disp(['Duplicated column headers found, removing duplicate : ' strjoin(DuplicatedHeaders)]);
    for iDuplicatedHeaders = 1 : numel(DuplicatedHeaders)
        DuplicatedHeaderLocs = find(contains(FeatureHeaders,DuplicatedHeaders{iDuplicatedHeaders}));
        MLData(:,DuplicatedHeaderLocs(1,2:end)) = []; % remove column(s)
    end
end

% remove NaN subjects
MLDataFindNaN = cellfun(@num2str,MLData,'un',0);
[NaNlocRow, ~] = find(contains(MLDataFindNaN(:,:),'n/a'));
UniqueNanLocRow = unique(NaNlocRow);
RemovedSubjectsList(1:size(MLDataFindNaN(UniqueNanLocRow,1),1),nDataset) = MLDataFindNaN(UniqueNanLocRow,1);
MLData(UniqueNanLocRow,:) = [];

% remove empty rows
MLData = MLData(~cellfun(@isempty,MLData(:,1)),:);
end

