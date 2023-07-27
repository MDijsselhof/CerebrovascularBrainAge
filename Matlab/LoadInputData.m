function [DataSets, SelectedImagingDataList] = LoadInputData(DataSetList, DataSetlocation, CBFAtlasType)
% this function reads a list of Datasets that are to be read for later conversion to ML datasets
NDataSets = numel(DataSetList); % amount of datasets to be read

for iDataSets = 1:NDataSets
    
    % imaging data
    DataSetPath{iDataSets} = fullfile(DataSetlocation,DataSetList{iDataSets},'/'); % create path to imaging data
    if contains(DataSetPath{iDataSets},'FeatureSets') == 1
        continue % skip this iteration as FeatureSets does not contain data
    end
    
    ImagingDataList{iDataSets,:} = xASL_adm_GetFileList(DataSetPath{iDataSets},'^.+PVC.+tsv$','List',[],false); % list imaging data
    
    % select data using Atlas input
    SelectedDataSetsList = contains(ImagingDataList{iDataSets,:}, CBFAtlasType);
    SelectedImagingDataList = ImagingDataList{iDataSets,:}(SelectedDataSetsList,:); % final list for reading imaging data
    
    NImagingData = numel(SelectedImagingDataList); % amount of imaging data per dataset
    
    for iImagingData = 1:NImagingData
        DataFilePath = fullfile(DataSetPath{iDataSets},char(SelectedImagingDataList{iImagingData}));
        DataCSV = xASL_tsvRead(char(DataFilePath));
        DataCSV(2,:) = []; % remove unit of measurements
        
        % calculate start of non-structural and motion data columns
        NonStructDataStart = 11; % always the same
        if contains(DataCSV{1,NonStructDataStart},'Motion') == 1
            DataStart = NonStructDataStart + 1;
        elseif contains(DataCSV{1,NonStructDataStart},'WMH') == 1
            DataStart = NonStructDataStart + 2;
            if contains(DataCSV{1, NonStructDataStart},'Motion') == 1
                 DataStart = NonStructDataStart + 3;
            end
        else
            DataStart = NonStructDataStart; 
        end
            
        
        % add CBF, CoV, ATT and Tex to column header for later identification
        if contains(DataFilePath,'CoV') == 1
           DataCSV(1,DataStart:end) = cellfun(@(c)[c '_CoV'],DataCSV(1,DataStart:end),'uni',false); 
        elseif contains(DataFilePath,'CBF') == 1
           DataCSV(1,DataStart:end) = cellfun(@(c)[c '_CBF'],DataCSV(1,DataStart:end),'uni',false); 
        elseif contains(DataFilePath,'ATT') == 1
            DataCSV(1,DataStart:end) = cellfun(@(c)[c '_ATT'],DataCSV(1,DataStart:end),'uni',false);
        elseif contains(DataFilePath,'Tex') == 1
            DataCSV(1,DataStart:end) = cellfun(@(c)[c '_Tex'],DataCSV(1,DataStart:end),'uni',false);
        end
         DataSet{iImagingData,:} = DataCSV;
    end
    % add age and sex data
    AgeSexDataPath = xASL_adm_GetFileList(DataSetPath{iDataSets},'^Age.+$','FPList',[],false); % all scans
    AgeSexData = xASL_csvRead(AgeSexDataPath{1});
    DataSet{iImagingData+1,:} = AgeSexData;
    
    DataSets{iDataSets,:} = DataSet; % final data
end

end