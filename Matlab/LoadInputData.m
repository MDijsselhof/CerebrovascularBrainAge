function DataSets = LoadInputData(DataSetList, DataSetlocation, CBFAtlasType)
% this function reads a list of Datasets that are to be read for later conversion to ML datasets
NDataSets = numel(DataSetList); % amount of datasets to be read

for nDataSets = 1:NDataSets
    
    % imaging data
    DataSetPath{nDataSets} = fullfile(DataSetlocation,DataSetList{nDataSets},'/'); % create path to imaging data
    if contains(DataSetPath{nDataSets},'FeatureSets') == 1
        continue % skip this iteration as FeatureSets does not contain data
    end
    
    ImagingDataList{nDataSets,:} = xASL_adm_GetFileList(DataSetPath{nDataSets},'^.+qCBF.+tsv$','List',[],false); % list imaging data
    
    % select data using Atlas input
    SelectedDataSetsList = contains(ImagingDataList{nDataSets,:}, CBFAtlasType);
    SelectedImagingDataList = ImagingDataList{nDataSets,:}(SelectedDataSetsList,:); % final list for reading imaging data
    
    NImagingData = numel(SelectedImagingDataList); % amount of imaging data per dataset
    
    for nImagingData = 1:NImagingData
        DataFilePath = fullfile(DataSetPath{nDataSets},char(SelectedImagingDataList{nImagingData}));
        DataCSV = xASL_tsvRead(char(DataFilePath));
        DataCSV(2,:) = []; % remove unit of measurements
        % add CBF or CoV to column header for later identification
        if contains(DataFilePath,'CoV') == 1
           DataCSV(1,13:end) = cellfun(@(c)[c '_CoV'],DataCSV(1,13:end),'uni',false); 
        else
           DataCSV(1,13:end) = cellfun(@(c)[c '_CBF'],DataCSV(1,13:end),'uni',false); 
        end
         DataSet{nImagingData,:} = DataCSV;
    end
    % add age and sex data
    AgeSexDataPath = xASL_adm_GetFileList(DataSetPath{nDataSets},'^Age.+$','FPList',[],false); % all scans
    AgeSexData = xASL_csvRead(AgeSexDataPath{1});
    DataSet{nImagingData+1,:} = AgeSexData;
    
    DataSets{nDataSets,:} = DataSet; % final data
end

end