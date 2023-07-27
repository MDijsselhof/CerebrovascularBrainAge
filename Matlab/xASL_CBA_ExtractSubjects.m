function ExtractedData  = xASL_CBA_ExtractSubjects(Settings, InputData, InputDataPaths)
% Extracts Subjects based on Age_Sex.csv
for NInputDataSets = 1 : size(InputData,1)
    for NInputImagingDataSets = 1 : size(InputData{NInputDataSets,1},1)
        EqualSizeTraining(NInputImagingDataSets,1) = size(InputData{1,1}{NInputImagingDataSets,1},1);
        UniqueSizesDatasets(NInputDataSets,1) = size(unique(EqualSizeTraining),1);
    end
end

if sum(UniqueSizesDatasets) > size(InputData,1) % if sum is larger, then more than one dataset somewhere is unique -> run subject extraction
    
    for iDataSet = 1 : size(InputData,1)
        SizeImageDatasets =   size(InputData{iDataSet,1},1); % size of datasets
        SubjectDataSet = InputData{iDataSet,1}{SizeImageDatasets,1}; % last is always Age_Sex.csv
        NSubjects = size(SubjectDataSet,1);
        for nSubject = 1 : NSubjects
            for NImageDataSet = 1 : SizeImageDatasets-1
                if nSubject == 1
                    ExtractedData{iDataSet,1}{NImageDataSet,:}(nSubject,:) = InputData{iDataSet,1}{NImageDataSet,:}(nSubject,:); % copy first row, headers of data
                else
                    SubjectName = SubjectDataSet(nSubject,1);
                    SubjectList = InputData{iDataSet,1}{NImageDataSet,1}(:,1); % list of subject names
                    SubjectLoc = find(contains(SubjectList,SubjectName)); % find subject location in image data set
                    if size(SubjectLoc,1) == 1
                        ExtractedData{iDataSet,1}{NImageDataSet,:}(nSubject,:) = InputData{iDataSet,1}{NImageDataSet,:}(SubjectLoc,:); % copy subject to new cell array
                    else
                        ExtractedData{iDataSet,1}{NImageDataSet,:}(nSubject,:) = InputData{iDataSet,1}{NImageDataSet,:}(SubjectLoc(1),:); % copy first of duplicate subject to new cell array
                    end
                end
            end
        end
        ExtractedData{iDataSet,1}{SizeImageDatasets,:} = SubjectDataSet; % copy Age_Sex.csv to new ExtractedData set
    end
    disp('Training subjects in Age_Sex.csv succesfully extracted, processing will continue with those')
else
    disp('All Training datasets are of equal size, skipping subject extraction')
    ExtractedData = InputData;
end
end