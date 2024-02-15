function ExtractedData  = xASL_CBA_ExtractSubjects(Settings, InputData, InputDataPaths)
% Extracts Subjects based on Age_Sex.csv
for NInputDataSets = 1 : size(InputData,1)
    for NInputImagingDataSets = 1 : size(InputData{NInputDataSets,1},1)
        EqualSizeTraining(NInputImagingDataSets,1) = size(InputData{NInputDataSets,1}{NInputImagingDataSets,1},1);
        UniqueSizesDatasets(NInputDataSets,1) = size(unique(EqualSizeTraining),1);
    end
end

if sum(UniqueSizesDatasets) > size(InputData,1) % if sum is larger, then more than one dataset somewhere is unique -> run subject extraction

    for iDataSet = 1 : size(InputData,1)
        SizeImageDatasets =   size(InputData{iDataSet,1},1); % size of datasets
        SubjectDataSet = InputData{iDataSet,1}{SizeImageDatasets,1}; % last is always Age_Sex.csv
        NSubjects = size(SubjectDataSet,1);
        CheckFinalData = 0; % to check for subjects that are not in imaging dataset
        for nSubject = 1 : NSubjects
            for NImageDataSet = 1 : SizeImageDatasets-1
                if nSubject == 1
                    ExtractedData{iDataSet,1}{NImageDataSet,:}(nSubject,:) = InputData{iDataSet,1}{NImageDataSet,:}(nSubject,:); % copy first row, headers of data
                else
                    SubjectName = SubjectDataSet(nSubject,1);
                    if contains(SubjectName,'"')
                        SubjectName = erase(SubjectName,'"');
                    end
                    %if ~contains(SubjectName,'sub') % add _ to make sure this whole name is used in case of names using similar characters/numbers
                    %    SubjectName = strcat(SubjectName,'_');
                    %end

                    SubjectList = InputData{iDataSet,1}{NImageDataSet,1}(:,1); % list of subject names
                    SubjectLoc = find(contains(SubjectList,SubjectName)); % find subject location in image data set
                    if size(SubjectLoc,1) > 1 % select first occurence
                        SubjectLoc = SubjectLoc(1,1);
                        warning('WARNING, Imaging data contains duplicate subjects, but Age_sex only one! Selecting first')
                    end
                    if isempty(SubjectLoc)
                        warning(['WARNING, image datasets do not contain subject ' num2str(SubjectLoc) 'provided, skipping.'])
                        CheckFinalData = 1;
                    else
                        if size(SubjectLoc,1) == 1
                            ExtractedData{iDataSet,1}{NImageDataSet,:}(nSubject,:) = InputData{iDataSet,1}{NImageDataSet,:}(SubjectLoc,:); % copy subject to new cell array
                        else
                            ExtractedData{iDataSet,1}{NImageDataSet,:}(nSubject,:) = InputData{iDataSet,1}{NImageDataSet,:}(SubjectLoc(1),:); % copy first of duplicate subject to new cell array
                        end
                    end
                end
            end
        end
        ExtractedData{iDataSet,1}{SizeImageDatasets,:} = SubjectDataSet; % copy Age_Sex.csv to new ExtractedData set
        if isequal(CheckFinalData,1) % remove empty rows of subjects that were not present in imaging datasets
            CheckEmptyRows = ExtractedData{iDataSet,1}{1,1}(:,1);
            [EmptyRow,~] = find(cellfun(@isempty,CheckEmptyRows));
            for kDataSet = 1 : SizeImageDatasets
                ExtractedData{iDataSet}{kDataSet}(unique(EmptyRow),:) = [];
            end
        end
    end
    
    
    
    disp('Subjects in Age_Sex.csv succesfully extracted, processing will continue with those')
else
    disp('All Training datasets are of equal size, skipping subject extraction')
    ExtractedData = InputData;
end
end