function Output = xASL_CBA_ShowResults(Settings, ResultsPath)
%% Extract and present Brain Age Estimation Results
% This wrapper loads ML results and displays best performing model outputs for selected results
% admin
ResultsString = ResultsPath{1}; % convert string array to string
ResultsName = ResultsString(strlength(Settings.Paths.ResultsPath)+1:end-4); % extract file name for later printing of values
ResultsFilePath = ResultsPath;
Results = xASL_csvRead(ResultsFilePath);
AlgorithmLoc = find(cellfun(@(x)strcmp(x,'Algorithm'),Results(1,:)));  % loc of algorithm column
FeatureSetLoc = find(cellfun(@(x)strcmp(x,'Feature_combo'),Results(1,:))); % loc of feature combo

% print results per column, assuming results start after 'Algorithm' column
ResultsNColumns = size(Results,2);

for iResultsColumn = AlgorithmLoc + 1 : ResultsNColumns
    
    if ~isnumeric(Results{2,iResultsColumn}) % check first value if numeric, otherwise convert
        Results(2:end,iResultsColumn) = cellfun(@str2num, Results(2:end,iResultsColumn),'un',0); % convert to double
    end
    
    ResultName =  Results{1,iResultsColumn}; % get name of result metric
    LowestValueResult = min(cell2mat(Results(2:end,iResultsColumn))); % lowest result metric value
    LowestValueResultLoc = find(cellfun(@(x)isequal(x,LowestValueResult),Results(2:end,iResultsColumn))); % location of result metric value
    LowestValueResultAlgorithm = Results(LowestValueResultLoc,AlgorithmLoc); % location of result metric value
    LowestValueResultFeatureset =Results(LowestValueResultLoc,FeatureSetLoc); % location of result metric value
    
    HighestValueResult = max(cell2mat(Results(2:end,iResultsColumn))); % highest result metric value
    HighestValueResultLoc = find(cellfun(@(x)isequal(x,HighestValueResult),Results(2:end,iResultsColumn))); % location of highest result metric value
    HighestValueResultAlgorithm = Results(HighestValueResultLoc,AlgorithmLoc); % location of highest result metric value
    HighestValueResultFeatureset =Results(HighestValueResultLoc,FeatureSetLoc); % location of highest result metric value
    
    disp(['---------------------------- Results of '  ResultsName  ' ----------------------------'])
    disp(['Lowest value of ' ResultName ': ' num2str(LowestValueResult) ' for ' LowestValueResultAlgorithm{1} ' and ' LowestValueResultFeatureset{1} ' for ' ResultsName ])
    disp(['Highest value of ' ResultName ': ' num2str(HighestValueResult) ' for ' HighestValueResultAlgorithm{1} ' and ' HighestValueResultFeatureset{1} ' for ' ResultsName ])
    
end

end