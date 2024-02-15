# Cerebrovascular-Brain-age
Repository for Cerebrovascular Brain-age estimation. This package predicts age based on T1w, FLAIR and ASL MRI imaging following these steps:

1. Create training-validation-testing datasets through, by default, ExploreASL output (T1w, FLAIR, ASL) in .csv format.
2. Create manually selected feature sets to perform age estimation.
3. Execute Python scripts to perform the actual machine learning.
4. Provide output (under development).

## Requirements
Cerebrovascular Brain-age estimation requires three folders:

- Data
- Matlab
- Python
	- Packages:
 		- argparse
  		- os
  		- numpy as np
  		- torch 
  		- shap 
  		- copy
  		- json
  		- matplotlib
  		- matplotlib.pyplot as plt
  		- pandas as pd
  		- scipy 
  		- tqdm 
  		- time 

#### Data

This folder consists of the following sub folders: 

- Training : Datasets used for training the model
- Validation : Datasets used for validating the model (can also be estimated from the training dataset, using permutation or K-fold approaches)
- Testing : Datasets used for testing the model/ predicting age (can be constructed from the training dataset)

All datasets require output of ExploreASL format (.csv) and an Age_sex.csv file containing the subject name, timepoint, and age and sex (1 for male, 0 for female).

#### Matlab

This folder contains the Matlab scripts required for executing the Cerebrovascular Brain-age estimation. ExploreASL is also required (optionally in a seperate directory).

#### Python

This folder contains the MachineLearning.py script, which executes the Machine Learning part. This folder may also contain a virtual environment, in which the MachineLearning.py script should be placed.

## How to start

1. Make sure every requirements described above is fullfilled.
2. Open the Cerebrovascular_Brainage.m script.
    1. Change settings if required
	1a. New featuresets can be added in the settings file. Please follow the used notation. Features not supported by ExploreASL might need manual tweaking to be able to work, and should follow notations and composition used by ExploreASL stats output files. 
    2. Create `Data` required folders using the script if required
3. Place training-validation-testing data in the newly created folders
4. Run the remainder of the script

## Example
See CerebrovascularBrainAge/ExampleData folder for an example of the training, testing and Age_sex.csv files. Note, these files are not actual data, and not usable by Cerebrovascular Brain-age 

## Settings
- Settings.DataFolder = "/FOLDER/CerebrovascularBrainAge/Data/"; %% Add folder containing Cerebrovascular Brain Age data used for training-validation-testing
- Settings.PythonEnvironment = "/FOLDER/CerebrovascularBrainAge/Python/Scripts"; % Python3 scripts used for ML 
- Settings.CondaEnvironmentPath = '/FOLDER/.conda/envs/CBA/';  % location of Conda environment containing required packages
- Settings.MLAlgorithms = ["All"]; % Select Machine Learning algorithms. Options are: ["All", "RandomForest", "DecisionTree", "XGBoost", "BayesianRidge", 
% "LinearReg", "SVR", "Lasso", "GPR", "ElasticNetCV", "ExtraTrees", "GradBoost", "AdaBoost", "KNN", 
% "LassoLarsCV", "LinearSVR", "RidgeCV", "SGDReg", "Ridge", "LassoLars", "ElasticNet", "RVM", "RVR"]
- Settings.CBFAtlasType = ["TotalGM","Tatu_ACA_MCA_PCA","DeepWM"]; % Select Atlas used for feature creation. Obtained from ExploreASL processing. Current supported options are:["TotalGM","DeepWM","ATTbasedFlowTerritories","Tatu_ACA_MCA_PCA","Desikan_Killiany_MNI_SPM12","Hammers",H0cort_CONN"]
- Settings.FeatureType = ["T1w","FLAIR","CBF","CoV"]; % Select feature types. Options are: ["T1w", "FLAIR", "CBF", "CoV", "ATT", "Tex", or all combinations in format ["T1W",FLAIR"]]. 
- Settings.HemisphereType = ["Both"]; % Use ExploreASL values for both hemispheres ["Both"] or single ["Single"]
- Settings.ValidationMethod = ['K-fold']; % Set validation method to preferred method. Options are: ['Permutation'],['K-fold'],['Stratified K-fold']
- Settings.PermutationSplitSize = []; % Set split size of validation set for permutations, between 0 and 1. Set to [] if using K-fold. 
- Settings.ValidationMethodRepeats = 5; % Set number of K-folds, or number of permutations.
- Settings.FeatureImportance = [1];  % Turn SHAP feature importance estimation method on or off;
- Settings.FeatureImportanceForAlgorithm = ["ExtraTrees"]; % If set to specific algorithms, this will make sure feature importance is only performed for this algorithm to speed up processing. Default = [], example = ["ExtraTrees"]

% !! add new features here if necessary !!
- Settings.FeatureSets.T1w = ["GM_vol","WM_vol","CSF_vol","GM_ICVRatio","GMWM_ICVRatio"];
- Settings.FeatureSets.FLAIR = ["WMHvol_WMvol","WMH_count"];
- Settings.FeatureSets.CBF = ["CBF"];
- Settings.FeatureSets.CoV = ["CoV"];
- Settings.FeatureSets.ASL = ["CBF", "CoV"];
- Settings.FeatureSets.ATT = ["ATT"];
- Settings.FeatureSets.Tex = ["Tex"];
% !! add new features here if necessary !!

% Subjects to be removed
% provide string of subject ID's
Settings.RemoveTrainingSubjectsList = [];
Settings.RemoveValidationSubjectsList = [];
Settings.RemoveTestingSubjectsList = []

## Output
Cerebrovascular Brain-age will output results in the CerebrovascularBrainAge/Data/Results folder.
Output is:
- CBA_estimation_test.csv % Output file that contains test set Feature_combo, No. features, Algorithm, mae, rmse, R2, explained_var for every model
- CBA_estimation_test_cor.csv % Output file that contains  test set Feature_combo, No. features, Algorithm, mae, rmse, R2, explained_var for every model, corrected for age estimation bias with linear regression.
- CBA_estimation_validation.csv % Output file that contains validation set Feature_combo, No. features, Algorithm, mae, rmse, R2, explained_var for every model.

- FeatureSet_PredictedAges_test.csv % Output file that contains test set predictions for every model, and chronological age, using the specific feature set combinations. 
- FeatureSet_PredictedAges_test_cor.csv % Output file that contains test set predictions for every model, and chronological age, using the specific feature set combinations, corrected for age estimation bias with linear regression
- FeatureSet_PredictedAges_test_validation.csv % Output file that contains validation set predictions for every model, and chronological age, using the specific feature set combinations. 

- SHAP_bar_FeatureSet_Model.svg % Image showing SHAP feature importance for the specified feature set combinations and model, obtained from the training set. 
- SHAP_dot_FeatureSet_Model.svg % Image showing SHAP feature importance model behaviour per subject,for the specified feature set combinations and model, obtained from the training set. 


