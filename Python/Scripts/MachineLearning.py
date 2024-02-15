# This script performs the brain age predictions using features and algorithms provided by ExploreASL
# Mathijs Dijsselhof 25-08-2023
#%% Load modules
# essentials
import argparse
import os
import numpy as np
import torch 
import shap 
import copy
import json
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from tqdm import tqdm
from time import sleep

# modelling setup
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

# sklearn models
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression, Lasso, ElasticNetCV, LassoLarsCV, RidgeCV, \
    SGDRegressor, Ridge, LassoLars, ElasticNet
from sklearn.svm import SVR, LinearSVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.inspection import permutation_importance

# external models
from xgboost import XGBRegressor
from sklearn_rvm import EMRVR # pip install sklearn-rvm
from skrvm import RVR # pip install https://github.com/JamesRitchie/scikit-rvm/archive/master.zip


#%% parse input json
parser = argparse.ArgumentParser(description = "Cerebrovascular Brain-age prediction script")
parser.add_argument("--MLInputJSON", help="Path to ML settings json",type=str)
MLInputJSONPath = parser.parse_args().MLInputJSON 
MLInputJSON = open(MLInputJSONPath)
Settings = json.load(MLInputJSON) 

# testing
#MLInputJSON = open('/home/radv/mdijsselhof/my-scratch/CerebrovascularBrainAge/Data/MLInputSettings.json')
#Settings = json.load(MLInputJSON)

#%% DataPaths
TrainingFeatureSetDataDir = Settings['Paths']['TrainingSetPath'] + 'FeatureSets/' # test script: 
ValidationFeatureSetDataDir = Settings['Paths']['ValidationSetPath'] + 'FeatureSets/' # test script: 
TestingFeatureSetDataDir = Settings['Paths']['TestingSetPath'] + 'FeatureSets/' # test script: 
ResultsDataDir = Settings['Paths']['ResultsPath'] + '/'  # test script: 
    
# feature set list
FeatureSetsList = Settings['FeatureSetsList']
FeatureSetsList = FeatureSetsList.split(',') 

# algorithm list
AlgorithmsList = Settings['MLAlgorithmsList']
AlgorithmsList = AlgorithmsList.split(',') 
SelectedAlgorithmsList = {}
Algorithms = {'RandomForest': RandomForestRegressor,
                  'DecisionTree': DecisionTreeRegressor,
                  'XGBoost': XGBRegressor,
                  'BayesianRidge': BayesianRidge,
                  'LinearReg': LinearRegression,
                  'SVR': SVR,
                  'Lasso': Lasso,
                  'GPR': GaussianProcessRegressor,
                  'ElasticNetCV': ElasticNetCV,
                  'ExtraTrees': ExtraTreesRegressor,
                  'GradBoost': GradientBoostingRegressor,
                  'AdaBoost': AdaBoostRegressor,
                  'KNN': KNeighborsRegressor,
                  'LassoLarsCV': LassoLarsCV,
                  'LinearSVR': LinearSVR,
                  'RidgeCV': RidgeCV,
                  'SGDReg': SGDRegressor,
                  'Ridge': Ridge,
                  'LassoLars': LassoLars,
                  'ElasticNet': ElasticNet,
                  'RVM': EMRVR,
                  'RVR': RVR,
                  }
Algorithm_Arguments = {'RVM':{'kernel': 'rbf'},
                       'RVR':{'kernel': 'linear'},
                       'GPR': {'normalize_y': True}
                       } # required for some algorithms

for SelectedAlgorithmName in AlgorithmsList:
    SelectedAlgorithmsList[SelectedAlgorithmName] = Algorithms.get(SelectedAlgorithmName) # get algorithms and build new dictonary

# feature importance
FeatureImportanceEstimationMethod = Settings['FeatureImportance']
ValidationMethod = Settings['ValidationMethod']
# results
Results = {'Feature_combo': [],
             'No. features': [],
             'Algorithm': [],
             'mae' : [],
             'rmse' : [],
             'R2' : [],
             'explained_var' : []}




#%% Seconday functions
# determine age bias 
def LinearAgeBias(y_val_pred, y_val):
        Age_delta = y_val_pred - y_val
        #Age_delta_r = Age_delta.values.reshape(-1, 1)
        y_val_r = y_val.values.reshape(-1,1)
        LinearReg = LinearRegression().fit(y_val_r,Age_delta)
        RegCoef = LinearReg.coef_
        RegIntercept = LinearReg.intercept_
        
        return RegCoef, RegIntercept

# global evaluation metrics
def evaluation_metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    R2 = r2_score(y_test, y_pred)
    exp_var = explained_variance_score(y_test, y_pred)

    return mae, rmse, R2, exp_var

# feature importance
def Feature_Importance(X_train_SC, AlgorithmName, Algorithm_instantiated, Settings):
    NoOutput = 0
    if Settings['FeatureImportanceForAlgorithm'] == AlgorithmName or Settings['FeatureImportanceForAlgorithm'] == []:
        if AlgorithmName in ['RandomForest','ExtraTrees','DecisionTree','GradBoost','XGBoost','AdaBoost']:
            try:
                explainer = shap.TreeExplainer(Algorithm_instantiated,X_train_SC)
                shap_values = explainer.shap_values(X_train_SC)
            except:
                explainer = shap.Explainer(Algorithm_instantiated.predict,X_train_SC)
                shap_values = explainer(X_train_SC) 
        elif AlgorithmName in [ 'BayesianRidge','Lasso','Ridge','LassoLars','ElasticNetCV','LinearReg','SGDReg','LassoLarsCV','ElasticNet','RidgeCV']:
            try:
                explainer = shap.LinearExplainer(Algorithm_instantiated,X_train_SC)
                shap_values = explainer.shap_values(X_train_SC)
            except: 
                explainer = shap.Explainer(Algorithm_instantiated.predict,X_train_SC)
                shap_values = explainer(X_train_SC) 
        elif AlgorithmName in [ 'GPR','SVR','RVM','RVR','LinearSVR']:
            #try:
            #    explainer = shap.LinearExplainer(Algorithm_instantiated,X_train_SC)
            #    shap_values = explainer.shap_values(X_train_SC)
            #except: 
            #    explainer = shap.Explainer(Algorithm_instantiated.predict,X_train_SC)
            #    shap_values = explainer(X_train_SC) 
            print('currently not implemented for GPR, SVR, RVM, RVR, LinearSVR due to long calculating times and bad general performance')
            NoOutput = 1
            shap_values = 0
        elif AlgorithmName in [ 'KNN']:
            try:
                X_train_SC_sample = shap.kmeans(X_train_SC, 300)
                explainer = shap.KernelExplainer(Algorithm_instantiated.predict,X_train_SC_sample)
                shap_values = explainer.shap_values(X_train_SC_sample)
            except: #
                explainer = shap.Explainer(Algorithm_instantiated.predict,X_train_SC)
                shap_values = explainer(X_train_SC) 
    else:
        print(f"Algorithm {AlgorithmName} not selected for feature importance estimation, skipping")
        NoOutput = 1
        shap_values = 0
    return  shap_values, NoOutput

# testing 
# TrainingSet, X_train_SC, FeatureSetName, Algorithm, AlgorithmName, ValidationMethod, Results, Settings['ValidationMethodRepeats'], Settings['PermutationSplitSize'])          

# create new validation function
def TrainingPerformanceEstimator(TrainingSet, X_train_SC, FeatureSetName, Algorithm, AlgorithmName, ValidationMethod, Results_validation, NPermutation, TestSize):
    NPermutation = NPermutation
    TestSize = TestSize
    KFoldSplits = NPermutation
    Results_validation = copy.deepcopy(Results_validation)
    TrainingSet = TrainingSet
    y_train =  TrainingSet['Age']

### add checking for N per bin, if < KFoldSplits try again with fewer bins
    if 'Site' in TrainingSet.columns and len(set(TrainingSet['Site'])) > 1:
        TrainingSet['Age_bins'] = pd.cut(TrainingSet['Age'],bins=10,labels=False)
        StratifyUniqueCombinations = TrainingSet[['Sex', 'Site','Age_bins']].drop_duplicates()
        StratifyUniqueCombinations['Stratify_Combinations'] = StratifyUniqueCombinations['Sex'].astype(str) + '_' + StratifyUniqueCombinations['Site'].astype(str) + '_' + StratifyUniqueCombinations['Age_bins'].astype(str)
        TrainingSet = TrainingSet.merge(StratifyUniqueCombinations, on=['Sex', 'Site','Age_bins'], how='left')
        y_stratify =TrainingSet['Stratify_Combinations'] 
        print('Site, Sex and Age variables are present, combining into unique combinations to include both in stratification using 10 bins for age')
    else:
        TrainingSet['Age_bins'] = pd.cut(TrainingSet['Age'],bins=5,labels=False)
        StratifyUniqueCombinations = TrainingSet[['Sex','Age_bins']].drop_duplicates()
        StratifyUniqueCombinations['Stratify_Combinations'] = StratifyUniqueCombinations['Sex'].astype(str) + '_' + StratifyUniqueCombinations['Age_bins'].astype(str)
        TrainingSet = TrainingSet.merge(StratifyUniqueCombinations, on=['Sex','Age_bins'], how='left')
        y_stratify =TrainingSet['Stratify_Combinations'] 
        print('Sex and Age variables are present, combining into unique combinations to include both in stratification')
        
        
    mae_concat = []
    rmse_concat = []
    R2_concat = []
    explained_var_concat = []
    RegCoef_concat = []   
    RegIntercept_concat = []

    if ValidationMethod == 'Permutation':
        print('Performing training set validation through permutation ')
        rs = ShuffleSplit(n_splits=NPermutation, test_size=(TestSize), random_state=0)
        rs.get_n_splits(X_train_SC)
        for i, (train_index, test_index) in tqdm(enumerate(rs.split(X_train_SC))):
            # subset selected train and validation data
            X_train_split_SC = X_train_SC[train_index]
            X_validate_split_SC = X_train_SC[test_index]
            y_train_split = y_train[train_index]
            y_validate_split = y_train[test_index]
            
            
            # add arguments for some algorithms
            if AlgorithmName not in Algorithm_Arguments.keys():
                kwargs = {}
            else:
                kwargs = Algorithm_Arguments[AlgorithmName]
        
            Algorithm_instantiated = Algorithm(**kwargs)
            Algorithm_instantiated.fit(X_train_split_SC, y_train_split)
                        
            # val set
            y_pred_validate = Algorithm_instantiated.predict(X_validate_split_SC)
            mae_val, rmse_val, R2_val, exp_var_val = evaluation_metrics(y_validate_split, y_pred_validate)
        
            mae_concat.append(mae_val)
            rmse_concat.append(rmse_val)
            R2_concat.append(R2_val)
            explained_var_concat.append(exp_var_val)
            
            RegCoef, RegIntercept = LinearAgeBias(y_pred_validate, y_validate_split)  # determine age bias
            RegCoef_concat.append(RegCoef)
            RegIntercept_concat.append(RegIntercept)
            
    elif ValidationMethod == 'Permutation' and 'Site' in TrainingSet.columns: # stratified shuffle-split
        print('Performing training set validation through permutation , including stratification for Site, Sex and Age')
        rs = StratifiedShuffleSplit(n_splits=NPermutation, test_size=(TestSize), random_state=0)
        rs.get_n_splits(X_train_SC,y_stratify)
        for i, (train_index, test_index) in tqdm(enumerate(rs.split(X_train_SC, y_stratify))):
            # subset selected train and validation data
            X_train_split_SC = X_train_SC[train_index]
            X_validate_split_SC = X_train_SC[test_index]
            y_train_split = y_train[train_index]
            y_validate_split = y_train[test_index]
            
            
            # add arguments for some algorithms
            if AlgorithmName not in Algorithm_Arguments.keys():
                kwargs = {}
            else:
                kwargs = Algorithm_Arguments[AlgorithmName]
        
            Algorithm_instantiated = Algorithm(**kwargs)
            Algorithm_instantiated.fit(X_train_split_SC, y_train_split)
                        
            # val set
            y_pred_validate = Algorithm_instantiated.predict(X_validate_split_SC)
            mae_val, rmse_val, R2_val, exp_var_val = evaluation_metrics(y_validate_split, y_pred_validate)
        
            mae_concat.append(mae_val)
            rmse_concat.append(rmse_val)
            R2_concat.append(R2_val)
            explained_var_concat.append(exp_var_val)
            
            RegCoef, RegIntercept = LinearAgeBias(y_pred_validate, y_validate_split)  # determine age bias
            RegCoef_concat.append(RegCoef)
            RegIntercept_concat.append(RegIntercept)       
            
            
    elif 'Site' in TrainingSet.columns: # stratified k-fold  
        print('Performing training set validation through stratified K-fold, including stratification for Site, Sex and Age')
        skf = StratifiedKFold(n_splits=KFoldSplits)
        skf.get_n_splits(X_train_SC, y_stratify)
        for i, (train_index, test_index) in enumerate(skf.split(X_train_SC, y_stratify)):
            # subset selected train and validation data
            X_train_split_SC = X_train_SC[train_index]
            X_validate_split_SC = X_train_SC[test_index]
            y_train_split = y_train[train_index]
            y_validate_split = y_train[test_index]
            
            
            # add arguments for some algorithms
            if AlgorithmName not in Algorithm_Arguments.keys():
                kwargs = {}
            else:
                kwargs = Algorithm_Arguments[AlgorithmName]
        
            Algorithm_instantiated = Algorithm(**kwargs)
            Algorithm_instantiated.fit(X_train_split_SC, y_train_split)
                        
            # val set
            y_pred_validate = Algorithm_instantiated.predict(X_validate_split_SC)
            mae_val, rmse_val, R2_val, exp_var_val = evaluation_metrics(y_validate_split, y_pred_validate)
        
            mae_concat.append(mae_val)
            rmse_concat.append(rmse_val)
            R2_concat.append(R2_val)
            explained_var_concat.append(exp_var_val)
            
            RegCoef, RegIntercept = LinearAgeBias(y_pred_validate, y_validate_split)  # determine age bias
            RegCoef_concat.append(RegCoef)
            RegIntercept_concat.append(RegIntercept)
    else:
        print('Performing training set validation through K-fold, including stratification for Sex and Age')
        kf = KFold(n_splits=KFoldSplits)
        kf.get_n_splits(X_train_SC)
        for i, (train_index, test_index) in tqdm(enumerate(kf.split(X_train_SC))):
            # subset selected train and validation data
            X_train_split_SC = X_train_SC[train_index]
            X_validate_split_SC = X_train_SC[test_index]
            y_train_split = y_train[train_index]
            y_validate_split = y_train[test_index]
            
            
            # add arguments for some algorithms
            if AlgorithmName not in Algorithm_Arguments.keys():
                kwargs = {}
            else:
                kwargs = Algorithm_Arguments[AlgorithmName]
        
            Algorithm_instantiated = Algorithm(**kwargs)
            Algorithm_instantiated.fit(X_train_split_SC, y_train_split)
                        
            # test set
            y_pred_validate = Algorithm_instantiated.predict(X_validate_split_SC)
            mae_test, rmse_test, R2_test, exp_var_test = evaluation_metrics(y_validate_split, y_pred_validate)
        
            # val set
            y_pred_validate = Algorithm_instantiated.predict(X_validate_split_SC)
            mae_val, rmse_val, R2_val, exp_var_val = evaluation_metrics(y_validate_split, y_pred_validate)
        
            mae_concat.append(mae_val)
            rmse_concat.append(rmse_val)
            R2_concat.append(R2_val)
            explained_var_concat.append(exp_var_val)
            
            RegCoef, RegIntercept = LinearAgeBias(y_pred_validate, y_validate_split)  # determine age bias
            RegCoef_concat.append(RegCoef)
            RegIntercept_concat.append(RegIntercept)
            
    Results_validation['Feature_combo'].append(FeatureSetName)
    Results_validation['No. features'].append(np.shape(X_train_SC)[1])
    Results_validation['Algorithm'].append(AlgorithmName)
    Results_validation['mae'] = np.average(mae_concat)
    Results_validation['rmse'] = np.average(rmse_concat)
    Results_validation['R2'] = np.average(R2_concat)
    Results_validation['explained_var'] = np.average(explained_var_concat)
    
    RegCoef = np.average(RegCoef_concat)
    RegIntercept = np.average(RegIntercept_concat)
    
    return Results_validation, RegCoef, RegIntercept

# %% Cerebrovascular brain-age prediction
def CBA_prediction (TrainingFeatureSetDataDir, ValidationFeatureSetDataDir, TestingFeatureSetDataDir, Results, FeatureSetsList, SelectedAlgorithmsList, FeatureImportanceEstimationMethod, ValidationMethod, Settings):
    # create result dicts
    Results_val = copy.deepcopy(Results)
    Results_test = copy.deepcopy(Results)
    Results_test_cor = copy.deepcopy(Results)
    
    # select feature set
    for iFeatureSetName, FeatureSetName in enumerate(FeatureSetsList):
        print("\n------------------------------------------------------------------------------")
        print(f"\n{FeatureSetName}, feature set {iFeatureSetName} out of {len(FeatureSetsList)}")
        print("\n------------------------------------------------------------------------------")
        # create dataframe for saving predicted and chronological brain ages
        PredictedAgeDF_val = pd.DataFrame(columns=(AlgorithmsList))
        PredictedAgeDF_test = pd.DataFrame(columns=(AlgorithmsList))
        PredictedAgeDF_test_cor = pd.DataFrame(columns=(AlgorithmsList))
        
        # load training, validation and testing data for feature set
        TrainingSetPath = TrainingFeatureSetDataDir + 'TrainingSet_' + FeatureSetName + '.tsv' 
        ValidationSetPath = ValidationFeatureSetDataDir + 'ValidationSet_' + FeatureSetName + '.tsv' 
        TestingSetPath = TestingFeatureSetDataDir + 'TestingSet_' + FeatureSetName + '.tsv'
        
        # read training, validation and testing data
        TrainingSet = pd.read_csv(TrainingSetPath,engine='python', encoding = 'utf8',sep='\t',)
        
        if os.path.isdir(ValidationFeatureSetDataDir) == 0:
            ValidationSetExists = 0 # continue with validation data creation using permutation or (stratified) K-fold splitting
        else: 
            ValidationSetExists = 1
            ValidationSet = pd.read_csv(ValidationSetPath,engine='python',encoding = 'utf8', sep='\t')
            
        if os.path.isdir(TestingFeatureSetDataDir) == 0:
            TestingSetExists = 0 # continue with validation data creation using permutation or (stratified) K-fold splitting
        else: 
            TestingSetExists = 1
            TestingSet = pd.read_csv(TestingSetPath,engine='python',encoding = 'utf8', sep='\t')        
    
        # split data for ML
        X_train, y_train = TrainingSet.drop(['participant_id', 'ID', 'Age', 'Sex','Site'], axis=1), TrainingSet['Age']
        if ValidationSetExists == 1:
            X_val, y_val = ValidationSet.drop(['participant_id', 'ID', 'Age', 'Sex','Site'], axis=1), ValidationSet['Age']     
        if TestingSetExists == 1:    
            X_test, y_test = TestingSet.drop(['participant_id', 'ID', 'Age', 'Sex','Site'], axis=1), TestingSet['Age']
        
        # perform standard scaling
        SC = StandardScaler()
        X_train_SC = SC.fit_transform(X_train)
        if ValidationSetExists == 1:
            X_val_SC = SC.transform(X_val)    
        if TestingSetExists == 1:
            X_test_SC = SC.transform(X_test)
        
        index = 0
       
        
        # select algorithm and perform ML
        for iAlgorithm, (AlgorithmName, Algorithm) in enumerate(SelectedAlgorithmsList.items()):
            print("\n-----------------------------------------------------------------------------------")
            print(f"\n{AlgorithmName}, algorithm {iAlgorithm} out of {len(SelectedAlgorithmsList.items())}")
            print("\n-----------------------------------------------------------------------------------")

            # add arguments for some algorithms
            if AlgorithmName not in Algorithm_Arguments.keys():
                kwargs = {}
            else:
                kwargs = Algorithm_Arguments[AlgorithmName]
    
            Algorithm_instantiated = Algorithm(**kwargs)
            Algorithm_instantiated.fit(X_train_SC, y_train)
               

            ValidationSetCorrection = []
            # validation set evaluation
            if  ValidationSetExists == 1: # use predefined validation set
                y_val_pred = Algorithm_instantiated.predict(X_val_SC)
                mae_val, rmse_val, R2_val, exp_var_val = evaluation_metrics(y_val, y_val_pred)
            
                Results_val['Feature_combo'].append(FeatureSetName)
                Results_val['No. features'].append(np.shape(X_val_SC)[1])
                Results_val['Algorithm'].append(AlgorithmName)
                Results_val['mae'].append(mae_val)
                Results_val['rmse'].append(rmse_val)
                Results_val['R2'].append(R2_val)
                Results_val['explained_var'].append(exp_var_val)
            
                PredictedAgeDF_val[AlgorithmName] = y_val_pred
                
                RegCoef, RegIntercept = LinearAgeBias(y_val_pred, y_val)  # determine age bias 
                
                ValidationSetCorrection = 1 # continue with validation steps

            elif ValidationSetExists == 0: # create validation dataset using permutation or (stratified) K-fold splitting
                Results_val_append, RegCoef, RegIntercept = TrainingPerformanceEstimator(TrainingSet, X_train_SC, FeatureSetName, Algorithm, AlgorithmName, ValidationMethod, Results, Settings['ValidationMethodRepeats'], Settings['PermutationSplitSize'])          
                Results_val['Feature_combo'].append(Results_val_append['Feature_combo']) 
                Results_val['No. features'].append(Results_val_append['No. features'])
                Results_val['Algorithm'].append(Results_val_append['Algorithm'])
                Results_val['mae'].append(Results_val_append['mae'])
                Results_val['rmse'].append(Results_val_append['rmse'])
                Results_val['R2'].append(Results_val_append['R2'])
                Results_val['explained_var'].append(Results_val_append['explained_var'])
                
                ValidationSetCorrection = 1 # continue with validation steps
            
            # test set
            
            if TestingSetExists == 1:
                y_pred_test = Algorithm_instantiated.predict(X_test_SC)
                mae_test, rmse_test, R2_test, exp_var_test = evaluation_metrics(y_test, y_pred_test)
                
                Results_test['Feature_combo'].append(FeatureSetName)
                Results_test['No. features'].append(len(X_test.columns))
                Results_test['Algorithm'].append(AlgorithmName)
                Results_test['mae'].append(mae_test)
                Results_test['rmse'].append(rmse_test)
                Results_test['R2'].append(R2_test)
                Results_test['explained_var'].append(exp_var_test)
                
                PredictedAgeDF_test[AlgorithmName] = y_pred_test
                
                if Settings['FeatureImportance'] == 1:
                    
                    print('Performing SHAP feature importance estimation')
                    SHAPbarplot = ResultsDataDir + 'SHAP_bar_' + FeatureSetName + '_'+AlgorithmName+'.svg' 
                    SHAPdotplot = ResultsDataDir + 'SHAP_dot_' + FeatureSetName + '_'+AlgorithmName+'.svg' 

                    shap_values, NoOutput = Feature_Importance(X_train_SC, AlgorithmName, Algorithm_instantiated,Settings)
                    
                    if NoOutput == 0:
                        fig = plt.figure()
                        shap.summary_plot(shap_values,X_train,plot_type='bar',show=False)
                        fig.savefig(SHAPbarplot)
                        plt.close(fig)
                        
                        fig = plt.figure()
                        shap.summary_plot(shap_values,X_train,show=False)
                        fig.savefig(SHAPdotplot)
                        plt.close(fig)
                
                if ValidationSetCorrection == 1:
                   # test  corrected set
                   y_pred_test_cor = y_pred_test - (RegCoef * y_test + RegIntercept)
                
                   mae_test_cor, rmse_test_cor, R2_test_cor, exp_var_test_cor = evaluation_metrics(y_test, y_pred_test_cor)
                
                   Results_test_cor['Feature_combo'].append(FeatureSetName)
                   Results_test_cor['No. features'].append(len(FeatureSetName))
                   Results_test_cor['Algorithm'].append(AlgorithmName)
                   Results_test_cor['mae'].append(mae_test_cor)
                   Results_test_cor['rmse'].append(rmse_test_cor)
                   Results_test_cor['R2'].append(R2_test_cor)
                   Results_test_cor['explained_var'].append(exp_var_test_cor)
            
                   PredictedAgeDF_test_cor[AlgorithmName] = y_pred_test_cor

                index += 1 # add index for next interation to append feature importances         
        # save predicted ages 
        if TestingSetExists == 1:
            PredictedAgeDF_test['Chronological_Age'] = y_test
            PredictedAgeDF_test['participant_id'] =  TestingSet['participant_id']
            if ValidationSetExists == 1:
                PredictedAgeDF_val['Chronological_Age'] = y_val
                PredictedAgeDF_test_cor['Chronological_Age'] = y_test
            elif ValidationSetCorrection == 1:
                PredictedAgeDF_test_cor['Chronological_Age'] = y_test
            
            TestFeatureSetPredictedAgePath = ResultsDataDir + FeatureSetName + '_PredictedAges_test.csv' # path for saving predicted ages
            PredictedAgeDF_test.to_csv(TestFeatureSetPredictedAgePath, index=False)
            ResultsDF_test = pd.DataFrame.from_dict(Results_test,orient='index')
            ResultsDF_test_final = ResultsDF_test.transpose() 
            
            
            if ValidationSetExists == 1 or ValidationSetCorrection == 1:
                ValFeatureSetPredictedAgePath = ResultsDataDir + FeatureSetName + '_PredictedAges_val.csv' # path for saving predicted ages
                TestCorFeatureSetPredictedAgePath = ResultsDataDir + FeatureSetName + '_PredictedAges_test_cor.csv' # path for saving predicted ages
                PredictedAgeDF_val.to_csv(ValFeatureSetPredictedAgePath, index=False)
                PredictedAgeDF_test_cor.to_csv(TestCorFeatureSetPredictedAgePath, index=False)
            
                ResultsDF_val = pd.DataFrame.from_dict(Results_val,orient='index')
                ResultsDF_test_cor = pd.DataFrame.from_dict(Results_test_cor,orient='index')
                ResultsDF_val_final = ResultsDF_val.transpose()
                ResultsDF_test_cor_final = ResultsDF_test_cor.transpose()
                
            else:
                ResultsDF_val_final = []
                ResultsDF_test_cor_final = []
                
        elif ValidationSetCorrection == 1:
            ResultsDF_val = pd.DataFrame.from_dict(Results_val,orient='index')
            ResultsDF_val_final = ResultsDF_val.transpose()
                
            #ValFeatureSetFeatureImportancePath = ResultsDataDir + FeatureSetName + '_PredictedAges_val_FI.csv' # path for saving validation predicted ages feature importance
            #ValFeatureSetFeatureImportanceScaledPath = ResultsDataDir + FeatureSetName + '_PredictedAges_val_FI_Scaled.csv' # path for saving validation predicted ages feature importance
     
            #FeatureImportance_val.to_csv(ValFeatureSetFeatureImportancePath, index=True)
            #FeatureImportanceScaled_val.to_csv(ValFeatureSetFeatureImportanceScaledPath, index=True)        
            
            ResultsDF_test_final = []
            ResultsDF_test_cor_final = []
            
        else:
            ResultsDF_val_final = []
            ResultsDF_test_final = []
            ResultsDF_test_cor_final = []
            
    return ResultsDF_val_final, ResultsDF_test_final, ResultsDF_test_cor_final, ValidationSetExists, TestingSetExists, ValidationSetCorrection
#%% Perform machine learning
results_val, results_test, results_test_cor, ValidationSetExists, TestingSetExists, ValidationSetCorrection = CBA_prediction(TrainingFeatureSetDataDir, ValidationFeatureSetDataDir, TestingFeatureSetDataDir, Results, FeatureSetsList, SelectedAlgorithmsList, FeatureImportanceEstimationMethod, ValidationMethod, Settings)

# save data to results dir
if TestingSetExists == 1:
    results_test.to_csv(ResultsDataDir + 'CBA_estimation_test.csv')
if ValidationSetExists == 1 or ValidationSetCorrection == 1:
    results_val.to_csv(ResultsDataDir + 'CBA_estimation_validation.csv')
if TestingSetExists == 1 and (ValidationSetExists == 1 or ValidationSetCorrection == 1):
    results_val.to_csv(ResultsDataDir + 'CBA_estimation_validation.csv')
    results_test_cor.to_csv(ResultsDataDir + 'CBA_estimation_test_cor.csv')
