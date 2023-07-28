# this script performs the brain age predictions using features and algorithms provided by ExploreASL
# input :
#           TrainingDataDirectory = sys.argv[1] : string containing path to training data directory
#           TestingDataDirectory = sys.argv[2] : string containing path to testing data directory
#           ResultsDataDirectory = sys.argv[3] : string containing path to results data directory
#           FeatureSetList = sys.argv[4] : List containing feature sets used for prediction
#           AlgorithmList = sys.argv[5] : List containing algorithms used for prediction

#%% JUST FOR TESTING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# TrainingDataDir = '/scratch/mdijsselhof/Cerebrovascular-Brain-age/Data/Training/' # test script: 
# TestingDataDir = '/scratch/mdijsselhof/Cerebrovascular-Brain-age/Data/Testing/'  # test script: 
# ResultsDataDir = '/scratch/mdijsselhof/Cerebrovascular-Brain-age/Data/Results/' # test script: 
# ValidationDataDir = '/scratch/mdijsselhof/Cerebrovascular-Brain-age/Data/Validation/' # test script:   
# FeatureSetsList =         'ATT,ATTTex,CBF,CBFATT,CBFATTTex,CBFTex,T1w,T1wATT,T1wATTTex,T1wCBF,T1wCBFATT,T1wCBFATTTex,T1wCBFTex,T1wTex,Tex'
# AlgorithmsList =  'RandomForest,DecisionTree,XGBoost,BayesianRidge,LinearReg,SVR,Lasso,GPR,ElasticNetCV,ExtraTrees,GradBoost,AdaBoost,KNN,LassoLarsCV,LinearSVR,RidgeCV,SGDReg,Ridge,LassoLars,ElasticNet,RVM,RVR'
# FeatureSetsList = FeatureSetsList.split(',') 
# AlgorithmsList = AlgorithmsList.split(',') 

#%% Load modules
# essentials
import argparse
import os
import numpy as np
import torch
import shap
import pandas as pd
from scipy import stats
from tqdm import tqdm
from time import sleep

# modelling setup
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

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
from sklearn_rvm import EMRVR
from skrvm import RVR # pip install https://github.com/JamesRitchie/scikit-rvm/archive/master.zip

#%% Parse input arguments
parser = argparse.ArgumentParser(description = "Cerebrovascular Brain-age prediction script")
parser.add_argument("--TrainingDataDir", help="Path to training data folder containing constructed feature sets",type=str)
parser.add_argument("--ValidationDataDir", help="Path to training data folder containing constructed feature sets",type=str)
parser.add_argument("--TestingDataDir", help="Path to testing data folder containing constructed feature sets",type=str)
parser.add_argument("--ResultsDataDir", help="Path to results data folder containing future predictions",type=str)
parser.add_argument("--FeatureSetsList", help="List of feature sets used for machine learning")
parser.add_argument("--AlgorithmsList", help="List of algorithms used for machine learning")

#%% DataPaths
TrainingDataDir = parser.parse_args().TrainingDataDir # test script: 
ValidationDataDir = parser.parse_args().ValidationDataDir # test script: 
TestingDataDir = parser.parse_args().TestingDataDir   # test script: 
ResultsDataDir = parser.parse_args().ResultsDataDir   # test script: 

TrainingFeatureSetDataDir = TrainingDataDir + 'FeatureSets/' # test script: 
ValidationFeatureSetDataDir = ValidationDataDir + 'FeatureSets/' # test script: 
TestingFeatureSetDataDir = TestingDataDir + 'FeatureSets/' # test script: 

# feature set list
FeatureSetsList = parser.parse_args().FeatureSetsList 
FeatureSetsList = FeatureSetsList.split(',') 

# algorithm list
AlgorithmsList = parser.parse_args().AlgorithmsList
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

# results
Results_val = {'Feature_combo': [],
             'No. features': [],
             'Algorithm': [],
             'mae' : [],
             'rmse' : [],
             'R2' : [],
             'explained_var' : []}

# results
Results_test = {'Feature_combo': [],
             'No. features': [],
             'Algorithm': [],
             'mae' : [],
             'rmse' : [],
             'R2' : [],
             'explained_var' : []}

Results_test_cor = {'Feature_combo': [],
             'No. features': [],
             'Algorithm': [],
             'mae' : [],
             'rmse' : [],
             'R2' : [],
             'explained_var' : []}


#%% Functions
# evaluation metrics
def evaluation_metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    R2 = r2_score(y_test, y_pred)
    exp_var = explained_variance_score(y_test, y_pred)

    return mae, rmse, R2, exp_var

# feature importance
def Feature_Importance(X_train, X_test, y_test, AlgorithmName, Algorithm_instantiated, SHAP):
    if SHAP == 0:
        if AlgorithmName == 'XGBoost' or AlgorithmName == 'BayesianRidge' or AlgorithmName == 'LinearReg' or AlgorithmName == 'SVR' or AlgorithmName == 'RidgeCV':
            Result = np.abs(Algorithm_instantiated.coef_)
        elif    AlgorithmName == 'Lasso' or AlgorithmName == 'LassoLarsCV' or AlgorithmName == 'LinearSVR' or AlgorithmName == 'SGDReg' or AlgorithmName == 'ElasticNetCV':
            Result = np.abs(Algorithm_instantiated.coef_)
        elif AlgorithmName == 'Ridge' or AlgorithmName == 'LassoLars' or AlgorithmName == 'ElasticNet' or AlgorithmName == 'RVR':
            Result = np.abs(Algorithm_instantiated.coef_)
        elif AlgorithmName == 'RandomForest' or AlgorithmName == 'DecisionTree' or AlgorithmName == 'GPR' or AlgorithmName == 'ExtraTrees' or AlgorithmName == 'GradBoost' or AlgorithmName == 'AdaBoost' or AlgorithmName == 'KNN' or AlgorithmName == 'RVM':
            Result = permutation_importance(Algorithm_instantiated, X_test, y_test, n_repeats=100, random_state=42)
        FeatureImportance = pd.DataFrame(Result.importances_mean, index=X_test.columns,columns=[AlgorithmName])
    else:
        def calculate_shap_feature_importance(Algorithm_instantiated, X_train):
            try: # for algorithms recognised by SHAP
                explainer = shap.Explainer(Algorithm_instantiated,X_train)
                shap_values = explainer.shap_values(X_train)
            except:
                try: # for algorithms not recognised by SHAP
                    explainer = shap.Explainer(Algorithm_instantiated.predict,X_train)
                    shap_values = explainer.shap_values(X_train) 
                except: # for regressions using kernels
                    explainer = shap.KernelExplainer(Algorithm_instantiated.predict,X_train)
                    shap_values = explainer.shap_values(X_train) 
            return shap_values
        FeatureImportance = {}
        shap_values = abs(calculate_shap_feature_importance(Algorithm_instantiated, X_train))
        FeatureImportance = pd.DataFrame(shap_values,columns=X_train.columns)
        FeatureImportance = pd.DataFrame(FeatureImportance.mean(axis=0),columns=[AlgorithmName])
    return FeatureImportance

# cerebrovascular brain-age prediction
def CBA_prediction (TrainingFeatureSetDataDir, ValidationFeatureSetDataDir, TestingFeatureSetDataDir, Results_val, Results_test, Results_test_cor, FeatureSetsList, SelectedAlgorithmsList):
    # select feature set
    for FeatureSetName in tqdm(FeatureSetsList):
        
        # create dataframe for saving predicted and chronological brain ages
        PredictedAgeDF_val = pd.DataFrame(columns=(AlgorithmsList))
        PredictedAgeDF_test = pd.DataFrame(columns=(AlgorithmsList))
        PredictedAgeDF_test_cor = pd.DataFrame(columns=(AlgorithmsList))
        
        # load training and testing data for feature set
        TrainingSetPath = TrainingFeatureSetDataDir + 'TrainingSet_' + FeatureSetName + '.tsv' 
        ValidationSetPath = ValidationFeatureSetDataDir + 'ValidationSet_' + FeatureSetName + '.tsv' 
        if len(os.listdir(ValidationDataDir)) == 0:
            ValidationSetExists = 0
        else:
            ValidationSetExists = 1
            
        
        TestingSetPath = TestingFeatureSetDataDir + 'TestingSet_' + FeatureSetName + '.tsv'

        TrainingSet = pd.read_csv(TrainingSetPath,engine='python', sep='\t')
        if ValidationSetExists == 1:
            ValidationSet = pd.read_csv(ValidationSetPath,engine='python', sep='\t')
            
        TestingSet = pd.read_csv(TestingSetPath,engine='python', sep='\t')

        X_train, y_train = TrainingSet.drop(['participant_id', 'ID', 'Age', 'Sex'], axis=1), TrainingSet['Age']
        if ValidationSetExists == 1:
            X_val, y_val = ValidationSet.drop(['participant_id', 'ID', 'Age', 'Sex'], axis=1), ValidationSet['Age']
            
        X_test, y_test = TestingSet.drop(['participant_id', 'ID', 'Age', 'Sex'], axis=1), TestingSet['Age']
        # perform standard scaling
        SC = StandardScaler()
        X_train_SC = SC.fit_transform(X_train)
        if ValidationSetExists == 1:
            X_val_SC = SC.transform(X_val)
            
        X_test_SC = SC.transform(X_test)
        
        index = 0
        
        # select algorithm
        for AlgorithmName, Algorithm in tqdm(SelectedAlgorithmsList.items(),desc='Algorithms',leave=False):
            print(index)
            # add arguments for some algorithms
            if AlgorithmName not in Algorithm_Arguments.keys():
                kwargs = {}
            else:
                kwargs = Algorithm_Arguments[AlgorithmName]

            Algorithm_instantiated = Algorithm(**kwargs)
            Algorithm_instantiated.fit(X_train_SC, y_train)
            
            # validation set
            if ValidationSetExists == 1:
                y_val_pred = Algorithm_instantiated.predict(X_val_SC)
                mae_val, rmse_val, R2_val, exp_var_val = evaluation_metrics(y_val, y_val_pred)
            
                Results_val['Feature_combo'].append(FeatureSetName)
                Results_val['No. features'].append(len(FeatureSetName))
                Results_val['Algorithm'].append(AlgorithmName)
                Results_val['mae'].append(mae_val)
                Results_val['rmse'].append(rmse_val)
                Results_val['R2'].append(R2_val)
                Results_val['explained_var'].append(exp_var_val)
            
                PredictedAgeDF_val[AlgorithmName] = y_val_pred
                
                if index == 0:
                    FeatureImportance_val = Feature_Importance(X_val, y_val, AlgorithmName, Algorithm_instantiated, 1) # test set feature importance first iteration
                else :
                    FeatureImportance_val_append = Feature_Importance(X_val, y_val, AlgorithmName, Algorithm_instantiated, 1) # test set feature importance after first iteration
                    FeatureImportance_val = FeatureImportance_val.insert(len(FeatureImportance_val),AlgorithmName,FeatureImportance_val_append.values) # combined feature importance of all argorithms, per feature set
                
                # determine age bias 
                Age_delta = y_val_pred - y_val
                #Age_delta_r = Age_delta.values.reshape(-1, 1)
                y_val_r = y_val.values.reshape(-1,1)
                LinearReg = LinearRegression().fit(y_val_r,Age_delta)
                RegCoef = LinearReg.coef_
                RegIntercept = LinearReg.intercept_
            
            # test set
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
            
            if index == 0:
                FeatureImportance_test = Feature_Importance(X_train, X_test, y_test, AlgorithmName, Algorithm_instantiated, 1) # test set feature importance first iteration
            else :
                FeatureImportance_test_append= Feature_Importance(X_train,X_test, y_test, AlgorithmName, Algorithm_instantiated, 1) # test set feature importance after first iteration
                FeatureImportance_test[AlgorithmName] = FeatureImportance_test_append[AlgorithmName].values # combined feature importance of all argorithms, per feature set
            
            
            
            if ValidationSetExists == 1:
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
               
               if index == 0:
                    FeatureImportance_test_cor = Feature_Importance(X_test, y_test, AlgorithmName, Algorithm_instantiated, 1) # test set feature importance first iteration
               else :
                    FeatureImportance_test_cor_append = Feature_Importance(X_test, y_test, AlgorithmName, Algorithm_instantiated, 1) # test set feature importance after first iteration     
                    FeatureImportance_test_cor = FeatureImportance_test_cor.insert(len(FeatureImportance_test_cor),AlgorithmName,FeatureImportance_test_cor_append.values)  # combined feature importance of all argorithms, per feature set


            index += 1 # add index for next interation to append feature importances         
        # save predicted ages 
        
        PredictedAgeDF_test['Chronological_Age'] = y_test
        if ValidationSetExists == 1:
            PredictedAgeDF_val['Chronological_Age'] = y_val
            PredictedAgeDF_test_cor['Chronological_Age'] = y_test
        
        TestFeatureSetPredictedAgePath = ResultsDataDir + FeatureSetName + '_PredictedAges_test.csv' # path for saving predicted ages
        PredictedAgeDF_test.to_csv(TestFeatureSetPredictedAgePath, index=False)
        ResultsDF_test = pd.DataFrame.from_dict(Results_test,orient='index')
        ResultsDF_test_final = ResultsDF_test.transpose() 
        
        TestFeatureSetFeatureImportancePath = ResultsDataDir + FeatureSetName + '_PredictedAges_test_FI.csv' # path for saving predicted ages feature importance
        FeatureImportance_test.to_csv(TestFeatureSetFeatureImportancePath, index=True)

        
        if ValidationSetExists == 1:
            ValFeatureSetPredictedAgePath = ResultsDataDir + FeatureSetName + '_PredictedAges_val.csv' # path for saving predicted ages
            TestCorFeatureSetPredictedAgePath = ResultsDataDir + FeatureSetName + '_PredictedAges_test_cor.csv' # path for saving predicted ages
            PredictedAgeDF_val.to_csv(ValFeatureSetPredictedAgePath, index=False)
            PredictedAgeDF_test_cor.to_csv(TestCorFeatureSetPredictedAgePath, index=False)
        
            ResultsDF_val = pd.DataFrame.from_dict(Results_val,orient='index')
            ResultsDF_test_cor = pd.DataFrame.from_dict(Results_test_cor,orient='index')
            ResultsDF_val_final = ResultsDF_val.transpose()
            ResultsDF_test_cor_final = ResultsDF_test_cor.transpose()
            
            TestFeatureSetCorFeatureImportancePath = ResultsDataDir + FeatureSetName + '_PredictedAges_test_cor_FI.csv' # path for saving corrected predicted ages feature importance
            FeatureImportance_test_cor.to_csv(TestFeatureSetCorFeatureImportancePath, index=True)
            
            ValFeatureSetFeatureImportancePath = ResultsDataDir + FeatureSetName + '_PredictedAges_val_FI.csv' # path for saving validation predicted ages feature importance
            FeatureImportance_val.to_csv(ValFeatureSetFeatureImportancePath, index=True)
            
        else:
            ResultsDF_val_final = []
            ResultsDF_test_cor_final = []
            
    return ResultsDF_val_final, ResultsDF_test_final, ResultsDF_test_cor_final, ValidationSetExists
#%% Perform machine learning
results_val, results_test, results_test_cor, ValidationSetExists = CBA_prediction(TrainingFeatureSetDataDir, ValidationFeatureSetDataDir, TestingFeatureSetDataDir, Results_val, Results_test, Results_test_cor, FeatureSetsList, SelectedAlgorithmsList)
    
# save data to results dir
results_test.to_csv(ResultsDataDir + 'CBA_estimation_test.csv')
if ValidationSetExists == 1:
    results_val.to_csv(ResultsDataDir + 'CBA_estimation_validation.csv')
    results_test_cor.to_csv(ResultsDataDir + 'CBA_estimation_test_cor.csv')
