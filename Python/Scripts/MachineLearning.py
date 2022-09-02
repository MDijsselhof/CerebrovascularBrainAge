# this script performs the brain age predictions using features and algorithms provided by ExploreASL
# input :
#           TrainingDataDirectory = sys.argv[1] : string containing path to training data directory
#           TestingDataDirectory = sys.argv[2] : string containing path to testing data directory
#           ResultsDataDirectory = sys.argv[3] : string containing path to results data directory
#           FeatureSetList = sys.argv[4] : List containing feature sets used for prediction
#           AlgorithmList = sys.argv[5] : List containing algorithms used for prediction

#%% JUST FOR TESTING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

TrainingDataDir = '/home/mdijsselhof/scratch/Brain_Age/Data/Training/' # test script: 
TestingDataDir = '/home/mdijsselhof/scratch/Brain_Age/Data/Testing/'  # test script: 
ResultsDataDir = '/home/mdijsselhof/scratch/Brain_Age/Data/Results/' # test script: 
ValidationDataDir = '/home/mdijsselhof/scratch/Brain_Age/Data/Validation/' # test script:   
    
FeatureSetsList = ['T1wFLAIRASL']
AlgorithmsList = ['ExtraTrees']
#%% Load modules
# essentials
import argparse
import numpy as np
import torch
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
parser.add_argument("--FeatureSetsList", nargs='+', help="List of feature sets used for machine learning")
parser.add_argument("--AlgorithmsList", nargs='+', help="List of algorithms used for machine learning")

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
print(FeatureSetsList)
SelectedFeatureSetsList = {}
FeatureSets = {'T1w': 'T1w', 
               'FLAIR': 'FLAIR', 
               'ASL': 'ASL', 
               'T1wFLAIR': 'T1wFLAIR', 
               'T1wASL': 'T1wASL', 
               'FLAIRASL': 'FLAIRASL', 
               'T1wFLAIRASL': 'T1wFLAIRASL'}

for SelectedFeatureName in FeatureSetsList:
    SelectedFeatureSetsList[SelectedFeatureName] = FeatureSets.get(SelectedFeatureName) # get algorithms and build new dictonary

# algorithm list
AlgorithmsList = parser.parse_args().AlgorithmsList 
SelectedAlgorithmsList = {}
print(AlgorithmsList)
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

# cerebrovascular brain-age prediction
def CBA_prediction (TrainingFeatureSetDataDir, ValidationFeatureSetDataDir, TestingFeatureSetDataDir, Results_val, Results_test, Results_test_cor, SelectedFeatureSetsList, SelectedAlgorithmsList):
    # select feature set
    for FeatureSet, FeatureSetName in tqdm(SelectedFeatureSetsList.items(),desc='Feature set'):
        
        # create dataframe for saving predicted and chronological brain ages
        PredictedAgeDF_val = pd.DataFrame(columns=(AlgorithmsList))
        PredictedAgeDF_test = pd.DataFrame(columns=(AlgorithmsList))
        PredictedAgeDF_test_cor = pd.DataFrame(columns=(AlgorithmsList))
        
        # load training and testing data for feature set
        TrainingSetPath = TrainingFeatureSetDataDir + 'TrainingSet_' + FeatureSetName + '.tsv' 
        ValidationSetPath = ValidationFeatureSetDataDir + 'ValidationSet_' + FeatureSetName + '.tsv' 
        TestingSetPath = TestingFeatureSetDataDir + 'TestingSet_' + FeatureSetName + '.tsv'

        TrainingSet = pd.read_csv(TrainingSetPath,engine='python', sep='\t')
        ValidationSet = pd.read_csv(ValidationSetPath,engine='python', sep='\t')
        TestingSet = pd.read_csv(TestingSetPath,engine='python', sep='\t')

        X_train, y_train = TrainingSet.drop(['SUBJECT', 'ID', 'Age', 'Sex'], axis=1), TrainingSet['Age']
        X_val, y_val = ValidationSet.drop(['SUBJECT', 'ID', 'Age', 'Sex'], axis=1), ValidationSet['Age']
        X_test, y_test = TestingSet.drop(['SUBJECT', 'ID', 'Age', 'Sex'], axis=1), TestingSet['Age']
        
        # perform standard scaling
        SC = StandardScaler()
        X_train_SC = SC.fit_transform(X_train)
        X_val_SC = SC.transform(X_val)
        X_test_SC = SC.transform(X_test)
        
        # select algorithm
        for AlgorithmName, Algorithm in tqdm(SelectedAlgorithmsList.items(),desc='Algorithms',leave=False):
                       
            # add arguments for some algorithms
            if AlgorithmName not in Algorithm_Arguments.keys():
                kwargs = {}
            else:
                kwargs = Algorithm_Arguments[AlgorithmName]

            Algorithm_instantiated = Algorithm(**kwargs)
            Algorithm_instantiated.fit(X_train_SC, y_train)
            
            # validation set
            y_val_pred = Algorithm_instantiated.predict(X_val_SC)
            mae_val, rmse_val, R2_val, exp_var_val = evaluation_metrics(y_val, y_val_pred)
            
            Results_val['Feature_combo'].append(FeatureSetName)
            Results_val['No. features'].append(len(FeatureSet))
            Results_val['Algorithm'].append(AlgorithmName)
            Results_val['mae'].append(mae_val)
            Results_val['rmse'].append(rmse_val)
            Results_val['R2'].append(R2_val)
            Results_val['explained_var'].append(exp_var_val)
            
            PredictedAgeDF_val[AlgorithmName] = y_val_pred
            
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
            Results_test['No. features'].append(len(FeatureSet))
            Results_test['Algorithm'].append(AlgorithmName)
            Results_test['mae'].append(mae_test)
            Results_test['rmse'].append(rmse_test)
            Results_test['R2'].append(R2_test)
            Results_test['explained_var'].append(exp_var_test)
            
            PredictedAgeDF_test[AlgorithmName] = y_pred_test
            
            # test  corrected set
            y_pred_test_cor = y_pred_test - (RegCoef * y_test + RegIntercept)
            
            mae_test_cor, rmse_test_cor, R2_test_cor, exp_var_test_cor = evaluation_metrics(y_test, y_pred_test_cor)
            
            Results_test_cor['Feature_combo'].append(FeatureSetName)
            Results_test_cor['No. features'].append(len(FeatureSet))
            Results_test_cor['Algorithm'].append(AlgorithmName)
            Results_test_cor['mae'].append(mae_test_cor)
            Results_test_cor['rmse'].append(rmse_test_cor)
            Results_test_cor['R2'].append(R2_test_cor)
            Results_test_cor['explained_var'].append(exp_var_test_cor)
            
            PredictedAgeDF_test_cor[AlgorithmName] = y_pred_test_cor
            
         
        # save predicted ages 
        PredictedAgeDF_val['Chronological_Age'] = y_val
        PredictedAgeDF_test['Chronological_Age'] = y_test
        PredictedAgeDF_test_cor['Chronological_Age'] = y_test
        
        ValFeatureSetPredictedAgePath = ResultsDataDir + FeatureSetName + '_PredictedAges_val.csv' # path for saving predicted ages
        TestFeatureSetPredictedAgePath = ResultsDataDir + FeatureSetName + '_PredictedAges_test.csv' # path for saving predicted ages
        TestCorFeatureSetPredictedAgePath = ResultsDataDir + FeatureSetName + '_PredictedAges_test_cor.csv' # path for saving predicted ages
        
        PredictedAgeDF_val.to_csv(ValFeatureSetPredictedAgePath, index=False)
        PredictedAgeDF_test.to_csv(TestFeatureSetPredictedAgePath, index=False)
        PredictedAgeDF_test_cor.to_csv(TestCorFeatureSetPredictedAgePath, index=False)
        
    ResultsDF_val = pd.DataFrame.from_dict(Results_val,orient='index')
    ResultsDF_test = pd.DataFrame.from_dict(Results_test,orient='index')
    ResultsDF_test_cor = pd.DataFrame.from_dict(Results_test_cor,orient='index')
    ResultsDF_val_final = ResultsDF_val.transpose()
    ResultsDF_test_final = ResultsDF_test.transpose() 
    ResultsDF_test_cor_final = ResultsDF_test_cor.transpose()
    
    return ResultsDF_val_final, ResultsDF_test_final, ResultsDF_test_cor_final
#%% Perform machine learning
results_val, results_test, results_test_cor = CBA_prediction(TrainingFeatureSetDataDir, ValidationFeatureSetDataDir, TestingFeatureSetDataDir, Results_val, Results_test, Results_test_cor, SelectedFeatureSetsList, SelectedAlgorithmsList)
    
# save data to results dir
results_val.to_csv(ResultsDataDir + 'CBA_estimation_validation.csv')
results_test.to_csv(ResultsDataDir + 'CBA_estimation_test.csv')
results_test_cor.to_csv(ResultsDataDir + 'CBA_estimation_test_cor.csv')
