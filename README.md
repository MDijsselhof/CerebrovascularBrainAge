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

#### Data

This folder consists of the following sub folders: 

- Training : Datasets used for training the model
- Validation : Datasets used for validating the model (can be constructed from the training dataset)
- Testing : Datasets used for testing the model/ predicting age (can be constructed from the training dataset)

All datasets require output of ExploreASL format (.csv) and an Age_sex.csv file containing the subject name, age and sex (1 for male, 0 for female)

#### Matlab

This folder contains the Matlab scripts required for executing the Cerebrovascular Brain-age estimation. ExploreASL is also required (optionally in a seperate directory).

#### Python

This folder contains the MachineLearning.py script, which executes the Machine Learning part. This folder may also contain a virtual environment, in which the MachineLearning.py script should be placed.

## How to start

1. Make sure every requirements described above is fullfilled.
2. Open the Cerebrovascular_Brainage.m script.
    1. Change settings if required
    2. Create `Data` required folders using the script if required
3. Place training-validation-testing data in the newly created folders
4. Run the remainder of the script


