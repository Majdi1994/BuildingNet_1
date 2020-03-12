# BuildingNet
VIbration-Based SHM for low - mid and high-rise buildings. This repo contains the main python files needed to run the DL models for this application. The reason for this is to accelerate the process of calculation using Google Colab.

The proposed dataset corresponds to the case of a mid-rise, 10-story-building. The data is recorded under a random shake excitation, the sampling frequency is 1000Hz. The time duration for each record is 256s.

The adopted scenario studies the case of 10 installed sensors on 10 floors (So, each sensor will be responsible of detecting the damage on each floor seperately).

Two folders are included in this repository, the first is 'models' folder and the second is 'data'. 'models' folder contains all the main functions that we need to call in the model_trainer model file.   
