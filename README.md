# CMSY++
Official repository of the CMSY++ data-limited stock assessment method.

# Instructions
All files should be copied in the same directory and the required libraries should be installed.

# Main files to run the CMSY++ method
CMSY++16.R: R-code executing the CMSY++ and BSM models

ffnn.bin: File containing the trained neural network used by CMSY++

Train_Catch_9e.csv: Catch and abundance data of the 400 stocks used for testing and training, in the format required by CMSY++
 
Train_ID_9e.csv: Stock identification and prior settings (defaults or user input), in the format required by CMSY++

# Additional files
NN_Bk_19a.R: R-code used to train and evaluate the neural network

Out_Train_ID_9e.csv: File containing the CMSY++ and BSM results for the 400 test stocks. This file was also used for training the neural network. 

Out_March032021_Train_ID_9c.csv: Data file used by NN_Bk_19a.R

Train_Catch_9b.csv: Data file used by NN_Bk_19a.R

SimSpecCPUE_4_NA.csv: Prior settings for running CMSY++ and BSM with simulated data

SimCatchCPUE_4.csv: Simulated catch data

CMSY++16HW.R: Version of CMSY++16.R cleaned up by Henning Winker, suggested for use with new data (will become version 17 after more testing)
