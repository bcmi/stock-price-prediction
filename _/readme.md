1. Run data_preprocessing.py
This will preprocess the train_data.csv, delete repeated rows and save the new file as train_mod.csv. (The only difference between train_data.csv and train_mod.csv is the latter one deletes all repeated rows.) It will also add new features, scale inputs and save the training pair and testing input as three '.npy' file. 
We use 'LastPrice', 'Volume', 'BidPrice', 'BidVolume', 'AskPrice', 'AskVolume' as input features, and we also add three new features: 'BidVolume - AskVolume', 'BidPrice - AskPrice', and 'BidVolume*BidPrice - AskVolume*AskPrice'. We take row difference on the original six features.
The shape of input is (9, 9), because after taking row difference, there are 9 rows left, and we have 9 features. Flatten this (9, 9) matrix to (81,) scaler, use it as the XGBRegrssor input.

2. Run XGB_GridsearchCV.py many times and take the average of all Prediction.csv as the final submission. (We added randomness during the training process, so each training result will be slightly different.)
