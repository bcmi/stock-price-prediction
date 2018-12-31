## Team Member
Yuyi Gu and Lingjia Li

## Testing Results
Model	Private Board	Public Board
1	CNN-RNN Regression	0.00139 ~ 0.00141	0.00149 ~ 0.00151
2	Linear NN Regression	0.00147	0.00151

## Quick Start
Put all the csv data files in the same directory of the model. 

For model 1, run `lstm_train.py`, then run `lstm_test.py`. The result will be saved in `result.csv` in the same directory.

For model 2, run `lgbm_train.py`, then run `lgbm_test.py`. The result will be saved in `result.csv` in the same directory.
