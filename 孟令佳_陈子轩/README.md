## Stock Price Prediction

## Environment
- python 3.6.2
- pytorch 0.4.1 or higher
- numpy 1.13.3

## Before training
- before training and after training, you need to check whether folder `data/model_LSTM` and `data/model_DNN` is created, where our trained model parameters is stored
- `KEEP_ON` in each model is to determine whether to load previous trained model, set it to `True` to keep on training. For each epoch, the model will be saved in folder `data/model_LSTM` and `data/model_DNN`. 
- make sure you got `test_data.csv` and `train_data.csv` in our data folder
## Hyperparameters
### LSTM
- EPOCH = 50
- BATCH_SIZE = 32
- TIME_STEP = 10          
- INPUT_SIZE = 8        
- LR = 0.001              
- DATASIZE = 427700
### DNN
- INPUT_SIZE = 8
- seq_len = 10
- BATCH_SIZE = 16
- LR = 1e-4
- EPOCH = 2
## How to run
If you set it up properly, just type `python + <modename.py>` and start training!
