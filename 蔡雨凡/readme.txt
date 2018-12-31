To run our project:

Make sure you have the following packages installed:

Tensorflow;
keras
xgboost
pandas

for the LSTM model
put train_data.csv  into folder data;
put test_data.csv into folder test_data;

Now,first run the process_data.py in the folder test_data;
and then run the run.py 
you can get your result

Some options suggested to know in our project:
Hyper-parameters: In file config.json and config_test.json
You can change these hyper-parameters, especially batch_size and n_epochs to tune the model.

xgboost
just put the train_data.csv and test_data.csv in the folder xgboost, then run the xgboost.py