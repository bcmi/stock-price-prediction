## Request
 - python $\ge$ 3.6
 - torch $\ge$ 0.4.0
 - numpy $\ge$  1.14
 - pands $\ge$ 0.23
 - cuda $\ge$ 9.0
 - sklearn $\ge$ 0.19
## Train and Test

There are two files in both the folders.  *db_init2.py*  will prepare the database for you and *DNN_LSTM_net.py* will train the model according to the database. To run our models, please 
1. Adjust the value of *FOLDER*, *TRAIN_SET_DATA_FILE* and *TEST _SET_DATA_FILE* in the *db_init2.py* according to your path.
**or**
2. create a new folder *data* in the current folder and add  the files *train_data.csv* and *test_data.csv* into the new folder.
Then use the command *python DNN_LSTM_net.py*. 