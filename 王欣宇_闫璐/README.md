# AI Course Project

(We do not have `model1` and `model2` folders simply becuase we have implemented all the models in one file `./train.py`. Continue reading to know more.)

Hello to everyone who wishes to review our piece of codes! This README is intended to help you with these confusing codes (my fault I have to say). For a detailed description of our model and data processing, refer to [this report](https://www.overleaf.com/read/bhmbjmyrxjfh).

To run our project:

* Make sure you have the following packages installed:

  * `Tensorflow`;
  * `Joblib`;

* Create a new folder `./data/` and put `train_data.csv` and `test_data.csv` into it;

* Now, just run the script `./train.py` using `python train.py`, and everything should be fine;

* Some options suggested to know about in our project:

  * **Hyper-parameters**: In file `./train.py`, about line 250, in function `train`:

    ```python
    n_inputs = 10
    n_outputs = 1
    n_features = None # get from data
    batch_size = 64
    n_epochs = 10
    ```

    You can modify these hyper-parameters, especially `batch_size` and `n_epochs` to tune the model. The picture below is some of my experiments on the tuning of hyper-paramters:
    ![Hyper-parameters](https://github.com/shawn233/stock-price-prediction/blob/master/%E7%8E%8B%E6%AC%A3%E5%AE%87_%E9%97%AB%E7%92%90/util/hyperparameters.jpg)

  * **Model**: In file `./train.py`, about line 274, in function `train`:

    ```python
    pred = get_simple_lstm_model (inputs_pl, is_training_pl)
    ```

    You can change `get_simple_lstm_model` to `get_cnn_lstm_model` to use the second model in our project, which is much more complicated, and needs about 40 epochs to converge. (Also refer to the picture above.)

  * **Re-process data**: If you find current generated data (train set and dev set) are not satifactory, you can change this option: In file `./train.py`, about line 256, in function `train`:

    ```python
    order_book = OrderBook (batch_size, DATA_DIR, data_regenerate_flag=False)
    ```

    You can change `data_regenerate_flag` to `True` to force the re-processing of data.

Notice: We actually implemented three models in the file `./train.py`. To select models, you actually have three options:

* A simple LSTM model: `get_simplt_lstm_model`;
* A complex CNN-LSTM model: `get_cnn_lstm_model`;
* A deprecated DNN model: `get_dnn_model`.

I have achieved at most 0.00140 for the private score (of course in late submissions). If you are lucky enough, you may get even higher scores~

Now more about the codes, this project mainly comprises of two source files:

* `./train.py` (five hunderd lines), mainly Tensorflow implementation of the following models:

  * A simple LSTM model, defined in function `get_simple_lstm_model`;
  * A CNN-LSTM hybrid model, defined in function `get_cnn_lstm_model`;
  * A DNN model, defined in function `get_dnn_model`.

  Function `train` defined to train the model using processed data.

* `util/data_util.py` (a thousand lines), all our tricks in processing data, with complete and friendly comments. Here is my suggestion on your review of this file:

  * This file contains one thing, a class named `OrderBook`. To get started, you only need to notice one function: `__data_process_procedure_based_on_day_normalization` (Its name is so long that I'll call it `__procedure` in abbreviate);
  * Follow the funtion `__procedure` to know our procedure in processing data; also, refer to the picture below to get a better understanding.
  ![](https://github.com/shawn233/stock-price-prediction/blob/master/%E7%8E%8B%E6%AC%A3%E5%AE%87_%E9%97%AB%E7%92%90/util/data-processing-procedure.jpg)

* You may also get curious about the other two files in the `util` folder, namely, `util/my_tf_util.py`, and `util/plot_util.py`. You may read them if you are really interested, but I do not suggest that because they do not help with the prediction.
