 # encoding: utf-8
# file: data_util.py
# author: shawn233

from __future__ import print_function
import os
import sys
import time
import numpy as np
from scipy import stats
import joblib

'''
Functions:
1. divide data into training set, dev set and test set (7:1:2)
2. provide function `next_batch()`, returns the next batch in one epoch;
   provide function `reset_batch()`, to reset the batch for a new epoch.

Usage tips:
1. Assume memory is large enough to store all data, will use `readlines()` to read data;
'''

'''
TODO:
1. askvolume and bidvolume use accumulation, volume use increment; (done)
2. askprice(t) = askprice(t)/midprice(t)-1, bidprice(t) = bidprice(t)/midprice(t) - 1; midprice(t) = midprice(t)/midprice(t-1)-1; (done)
3. Use morning mean and stddev to normalize;
'''

TRAIN_INPUTS_FILENAME = 'train_inputs.npy'
TRAIN_LABELS_FILENAME = 'train_labels.npy'
DEV_INPUTS_FILENAME = 'dev_inputs.npy'
DEV_LABELS_FILENAME = 'dev_labels.npy'
TEST_INPUTS_FILENAME = 'test_inputs.dump'
TEST_LABELS_FILENAME = 'test_labels.dump'

TRAIN_MEANS_FILENAME = 'train_means.npy'
TRAIN_STDDEVS_FILENAME = 'train_stddevs.npy'
DEV_MEANS_FILENAME = 'dev_means.npy'
DEV_STDDEVS_FILENAME = 'dev_stddevs.npy'
TEST_MEANS_FILENAME = 'test_means.dump'
TEST_STDDEVS_FILENAME = 'test_stddevs.dump'

TRAIN_DATA_FILENAME = 'train_data.csv'
TEST_DATA_FILENAME = 'test_data.csv'

TRAIN_DATA_PRE_PROCESSED_FILENAME = 'train_preprocessed.npy'
TEST_DATA_PRE_PROCESSED_FILENAME = 'test_preprocessed.npy'
PRE_PROCESS_RECORD_FILENAME = 'preprocess.txt'


class OrderBook:

    '''
    Order book class, designed mainly for data input
    '''

    def __init__ (self, batch_size, data_dir, 
        num_inputs=10,\
        num_labels=20,\
        data_regenerate_flag=False):
        '''
        Initialization, open the files and set the arguments

        Args:
        - batch_size: int;
        - data_dir: string, directory of the data
        - data_regenerate_flag: bool, True if re-process data, False if use stored data
        '''

        self._batch_size = batch_size
        self.batch_ind = 0
        self._data_dir = data_dir
        self._num_inputs = num_inputs
        self._num_labels = num_labels
        self.num_features = None # will be later set after processing data

        # vars for training set
        self.train_inputs = None
        self.train_labels = None
        self.train_means = None
        self.train_stddevs = None

        # vars for dev set
        self.dev_inputs = None
        self.dev_labels = None
        self.dev_means = None
        self.dev_stddevs = None

        # vars for test set
        self.test_inputs_list = None
        self.test_labels_list = None
        self.test_means_list = None
        self.test_stddevs_list = None

        # var for recording index in data matrix
        self.index = {
            'Date':1,
            'Time':2,
            'MidPrice':3,
            'LastPrice':4,
            'Volume':5,
            'BidPrice1':6,
            'BidVolume1':7,
            'AskPrice1':8,
            'AskVolume1':9,
            'TimeStamp':10
        }

        all_file_exist_flag = True
        for filename in [TRAIN_INPUTS_FILENAME, TRAIN_LABELS_FILENAME, TRAIN_MEANS_FILENAME, TRAIN_STDDEVS_FILENAME,\
                        DEV_INPUTS_FILENAME, DEV_LABELS_FILENAME, DEV_MEANS_FILENAME, DEV_STDDEVS_FILENAME,\
                        TEST_INPUTS_FILENAME, TEST_MEANS_FILENAME, TEST_STDDEVS_FILENAME]:
            if data_regenerate_flag or not os.path.exists (os.path.join (self._data_dir, filename)):
                self.__data_process_procedure_based_on_day_normalization()
                all_file_exist_flag = False
                break

        if all_file_exist_flag:
            print ("Loading train data...")
            self.train_inputs = np.load (os.path.join (self._data_dir, TRAIN_INPUTS_FILENAME))
            self.train_labels = np.load (os.path.join (self._data_dir, TRAIN_LABELS_FILENAME))
            self.train_means = np.load (os.path.join (self._data_dir, TRAIN_MEANS_FILENAME))
            self.train_stddevs = np.load (os.path.join (self._data_dir, TRAIN_STDDEVS_FILENAME))

            self.dev_inputs = np.load (os.path.join (self._data_dir, DEV_INPUTS_FILENAME))
            self.dev_labels = np.load (os.path.join (self._data_dir, DEV_LABELS_FILENAME))
            self.dev_means = np.load (os.path.join (self._data_dir, DEV_MEANS_FILENAME))
            self.dev_stddevs = np.load (os.path.join (self._data_dir, DEV_STDDEVS_FILENAME))

            print ("Loading test data...")
            self.test_inputs_list = joblib.load (os.path.join (self._data_dir, TEST_INPUTS_FILENAME))
            self.test_means_list = joblib.load (os.path.join (self._data_dir, TEST_MEANS_FILENAME))
            self.test_stddevs_list = joblib.load (os.path.join (self._data_dir, TEST_STDDEVS_FILENAME))

        
        if self.train_inputs is not None:
            self.num_features = self.train_inputs.shape[2]





    @property
    def batch_size (self):
        return self._batch_size

    
    @batch_size.setter
    def batch_size (self, value):
        self._batch_size = value

    
    @property
    def data_dir (self):
        return self._data_dir


    @data_dir.setter
    def data_dir (self, value):
        self._data_dir = value


    @property
    def num_samples (self):
        '''
        Number of training samples
        '''

        return self.train_inputs.shape[0]

    
    @property
    def num_batches (self):
        '''
        Maximum number of batches that can be provided in one epoch
        '''
        return int (self.num_samples / self.batch_size)




    def __data_process_procedure_based_on_day_normalization (self):
        '''
        Feature the normalization for each day
        '''

        # train data
        print ("Reading train data...")
        train_data_matrix = self.__read_data_matrix (os.path.join (self._data_dir, TRAIN_DATA_FILENAME))
        train_data_matrix = self.__clean_data_matrix (train_data_matrix)
        
        print ("Dividing train data by half-days...")
        train_day_matrix_list = self.__divide_by_day (train_data_matrix)
        for i in range (len(train_day_matrix_list)):
            #train_day_matrix_list[i] = self.__price_preprocess (train_day_matrix_list[i])
            train_day_matrix_list[i] = self.__volume_subtraction (train_day_matrix_list[i])

        # shuffle by half-day
        
        indices = np.arange (len(train_day_matrix_list))
        np.random.shuffle (indices)
        new_train_day_matrix_list = []
        for ind in indices:
            new_train_day_matrix_list.append (train_day_matrix_list[ind])
        train_day_matrix_list = new_train_day_matrix_list
        
        np.random.shuffle (train_day_matrix_list)

        print ("Running day normalization...")
        train_day_matrix_list, train_day_mean_list, train_day_stddev_list, train_base_index = self.__day_normalization (train_day_matrix_list)

        print ("Generating samples...")
        sample_inputs_list, sample_labels_list, train_means_list, train_stddevs_list, train_base_index2 = \
                                            self.__generate_raw_samples (train_day_matrix_list, train_day_mean_list, train_day_stddev_list)
        for i in range (len(sample_inputs_list)):
            for ind in [self.index['Volume']]: #, self.index['BidVolume1'], self.index['AskVolume1']]:
                sample_inputs_list[i][0, ind-train_base_index] = 0.0 # TODO improve IV of volumes, may learn from data
        assert train_base_index == train_base_index2 # so that I will use train_base_index in the following
        train_means_list = np.asarray (train_means_list)
        train_stddevs_list = np.asarray (train_stddevs_list)

        print ("Selecting and dividing samples...")
        sample_inputs_list, sample_labels_list, train_means_list, train_stddevs_list, train_base_index3 = \
                self.__sample_selection (sample_inputs_list, sample_labels_list, train_means_list, train_stddevs_list, train_base_index)
        assert train_base_index == train_base_index3
        # remove mid price and time stamp from feature
        for i in range (len (sample_inputs_list)):
            sample_inputs_list[i] = sample_inputs_list[i][:, self.index['LastPrice']-train_base_index:self.index['TimeStamp']-train_base_index]
        train_inputs_list, train_labels_list, train_means_list, train_stddevs_list,\
            dev_inputs_list, dev_labels_list, dev_means_list, dev_stddevs_list = \
            self.__sample_division (sample_inputs_list, sample_labels_list, train_means_list, train_stddevs_list, train_fraction=0.9)
        
        print ("Saving train data...")
        self.train_inputs = np.asarray (train_inputs_list, dtype=np.float32)
        self.train_labels = np.asarray (train_labels_list, dtype=np.float32).reshape([-1, 1])
        self.train_means = np.asarray(train_means_list)
        self.train_stddevs = np.asarray(train_stddevs_list)
        self.dev_inputs = np.asarray (dev_inputs_list, dtype=np.float32)
        self.dev_labels = np.asarray (dev_labels_list, dtype=np.float32).reshape([-1, 1])
        self.dev_means = np.asarray(dev_means_list)
        self.dev_stddevs = np.asarray(dev_stddevs_list)
        print ("train inputs shape", self.train_inputs.shape)
        print ("train labels shape", self.train_labels.shape)
        print ("train means shape", self.train_means.shape)
        print ("train stddevs shape", self.train_stddevs.shape)
        print ("dev inputs shape", self.dev_inputs.shape)
        print ("dev labels shape", self.dev_labels.shape)
        print ("dev means shape", self.dev_means.shape)
        print ("dev stddevs shape", self.dev_stddevs.shape)
        np.save (os.path.join (self._data_dir, TRAIN_INPUTS_FILENAME), self.train_inputs)
        np.save (os.path.join (self._data_dir, TRAIN_LABELS_FILENAME), self.train_labels)
        np.save (os.path.join (self._data_dir, TRAIN_MEANS_FILENAME), self.train_means)
        np.save (os.path.join (self._data_dir, TRAIN_STDDEVS_FILENAME), self.train_stddevs)
        np.save (os.path.join (self._data_dir, DEV_INPUTS_FILENAME), self.dev_inputs)
        np.save (os.path.join (self._data_dir, DEV_LABELS_FILENAME), self.dev_labels)
        np.save (os.path.join (self._data_dir, DEV_MEANS_FILENAME), self.dev_means)
        np.save (os.path.join (self._data_dir, DEV_STDDEVS_FILENAME), self.dev_stddevs)

        # test data
        test_data_matrix = self.__read_data_matrix (os.path.join (self._data_dir, TEST_DATA_FILENAME))
        test_data_matrix = test_data_matrix[142*self._num_inputs:, :]
        
        test_day_matrix_list = self.__divide_by_day (test_data_matrix)
        print ("Number of half days in test data", len(test_day_matrix_list))
        for i in range (len (test_day_matrix_list)):
            #test_day_matrix_list[i] = self.__price_preprocess (test_day_matrix_list[i])
            test_day_matrix_list[i] = self.__volume_subtraction(test_day_matrix_list[i])

        test_day_matrix_list, test_mean_list, test_stddev_list, test_base_index = self.__day_normalization (test_day_matrix_list)
        self.test_means_list = np.asarray (test_mean_list)[:, self.index['MidPrice'] - test_base_index]
        self.test_stddevs_list = np.asarray (test_stddev_list)[:, self.index['MidPrice'] - test_base_index]

        print ("test means list shape", self.test_means_list.shape)
        print ("test stddevs list shape", self.test_stddevs_list.shape)

        self.test_inputs_list = [] # a list of 3-d np arrays
        for i in range (len (test_day_matrix_list)):
            test_day_matrix = test_day_matrix_list[i]
            test_day_inputs_list, _ = self.__parse_test_data (test_day_matrix)
            for j in range (len(test_day_inputs_list)):
                for ind in [self.index['Volume']]: #, self.index['BidVolume1'], self.index['AskVolume1']]:
                    test_day_inputs_list[j][0, ind-test_base_index] = 0.0
                test_day_inputs_list[j] = test_day_inputs_list[j][:, self.index['LastPrice']-test_base_index:self.index['TimeStamp']-test_base_index]
            test_day_inputs_list = np.asarray(test_day_inputs_list, dtype=np.float32)
            #print ('test day inputs list shape', test_day_inputs_list.shape)
            self.test_inputs_list.append (test_day_inputs_list)
  
        assert len(self.test_means_list) == len(self.test_inputs_list)

        print ("Saving test data...")
        joblib.dump (self.test_inputs_list, os.path.join (self._data_dir, TEST_INPUTS_FILENAME))
        joblib.dump (self.test_means_list, os.path.join (self._data_dir, TEST_MEANS_FILENAME))
        joblib.dump (self.test_stddevs_list, os.path.join (self._data_dir, TEST_STDDEVS_FILENAME))




    def __read_data_matrix(self, in_filename):
        '''
        Read the train data matrix

        Args:
        - in_filename: string, input file name;

        Returns:
        - data_matrix: 2-d np matrix, dtype=<U32; 
        '''
        in_f = open (in_filename, 'r')

        data_matrix = []

        in_f.readline()
        for raw_line in in_f:
            # jump through empty lines
            line = raw_line.strip()
            if line == '':
                continue
            # process csv line
            line = line.split (',')
            line.append (self.__get_time_stamp(line[self.index['Date']], line[self.index['Time']]))
            data_matrix.append (line)
        
        in_f.close()

        data_matrix = np.asarray (data_matrix, dtype=np.dtype('U32'))
        print ('data matrix dtype', data_matrix.dtype) # if not <U32, care possible type conversion errors
        print ('data matrix shape:', data_matrix.shape)
        
        return data_matrix



    def __divide_by_day (self, input_matrix):
        '''
        Divide train data by half-days

        Args:
        - input_matrix: 2-d np matrix, dtype=<U32;

        Returns:
        - day_matrix_list: a list of 2-d np matrix;
        '''
        
        prev_date = None
        day_split_pos_list = [] # records the index of the first row in a new day

        for i in range (input_matrix.shape[0]):
            if prev_date is None or prev_date != input_matrix[i, self.index['Date']]:
                # a new day
                day_split_pos_list.append (i)
                prev_date = input_matrix[i, self.index['Date']]
            elif int(input_matrix[i, self.index['TimeStamp']]) - int(input_matrix[i-1, self.index['TimeStamp']]) > 1800:
                # split afternoon from morning
                day_split_pos_list.append (i)

        print ('total number of half-days:', len (day_split_pos_list))

        day_matrix_list = []
        for i in range (len (day_split_pos_list)-1):
            day_matrix_list.append (input_matrix[day_split_pos_list[i]:day_split_pos_list[i+1], :])
        day_matrix_list.append (input_matrix[day_split_pos_list[-1]:, :])

        return day_matrix_list






    def __clean_data_matrix (self, data_matrix):
        '''
        Remove rows that are too close (time stamp interval < 3.0)

        Args:
        - data_matrix;

        Returns:
        - data_matrix: cleaned data matrix;
        '''
        rows_to_delete = []
        print ("Data matrix shape before cleaning", data_matrix.shape)
        for i in range (1, data_matrix.shape[0]):
            if int(data_matrix[i, self.index['TimeStamp']]) - int(data_matrix[i-1, self.index['TimeStamp']]) <= 2:
                rows_to_delete.append (i)
                data_matrix[i, self.index['TimeStamp']] = data_matrix[i-1, self.index['TimeStamp']]
        
        data_matrix = np.delete (data_matrix, rows_to_delete, axis=0)
        print ("Total number of rows to delete", len (rows_to_delete))
        print ("Data matrix shape after cleaning", data_matrix.shape)

        return data_matrix




    def __day_normalization (self, day_matrix_list):
        '''
        Run normalization for each half-day, time stamp is not included!

        Args:
        - day_matrix: a list of 2-d np array, each is a half-day data;

        Returns:
        - day_matrix_list: normalized day_matrix_list;
        - mean_list: a list of 1-d array, each is means for all features;
        - stddev_list: a list of 1-d array, each is stddevs for all features;
        - base_index: int, the start index of mean_list and stddev_list;
        '''

        base_index = self.index['MidPrice']
        mean_list = []
        stddev_list = []

        for i in range (len (day_matrix_list)):
            day_matrix = day_matrix_list[i]
            part_day_matrix = day_matrix[:, base_index:self.index['TimeStamp']].astype (np.float32)
            part_mean = np.mean (part_day_matrix, axis=0)
            part_stddev = np.std (part_day_matrix, axis=0)
            day_matrix[:, base_index:self.index['TimeStamp']] = ((part_day_matrix - part_mean) / part_stddev).astype (np.str)
            day_matrix_list[i] = day_matrix
            mean_list.append (part_mean)
            stddev_list.append (part_stddev)
        
        return day_matrix_list, mean_list, stddev_list, base_index




    def __price_preprocess (self, data_matrix):
        '''
        Pre-process price data, including: midprice, bidprice1, and askprice1:
        - bidprice1 (t) = bidprice1(t) / midprice (t) - 1;
        - askprice1 (t) = askprice1(t) / midprice (t) - 1;
        - midprice (t) = midprice (t) / midprice (t-1) - 1; (not implemented currently)
        
        Goal is to make prices more smooth.

        Args:
        - data_matrix: 2-d np matrix;

        Returns:
        - data_matrix: 2-d np matrix;
        '''
        base_index = self.index['MidPrice']
        part_data_matrix = data_matrix[:, base_index:].astype (np.float32)
        new_part_data_matrix = np.copy (part_data_matrix)

        for i in range (1, data_matrix.shape[0]):
            for ind in [self.index['BidPrice1'], self.index['AskPrice1']]:
                new_part_data_matrix[i, ind-base_index] = (part_data_matrix[i, ind-base_index] / \
                        part_data_matrix[i, self.index['MidPrice']-base_index]) - 1.0
            '''
            new_part_data_matrix[i, self.index['MidPrice']-base_index] = \
                    (part_data_matrix[i, self.index['MidPrice']-base_index] / \
                    part_data_matrix[i-1, self.index['MidPrice']-base_index]) - 1.0
            '''

        #new_part_data_matrix[0, self.index['MidPrice']-base_index] = 0.0
        data_matrix[:, base_index:] = new_part_data_matrix.astype (np.dtype('U32'))

        return data_matrix




    def __volume_subtraction (self, data_matrix):
        '''
        Subtract volumes

        Args:
        - data_matrix: 2-d np matrix;

        Returns:
        - data_matrix: 2-d np matrix;
        '''

        volume_indices = [self.index['Volume']] # , self.index['BidVolume1'], self.index['AskVolume1']]
        for ind in volume_indices:
            part_data_matrix = data_matrix[:, ind:ind+1].astype (np.float32)
            for i in range (part_data_matrix.shape[0]-1, 0, -1):
                part_data_matrix[i, 0] = part_data_matrix[i, 0] - part_data_matrix[i-1, 0]
            part_data_matrix[0, 0] = 0.0
            data_matrix[:, ind:ind+1] = part_data_matrix.astype(np.str)

        return data_matrix 




    def __generate_raw_samples (self, day_matrix_list, train_day_mean_list, train_day_stddev_list):
        '''
        From the list of day matrix, generate raw samples, here raw means each input contains
        50 records, which will be used in sample selection, and will be cut to 10 records there.
        
        Note: Samples generated here do not fit the model!! Length of every input is 50.

        Args:
        - day_matrix_list: a list of 2-d np matrix, dtype=<U32;
        - train_day_mean_list; a list of 1-d array;
        - train_day_stddev_list: a list of 1-d array;

        Returns:
        - sample_inputs_list: a list of input samples (2-d np matrix);
        - sample_labels_list: a list of labels, which is a float numebr;
        - train_means_list: a list of float;
        - train_stddevs_list: a list of float;
        - base_index: int, the first column of inputs is the base_index-th column in data_matrix;
        '''
        
        sample_inputs_list = []
        sample_labels_list = []
        train_means_list = []
        train_stddevs_list = []

        raw_n_inputs = 10 # not 10, see comment of this function

        base_index = self.index['MidPrice']

        for ind_day in range (len(day_matrix_list)):
            day_matrix = day_matrix_list[ind_day]
            day_mean = train_day_mean_list[ind_day][self.index['MidPrice']-base_index]
            day_stddev = train_day_stddev_list[ind_day][self.index['MidPrice']-base_index]
            num_samples = day_matrix.shape[0] - self._num_inputs - self._num_labels + 1 # yes it is
            
            for i in range (num_samples):
                new_input = day_matrix[i:(i+raw_n_inputs), base_index:].astype (np.float32)
                new_label = np.mean (day_matrix[(i+self._num_inputs): (i+self._num_inputs+self._num_labels),\
                                     self.index['MidPrice']].astype(np.float32))

                sample_inputs_list.append (new_input)
                sample_labels_list.append (new_label)
                train_means_list.append (day_mean)
                train_stddevs_list.append (day_stddev)

            if ind_day % 10 == 0:
                print ("[", ind_day, "/", len(day_matrix_list), "]")

            # TODO: may need drop the first sample

        print ("Total number of samples", len(sample_inputs_list))
        print ("Sample shape", sample_inputs_list[5].shape)
        return sample_inputs_list, sample_labels_list, train_means_list, train_stddevs_list, base_index




    def __sample_selection (self, sample_inputs_list, sample_labels_list, train_means_list, train_stddevs_list, base_index):
        '''
        Select valid samples. Define valid:
        - Samples with time stamp interval all equal to 3

        Raw samples generated from self.__generate_raw_samples will be cut here
        into inputs of 10 steps, which fits the model.

        Args:
        - sample_inputs_list;
        - sample_labels_list;
        - train_means_list;
        - train_stddevs_list;
        - base_index;

        Returns:
        - new_sample_inputs_list;
        - new_sample_labels_list;
        - new_train_means_list;
        - new_train_stddevs_list;
        - base_index;
        '''
        new_sample_inputs_list = []
        new_sample_labels_list = []
        new_train_means_list = []
        new_train_stddevs_list = []

        print ("Total number of samples before selection", len (sample_inputs_list))

        for i in range (len (sample_inputs_list)):
            if self.__validate_input (sample_inputs_list[i], base_index):
                new_sample_inputs_list.append (sample_inputs_list[i][:self._num_inputs, :])
                new_sample_labels_list.append (sample_labels_list[i])
                new_train_means_list.append (train_means_list[i])
                new_train_stddevs_list.append (train_stddevs_list[i])
        
        print ("Total number of samples after selection", len (new_sample_inputs_list))

        return new_sample_inputs_list, new_sample_labels_list, new_train_means_list, new_train_stddevs_list, base_index



    def __sample_normalization (self, sample_inputs_list, sample_labels_list, base_index, global_normalization=False):
        '''
        Normalize the samples, formula:
            value = (value - mean) / stddev
        Update:
            global features should not be normalized!

        Args:
        - sample_inputs_list: a list of input samples (2-d np matrix);
        - sample_labels_list: a list of labels, which is a float numebr;
        - base_index: int, the first column of inputs is the base_index-th column in data_matrix;
        - global_normalization: bool, True if global normalization is used;

        Returns:
        - sample_inputs_list: a list of normalized input samples (2-d np matrix);
        - sample_lables_list: a list of labels (float); label is
            (original label (mean mid price)- mean mid price in inputs) / stddev mid price in inputs;
        - mean_list: a list of float, mean value of mid price;
        - stddev_list: a list of float, stddev value of mid price;
        '''
        
        assert len (sample_inputs_list) == len (sample_labels_list)

        mean_list = []
        stddev_list = []

        global_feature_indexes = [
            self.index['Volume'],
            self.index['BidVolume1'],
            self.index['AskVolume1']
        ]

        for i in range (len (sample_inputs_list)):
            sample_inputs = sample_inputs_list[i]
            sample_labels = sample_labels_list[i]

            mean = np.mean (sample_inputs, axis=0)
            stddev = np.std (sample_inputs, axis=0)
            stddev = np.maximum (stddev, 1e-6) # to prevent zero division problems

            mean_list.append (mean[self.index['MidPrice'] - base_index])
            stddev_list.append (stddev[self.index['MidPrice'] - base_index])

            if global_normalization:
                # remain the value of global features, because 
                # it is the absolute values of global features that matter
                for ind in global_feature_indexes:
                    mean[ind-base_index] = 0.0
                    stddev[ind-base_index] = 1.0

            # normalize inputs, calculate new label
            sample_inputs = (sample_inputs - mean) / stddev
            sample_labels = sample_labels - mean_list[i]

            sample_inputs_list[i] = sample_inputs[:, (self.index['LastPrice'] - base_index):]
            sample_labels_list[i] = sample_labels

        return sample_inputs_list, sample_labels_list, mean_list, stddev_list




    def __sample_division (self, sample_inputs_list, sample_labels_list, sample_means_list, sample_stddevs_list, train_fraction=0.9):
        '''
        Divide samples into train and dev set

        Args:
        - sample_inputs_list: a list of input samples (2-d np array)
        - sample_labels_list: a list of labels;
        - sample_means_list;
        - sample_stddevs_list;
        - train_fraction: float, the fraction of train set in all samples;

        Returns:
        - train_inputs_list;
        - train_labels_list;
        - train_means_list;
        - train_stddevs_list;
        - dev_inputs_list;
        - dev_labels_list;
        - dev_means_list;
        - dev_stddevs_list;
        '''

        if train_fraction < 1.0:
            train_bound = int (np.ceil(len(sample_inputs_list)*train_fraction))

            train_inputs_list = sample_inputs_list[:train_bound]
            train_labels_list = sample_labels_list[:train_bound]
            train_means_list = sample_means_list[:train_bound]
            train_stddevs_list = sample_stddevs_list[:train_bound]
            dev_inputs_list = sample_inputs_list[train_bound:]
            dev_labels_list = sample_labels_list[train_bound:]
            dev_means_list = sample_means_list[train_bound:]
            dev_stddevs_list = sample_stddevs_list[train_bound:]

            return train_inputs_list, train_labels_list, train_means_list, train_stddevs_list,\
                dev_inputs_list, dev_labels_list, dev_means_list, dev_stddevs_list
        else:
            return sample_inputs_list, sample_labels_list, sample_means_list, sample_stddevs_list,\
                [], [], [], []




    def __store_inputs_and_labels (self, sample_inputs_list, sample_labels_list, mean_list, stddev_list):
        '''
        Store sample inputs and labels as np arrays

        Args:
        - sample_inputs_list: a list of inputs;
        - sample_lables_list: a list of labels;

        Returns:
        inputs_file_path: full path of inputs file;
        labels_file_path: full path of labels file;
        mean_file_path: full path of mean value file;
        stddev_file_path: full path of stddev value file;
        '''

        sample_inputs_list = np.asarray (sample_inputs_list, dtype=np.float32)
        sample_labels_list = np.asarray (sample_labels_list, dtype=np.float32).reshape([-1, 1])
        mean_list = np.asarray (mean_list, dtype=np.float32)
        stddev_list = np.asarray (stddev_list, dtype=np.float32)

        print ("sample inputs shape", sample_inputs_list.shape)
        print ("sample labels shape", sample_labels_list.shape)
        print ("mean list shape", mean_list.shape)
        print ("stddev list shape", stddev_list.shape)
        self.num_features = sample_inputs_list.shape[2]

        inputs_file_path = os.path.join (self._data_dir, TRAIN_INPUTS_FILENAME)
        labels_file_path = os.path.join (self._data_dir, TRAIN_LABELS_FILENAME)
        mean_file_path = os.path.join (self._data_dir, TRAIN_MEANS_FILENAME)
        stddev_file_path = os.path.join (self._data_dir, TRAIN_STDDEVS_FILENAME)
        
        np.save (inputs_file_path, sample_inputs_list)
        np.save (labels_file_path, sample_labels_list)
        np.save (mean_file_path, mean_list)
        np.save (stddev_file_path, stddev_list)

        return inputs_file_path,\
               labels_file_path,\
               mean_file_path,\
               stddev_file_path        




    def __load_inputs_and_labels (self, inputs_file_path, labels_file_path, mean_file_path, stddev_file_path):
        '''
        Load sample inputs and labels

        Args:
        - inputs_file_path: string;
        - labels_file_path: string;
        - mean_file_path: string;
        - stddev_file_path: string;

        Returns:
        - inputs: 3-d np array;
        - labels: 2-d np array;
        - means: 1-d np array;
        - stddevs: 1-d np array;
        '''

        return np.load (inputs_file_path),\
               np.load (labels_file_path),\
               np.load (mean_file_path),\
               np.load (stddev_file_path)




    def next_batch_with_mean_and_stddev (self):
        '''
        next_batch() interface for normalization dedicated for every input

        Args:
        None

        Returns:
        - inputs: a 3-d np array, batch_size x self._num_inputs x self._num_features;
        - labels: a 2-d np array, batch_size x 1;
        - means: a 1-d np array, batch_size
        - stddevs: a 1-d np array, batch_size
        '''

        assert self.batch_ind + self.batch_size <= self.train_inputs.shape[0]

        train_batch_inputs = self.train_inputs [self.batch_ind: self.batch_ind + self.batch_size, :, :]
        train_batch_labels = self.train_labels [self.batch_ind: self.batch_ind + self.batch_size, :]
        train_batch_means = self.train_means [self.batch_ind: self.batch_ind + self.batch_size]
        train_batch_stddevs = self.train_stddevs [self.batch_ind: self.batch_ind + self.batch_size]

        self.batch_ind += self.batch_size

        return train_batch_inputs,\
               train_batch_labels,\
               train_batch_means,\
               train_batch_stddevs



    def __parse_test_data (self, data_matrix):
        '''
        Parse test data into samples

        Args:
        - data_matrix: 2-d np matrix, dtype=<U32

        Returns:
        - test_inputs_list: a list of test inputs, which is a 2-d np matrix;
        - base_index: int, see __sample_normalization ()
        '''

        test_inputs_list = []
        num_inputs = data_matrix.shape[0] // self._num_inputs

        assert type (num_inputs) == type (1) # at least I think so
        assert num_inputs * self._num_inputs == data_matrix.shape[0]

        base_index = self.index['MidPrice']
        for i in range (num_inputs):
            test_inputs_list.append (data_matrix[i*self._num_inputs:(i+1)*self._num_inputs,\
                                                 base_index:].astype (np.float32))

        print ('Total number of test inputs', len (test_inputs_list))

        return test_inputs_list, base_index



    def __store_test_inputs (self, test_inputs_list, mean_list, stddev_list):
        '''
        Save test inputs data as np arrays

        Args:
        - test_inputs_list: a list of test inputs (2-d np array);
        - mean_list: 1-d np array;
        - stddev_list: 1-d np array;

        Returns:
        - test_inputs_path;
        - test_means_path;
        - test_stddevs_path;
        '''

        test_inputs_path = os.path.join (self._data_dir, TEST_INPUTS_FILENAME)
        test_means_path = os.path.join (self._data_dir, TEST_MEANS_FILENAME)
        test_stddevs_path = os.path.join (self._data_dir, TEST_STDDEVS_FILENAME)
        
        np.save (test_inputs_path, np.asarray (test_inputs_list))
        np.save (test_means_path, mean_list)
        np.save (test_stddevs_path, stddev_list)

        return test_inputs_path, test_means_path, test_stddevs_path


    def __load_test_inputs (self, test_inputs_path, test_means_path, test_stddevs_path):
        '''
        Load test inputs, means and stddevs

        Args:
        - test_inputs_path;
        - test_means_path;
        - test_stddevs_path;

        Returns:
        - test_inputs_list: 3-d np array;
        - test_means_list: 1-d np array;
        - test_stddevs_list: 1-d np array;
        '''

        return np.load (test_inputs_path), np.load (test_means_path), np.load (test_stddevs_path)



    def __get_time_stamp (self, date, acc_time):
        '''
        Get the time stamp from date and time

        Args:
        - date: string, form: %Y/%m/%d;
        - acc_time: string, form: %H:%M:%S;
        (for info of %Y, %m, %M, etc., see doc for time.strptime)

        Returns:
        - timestamp: int
        '''

        form = r'%Y-%m-%d %H:%M:%S'
        #print ('date', date)
        #print ('time', acc_time)
        time_array = time.strptime (date+' '+acc_time, form)
        time_stamp = int (time.mktime (time_array))
        #print ('timestamp', str(time_stamp))

        return time_stamp



    def __validate_input (self, input_matrix, base_index):
        '''
        Validate a input data

        Args:
        - input_matrix: 2-d np array;
        - base_index: int;

        Returns:
        - boolean value
        '''
        
        time_stamp_index = self.index['TimeStamp'] - base_index
        for i in range (1, input_matrix.shape[0]):
            if (input_matrix[i, time_stamp_index] - input_matrix[i-1, time_stamp_index]) > 3.2:
                return False

        return True




    def reset_batch (self):
        '''
        Reset self.batch_ind for a new epoch
        '''

        self.batch_ind = 0



    def dev_set (self):    
        '''
        Get the padded dev inputs and labels

        Returns:
            dev_inputs: a list of inputs (lists);
            dev_lables: a list of labels
        '''

        return self.dev_inputs, self.dev_labels, self.dev_means, self.dev_stddevs



    def test_set (self):
        '''
        Get the test inputs, means and stddevs

        Returns:
            test_inputs: 3-d np array;
            test_means: 1-d np array;
            test_stddevs: 1-d np array;
        '''

        if self.test_inputs is None:
            self.test_inputs, self.test_means, self.test_stddevs = \
                self.__load_test_inputs (os.path.join (self._data_dir, TEST_INPUTS_FILENAME),\
                                         os.path.join (self._data_dir, TEST_MEANS_FILENAME),\
                                         os.path.join (self._data_dir, TEST_STDDEVS_FILENAME))

        return self.test_inputs, self.test_means, self.test_stddevs



    def get_data_matrix (self):
        '''
        Get the train and test data matrix

        Args:
        None

        Returns:
        - train_data_matrix: 2-d np array;
        - test_data_matrix: 2-d np array;
        '''

        train_data_matrix = self.__read_data_matrix (os.path.join (self._data_dir, TRAIN_DATA_FILENAME))
        test_data_matrix = self.__read_data_matrix (os.path.join (self._data_dir, TEST_DATA_FILENAME))
        train_data_matrix = self.__clean_data_matrix (train_data_matrix)
        test_data_matrix = test_data_matrix[142*self._num_inputs:, :]

        return train_data_matrix, test_data_matrix
        


    def divide_by_day (self, data_matrix):
        return self.__divide_by_day (data_matrix)





if __name__ == "__main__":
    BASE_DIR = os.path.dirname (os.path.abspath(sys.argv[0]))
    #INPUT_FILENAME = 'train1.csv'
    PROJECT_DIR = os.path.dirname (BASE_DIR)
    DATA_DIR = os.path.join (PROJECT_DIR, 'data')

    order_book = OrderBook (2, DATA_DIR, data_regenerate_flag=True)
    
    print (order_book.num_batches)
    for i in range (10):
        inputs, labels, mean, stddev = order_book.next_batch_with_mean_and_stddev ()
        print (inputs)
        print (labels)
        print (mean)
        print (stddev)
        input ()

    #test_inputs, _ = order_book.test_set()
    #print (test_inputs.shape)

