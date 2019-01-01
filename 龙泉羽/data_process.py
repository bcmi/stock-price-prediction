import numpy as np
import os
import sys

TRAIN_DATA_FILENAME = 'train_data.csv'
TEST_DATA_FILENAME = 'test_data.csv'


class Data:
    
    def __init__ (self, batch_size):
        self.batch_size = batch_size
        
        self.batch_ind = 0
        
        self.train_num_half_day = 0
        self.test_num_half_day = 0
        
        '''
        train_inputs_list: list, an element is a half day train data list, element of which is an np array with shape (10,6)
        168 * XXX * 10 * 6
        train_labels_list: list, an element is a half day train data list, element of which is a num
        train_means_list: list, an elenemt is np array with shape (7,)
        train_stddevs_list: list, an element is np array with shape (7,)
        '''
        self.train_inputs_list = None
        self.train_labels_list = None
        self.train_means_list = None
        self.train_stddevs_list = None
        
        self.train_inputs = None
        self.train_labels = None
        self.train_means = None
        self.train_stddevs = None
        
        self.dev_inputs = None
        self.dev_labels = None
        self.dev_means = None
        self.dev_stddevs = None
        
        '''
        18 * XXX * 10 * 6
        '''
        self.test_inputs_list = None
        self.test_means_list = None
        self.test_stddevs_list = None
        
        
        self.__preprocess(TRAIN_DATA_FILENAME, is_train = True)
        self.__preprocess(TEST_DATA_FILENAME, is_train = False)
        #self.__shuffle_half_day_data()
        self.__combine_data()
        
        
        
    def __preprocess (self, inFileName, is_train):
        #read data
        infile = open(inFileName, 'r')
        infile.readline()
        m1 = []
        for line in infile.readlines():
            line = line.strip()
            if line == '':
                continue
            line = line.strip('\n')
            line = line.split(',')[1:]
            line[1] = line[1].strip()
            #print line[1]
            for i in range(2, len(line)):
                line[i] = float(line[i])
            m1.append(line)
        #m1 date time mid ...
        
        if is_train:
            #split train data according to the time
            m2 = []
            l = r = 0
            t1 = '09:30'
            t2 = '11:29'
            t3 = '13:00'
            t4 = '14:59'
            for i in range(1, len(m1)-1):
                if m1[i][1].startswith(t1) and not m1[i-1][1].startswith(t1):
                    l = i
                if m1[i][1].startswith(t2) and not m1[i+1][1].startswith(t2):
                    r = i+1
                    m2.append(m1[l:r])
                    #print m1[i][0],l,r
                if m1[i][1].startswith(t3) and not m1[i-1][1].startswith(t3):
                    l = i
                if m1[i][1].startswith(t4) and not m1[i+1][1].startswith(t4):
                    r = i+1
                    m2.append(m1[l:r])
                    #print m1[i][0],l,r
                    
            #delete some redundant data
            for i in range(len(m2)):
                del m2[i][0]
                s = len(m2[i]) - len(m2[i])%10
                m2[i] = m2[i][:s]
                
            
            #m2: date time midPrice ...
            #m2: split morning and afternoon, delete redundant data
            
            #half day shuffle
            
            #nomalization
            train_mean = []
            train_stddev = []
            for i in range(len(m2)):
                half_day_matrix = np.asarray(m2[i])
                half_day_matrix = half_day_matrix[..., 2:].astype(float)
                mean = np.mean(half_day_matrix, axis = 0)
                stddev = np.std(half_day_matrix, axis = 0)
                half_day_matrix = (half_day_matrix - mean) / stddev
                m2[i] = half_day_matrix
                train_mean.append(mean)
                train_stddev.append(stddev)
            
            #generate train data
            xMatrix = []
            yMatrix = []
            for mi in m2:
                tmp = []
                for i in range(0, len(mi)-30+1, 1):
                    tmp1 = mi[i:i+10, 1:]
                    #tmp1 = mi[i:i+10, :]
                    tmp.append(tmp1)
                xMatrix.append(tmp)
                tmp = []
                for i in range(10, len(mi)-20+1, 1):
                    tmp2 = mi[i:i+20, 0:1]
                    tmp.append(np.mean(tmp2))
                yMatrix.append(tmp)
                    
            '''
            #shuffle
            state = np.random.get_state()
            np.random.shuffle(xMatrix)
            np.random.set_state(state)
            np.random.shuffle(yMatrix)
            '''
            
            '''
            #nomalization
            for i in range(len(xMatrix)):
                xMatrix[i] = np.asarray(xMatrix[i]).astype(float)
                mean = np.mean(xMatrix[i], axis = 0)
                std = np.std(xMatrix[i], axis = 0) + 1E-20
                xMatrix[i] = (xMatrix[i] - mean) / std
            '''
            
            self.train_inputs_list = xMatrix
            self.train_labels_list = yMatrix
            self.train_means_list = train_mean
            self.train_stddevs_list = train_stddev
            self.train_num_half_day = len(xMatrix)
            
            
        else:
            #test case 143:1000 858
            m1 = m1[1420:]

            m2 = []
            l = r = 0
            t1 = '09'
            t2 = '11'
            t3 = '13'
            t4 = '14'
            for i in range(1, len(m1)):
                if m1[i][1].startswith(t1) and not m1[i-1][1].startswith(t1):
                    l = i
                if m1[i][1].startswith(t2) and not m1[i+1][1].startswith(t2):
                    r = i+1
                    m2.append(m1[l:r])
                    #print m1[i][0],l,r
                if m1[i][1].startswith(t3) and not m1[i-1][1].startswith(t3):
                    l = i
                if i == len(m1)-1:
                    r = i+1
                    m2.append(m1[l:r])
                    #print m1[i][0],l,r
                    continue
                if m1[i][1].startswith(t4) and not m1[i+1][1].startswith(t4):
                    r = i+1
                    m2.append(m1[l:r])
                    #print m1[i][0],l,r
            '''
            for mi in m2:
                print len(mi)
            '''
            
            #nomalization
            test_mean = []
            test_stddev = []
            for i in range(len(m2)):
                half_day_matrix = np.asarray(m2[i])
                half_day_matrix = half_day_matrix[..., 2:].astype(float)
                mean = np.mean(half_day_matrix, axis = 0)
                stddev = np.std(half_day_matrix, axis = 0)
                half_day_matrix = (half_day_matrix - mean) / stddev
                m2[i] = half_day_matrix
                test_mean.append(mean)
                test_stddev.append(stddev)

            m3 = []
            for mi in m2:
                tmp = []
                for i in range(0, len(mi), 10):
                    tmp.append(mi[i:i+10, 1:])
                    #tmp.append(mi[i:i+10, :])
                m3.append(tmp)
                
            self.test_inputs_list = m3
            self.test_stddevs_list = test_stddev
            self.test_means_list = test_mean
            self.test_num_half_day = len(m3)
    
    def __shuffle_half_day_data(self):
        for i in range(self.train_num_half_day):
            state = np.random.get_state()
            np.random.shuffle(self.train_inputs_list[i])
            np.random.set_state(state)
            np.random.shuffle(self.train_labels_list[i])
    
    
    def __combine_data(self):
        train_means = []
        train_stddevs = []
        train_inputs = []
        train_labels = []
        for i in range(self.train_num_half_day):
            half_day_inputs = self.train_inputs_list[i]
            half_day_labels = self.train_labels_list[i]
            num = len(half_day_inputs)
            mean = self.train_means_list[i][0]
            stddev = self.train_stddevs_list[i][0]
            for j in range(num):
                train_inputs.append(self.train_inputs_list[i][j])
                train_labels.append(self.train_labels_list[i][j])
                train_means.append(mean)
                train_stddevs.append(stddev)
        '''
        state = np.random.get_state()
        np.random.shuffle(train_inputs)
        np.random.set_state(state)
        np.random.shuffle(train_labels)
        np.random.set_state(state)
        np.random.shuffle(train_labels)
        np.random.set_state(state)
        np.random.shuffle(train_means)
        np.random.set_state(state)
        np.random.shuffle(train_stddevs)
        '''
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.train_means = train_means
        self.train_stddevs = train_stddevs
        
        self.train_inputs_list = None
        self.train_labels_list = None
        self.train_means_list = None
        self.train_stddevs_list = None
    
    
    def __devide_data(self):
        #9:1
        c = len(self.inputs) /10
        c = c * 9
        self.train_inputs = self.inputs[:c]
        self.train_labels = self.labels[:c]
        self.dev_inputs = self.inputs[c:]
        self.dev_labels = self.labels[c:]
    
    
    def num_batches(self):
        return int(len(self.train_labels) / self.batch_size)
    
    def reset_batch(self):
        self.batch_ind = 0
        
    def next_batch(self):
        
        assert self.batch_ind + self.batch_size <= len(self.train_labels)
        
        train_batch_inputs = self.train_inputs [self.batch_ind: self.batch_ind + self.batch_size]
        train_batch_inputs = np.asarray(train_batch_inputs)
        train_batch_labels = self.train_labels [self.batch_ind: self.batch_ind + self.batch_size]
        train_batch_labels = np.asarray(train_batch_labels).reshape([-1,1])
        train_batch_means = self.train_means [self.batch_ind: self.batch_ind + self.batch_size]
        train_batch_means = np.asarray(train_batch_means).reshape([-1,1])
        train_batch_stddevs = self.train_stddevs [self.batch_ind: self.batch_ind + self.batch_size]
        train_batch_stddevs = np.asarray(train_batch_stddevs).reshape([-1,1])
        
        self.batch_ind += self.batch_size
        
        return train_batch_inputs, train_batch_labels, train_batch_means, train_batch_stddevs
        
        
    def get_test_data(self):
        return self.test_inputs_list, self.test_means_list, self.test_stddevs_list
        
    def get_dev_data(self):
        return np.asarray(self.dev_inputs), np.asarray(self.dev_labels).reshape(-1,1)


if __name__ == "__main__":
    data = Data(32)
    print data.num_batches()
    print len(data.train_inputs), len(data.train_labels), len(data.train_means), len(data.train_stddevs)
    for i in range(data.num_batches()):
        train_batch_inputs, train_batch_labels, train_batch_means, train_batch_stddevs = data.next_batch()
        print train_batch_inputs.shape, train_batch_labels.shape, train_batch_means.shape, train_batch_stddevs.shape
