#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
from gensim.models import word2vec


class DiaNet(nn.Module):
    def __init__(self, input_hidden_size, Rnn_hidden_layer, Rnn_hidden_size,
                 Rnn_type, input_size, use_attention, use_RNN, task_type):
        super(DiaNet, self).__init__()
        self.input_hidden_size = input_hidden_size
        self.Rnn_hidden_layer = Rnn_hidden_layer
        self.Rnn_hidden_size = Rnn_hidden_size
        self.Rnn_type = Rnn_type
        self.input_size = input_size
        self.use_attention = use_attention
        self.use_RNN = use_RNN
        self.task_type = task_type
        self.time_interval = 'am'

        self.input_attention_layer = nn.Sequential(nn.Linear(2, self.input_hidden_size),
                                                   nn.Dropout(0),
                                                   nn.Tanh())
        self.input_layer = nn.Sequential(nn.Linear(self.input_size, self.input_hidden_size),
                                         nn.Dropout(0),
                                         nn.Sigmoid())#, nn.LayerNorm(self.input_hidden_size))
        self.NN_input_layer = nn.Sequential(nn.Linear(self.input_size * 10, self.input_hidden_size),
                                            nn.Dropout(0),
                                            nn.ReLU())
########################################################################################################################
        if self.Rnn_type == 'LSTM':
            self.encoder_Rnn = nn.LSTM(self.input_hidden_size, self.Rnn_hidden_size, self.Rnn_hidden_layer, bidirectional=True)
            self.Rnn = nn.LSTM(self.input_hidden_size, self.Rnn_hidden_size, 1)
        elif self.Rnn_type == 'GRU':
            self.encoder_Rnn = nn.GRU(self.input_hidden_size, self.Rnn_hidden_size, self.Rnn_hidden_layer, bidirectional=True)
            self.Rnn = nn.GRU(self.input_hidden_size, self.Rnn_hidden_size, 1)
        else:
            self.encoder_Rnn = nn.RNN(self.input_hidden_size, self.Rnn_hidden_size, self.Rnn_hidden_layer, bidirectional=True)
            self.Rnn = nn.RNN(self.input_hidden_size, self.Rnn_hidden_size, 1)
########################################################################################################################
        if self.Rnn_type == 'LSTM':
            self.AmRnn = nn.LSTM(self.input_hidden_size, self.Rnn_hidden_size, self.Rnn_hidden_layer)
            self.PmRnn = nn.LSTM(self.input_hidden_size, self.Rnn_hidden_size, self.Rnn_hidden_layer)
        elif self.Rnn_type == 'GRU':
            self.PmRnn = nn.GRU(self.input_hidden_size, self.Rnn_hidden_size, self.Rnn_hidden_layer)
            self.AmRnn = nn.GRU(self.input_hidden_size, self.Rnn_hidden_size, self.Rnn_hidden_layer)
        else:
            self.PmRnn = nn.RNN(self.input_hidden_size, self.Rnn_hidden_size, self.Rnn_hidden_layer)
            self.AmRnn = nn.RNN(self.input_hidden_size, self.Rnn_hidden_size, self.Rnn_hidden_layer)
########################################################################################################################
        #self.DRnn = nn.LSTMCell(2, self.Rnn_hidden_size)
        self.DRnn = nn.LSTMCell(self.input_hidden_size, self.Rnn_hidden_size)


        if self.task_type == 'C':
            self.output_layer = nn.Sequential(nn.Linear(self.Rnn_hidden_size, self.Rnn_hidden_size),
                                              nn.Dropout(0),
                                              nn.ReLU(),
                                              nn.Linear(self.Rnn_hidden_size, 2))

            self.NN_output_layer = nn.Sequential(nn.Linear(self.input_hidden_size, self.Rnn_hidden_size),
                                                 nn.Dropout(0),
                                                 nn.ReLU(),
                                                 nn.Linear(self.Rnn_hidden_size, 2))
        else:
            self.output_layer = nn.Sequential(nn.Linear(self.Rnn_hidden_size, self.Rnn_hidden_size),
                                              nn.Dropout(0),
                                              nn.Sigmoid(),
                                              nn.Linear(self.Rnn_hidden_size, 1))

            self.NN_output_layer = nn.Sequential(nn.Linear(self.input_hidden_size, self.Rnn_hidden_size),
                                                 nn.Dropout(0),
                                                 nn.Tanh(),
                                                 nn.Linear(self.Rnn_hidden_size, 1))

        self.am_layer = nn.Sequential(nn.Linear(self.input_size * 10, self.input_hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.Rnn_hidden_size, self.Rnn_hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.Rnn_hidden_size, 2))

        self.pm_layer = nn.Sequential(nn.Linear(self.input_size * 10, self.input_hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.Rnn_hidden_size, self.Rnn_hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.Rnn_hidden_size, 2))

        self.am_op_layer = nn.Sequential(nn.Linear(self.Rnn_hidden_size, self.Rnn_hidden_size),
                                         nn.Sigmoid(),
                                         nn.Linear(self.Rnn_hidden_size, 1))

        self.pm_op_layer = nn.Sequential(nn.Linear(self.Rnn_hidden_size, self.Rnn_hidden_size),
                                         nn.Sigmoid(),
                                         nn.Linear(self.Rnn_hidden_size, 1))

        self.attention_layer = nn.Sequential(nn.Linear(self.Rnn_hidden_size, 1),
                                             nn.Tanh())

        self.attention_output_layer = nn.Sequential(nn.Linear(self.input_size, self.Rnn_hidden_size), nn.Tanh())

    def hidden_init(self, batch_size):
        self.D_hidden = Variable(torch.zeros(batch_size, self.Rnn_hidden_size))
        self.D_memory = Variable(torch.zeros(batch_size, self.Rnn_hidden_size))

    def set_time(self, time_interval):
        self.time_interval = time_interval

    def forward(self, input):

        if self.use_attention:
            attention_encoder = []
            for i in range(10):
                #print(input, input[i].shape, input[i][:,1:3].shape)
                attention_encoder.append(self.input_attention_layer(input[i][:,1:3]))
                #attention_encoder.append(input[i][:, 1:3])

            attention_encoder = Variable(torch.stack(attention_encoder))

            hidden_state = Variable(torch.zeros((attention_encoder.shape[1], self.Rnn_hidden_size)))
            memory_cell = Variable(torch.zeros((attention_encoder.shape[1], self.Rnn_hidden_size)))

            timestep_h = []
            timestep_c = []

            for i in range(10):
                hidden_state, memory_cell = self.DRnn(attention_encoder[i], (hidden_state, memory_cell))
                timestep_h.append(hidden_state)
                timestep_c.append(memory_cell)

            #rnn_out, hidden = self.encoder_Rnn(attention_encoder, (hidden_state, memory_cell))


            #hidden_state = Variable(torch.zeros((2 * self.Rnn_hidden_layer, attention_encoder.shape[1], self.Rnn_hidden_size)))
            #memory_cell = Variable(torch.zeros((2 * self.Rnn_hidden_layer, attention_encoder.shape[1], self.Rnn_hidden_size)))

            #rnn_out, hidden = self.encoder_Rnn(attention_encoder, (hidden_state, memory_cell))

            attention_h = []
            attention_c = []
            for i in range(10):
                attention_h.append(self.attention_layer(timestep_h[i]))
                attention_c.append(self.attention_layer(timestep_c[i]))

            attention_h = Variable(torch.cat(attention_h, dim=1))
            attention_c = Variable(torch.cat(attention_c, dim=1))

            attention_h = attention_h.softmax(dim=1)
            attention_c = attention_c.softmax(dim=1)

            new_h = torch.zeros(input[0].shape)
            new_c = torch.zeros(input[0].shape)

            for i in range(10):
                new_h += attention_h[:, i].reshape((-1, 1)) * input[i]
                new_c += attention_c[:, i].reshape((-1, 1)) * input[i]

            new_hidden = self.attention_output_layer(new_h)
            new_hidden = Variable(torch.stack([new_hidden]))

            new_memory = self.attention_output_layer(new_c)
            new_memory = Variable(torch.stack([new_memory]))

            encoder = []
            for i in range(10):
                encoder.append(self.input_layer(input[i]))

            encoder = Variable(torch.stack(encoder))

            output, hidden = self.Rnn(encoder, (new_hidden, new_memory))

            output = self.output_layer(output[-1])

            return output

        elif self.use_RNN:
            encoder = []
            for i in range(10):
                encoder.append(self.input_layer(input[i]))

            encoder = Variable(torch.stack(encoder))

            hidden_state = Variable(torch.zeros((self.Rnn_hidden_layer, encoder.shape[1], self.Rnn_hidden_size)))
            memory_cell = Variable(torch.zeros((self.Rnn_hidden_layer, encoder.shape[1], self.Rnn_hidden_size)))

            if self.time_interval == 'am':
                rnn_out, hidden = self.PmRnn(encoder, (hidden_state, memory_cell))
                # print(rnn_out[-1][-1])
            else:
                rnn_out, hidden = self.PmRnn(encoder, (hidden_state, memory_cell))

            if self.time_interval == 'am':
                return self.am_op_layer(rnn_out[-1])
            else:
                return self.pm_op_layer(rnn_out[-1])
            # output = self.output_layer(hidden[0][-1])
            #output = self.output_layer(rnn_out[-1])
            return output
        else:
            encoder = []
            for i in range(10):
                encoder.append(input[i])

            encoder = Variable(torch.cat(encoder, dim = 1))
            output = self.NN_input_layer(encoder)
            # output, memory = self.DRnn(output, (self.D_hidden, self.D_memory))
            # self.D_hidden = output
            # self.D_memory = memory
            output = self.NN_output_layer(output)
            return output

class ClassiNet(nn.Module):
    def __init__(self, input_hidden_size, Rnn_hidden_size, Rnn_hidden_layer):
        super(ClassiNet, self).__init__()
        self.input_hidden_size = input_hidden_size
        self.Rnn_hidden_size = Rnn_hidden_size
        self.Rnn_hidden_layer = Rnn_hidden_layer
        self.input_layer = nn.Sequential(nn.Linear(100, self.input_hidden_size), nn.Dropout(0.0), nn.ReLU())
        self.encoder = nn.LSTM(self.input_hidden_size, self.Rnn_hidden_size, num_layers=self.Rnn_hidden_layer, bidirectional=True)
        '''
        self.output_layer = nn.Sequential(nn.Linear(2 * self.Rnn_hidden_size, 64), nn.ReLU(),
                                          nn.Linear(64, 32), nn.ReLU(),
                                          nn.Linear(32, 16), nn.ReLU(),
                                          nn.Linear(16, 8), nn.ReLU(),
                                          nn.Linear(8, 4), nn.ReLU(),
                                          nn.Linear(4, 2), nn.ReLU())
        '''
        self.output_layer = nn.Sequential(nn.Linear(2 * self.Rnn_hidden_size, 2))

    def forward(self, input):
        res = []
        #print(input)
        for step in input:
            #res.append(step)
            res.append(self.input_layer(step))
        res = torch.stack(res)
        #res = Variable(torch.cat(res, dim=1))
        #res = self.input_layer(res)

        hidden = torch.zeros((self.Rnn_hidden_layer * 2, (input[0].shape)[0], self.Rnn_hidden_size))
        memory = torch.zeros((self.Rnn_hidden_layer * 2, (input[0].shape)[0], self.Rnn_hidden_size))

        res, _ = self.encoder(res, (hidden, memory))

        res = self.output_layer(res[-1])
        #res = self.output_layer(res)

        return res


if __name__ == '__main__':
    sentences = word2vec.Text8Corpus('gramma.txt')
    model = word2vec.Word2Vec(sentences, size=100, min_count=2, window=10, iter=15)
    model.save('word2vec.bin')
    #model = gensim.models.Word2Vec.load('word2vec.bin')
    #print(torch.Tensor(model['b0d0d1b']))





