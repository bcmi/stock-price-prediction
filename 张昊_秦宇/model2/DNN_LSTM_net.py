import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from os.path import exists
from db_init2 import initDatabase
from db_init2 import init_test_data
import torch.utils.data as Data

class OurDNN(nn.Module):
    def __init__(self, seq_len):
        super(OurDNN, self).__init__()
        self.seq_len = seq_len
        self.fc1 = nn.Linear(seq_len, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, inputs): 
        x = inputs
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class OurLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, seq_len):
        super(OurLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first = True)
        self.linear1 = nn.Linear(hidden_dim , output_dim)
        self.linear2 = nn.Linear(seq_len * output_dim, 1)

    def forward(self, inputs): # input:[batch_size, seq_len, input_size]
        x = inputs
        inSize = inputs.size()
        #hid = self.init_hid(inSize[0])
        x, hid = self.lstm(x, None)
        #x = x[:,-1,:]
        #x = F.relu(x)
        #x = x.view(inSize[0], -1)
        x = F.relu(self.linear1(x))
        #x = self.linear1(x)
        x = x.view(inSize[0], -1)
        x = self.linear2(x)
        return x

class combine_net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, seq_len):
        super(combine_net, self).__init__()
        self.lstm1 = OurLSTM(input_dim, hidden_dim, output_dim, num_layers, seq_len)
        self.lstm2 = OurLSTM(input_dim, hidden_dim * 2, output_dim * 2, num_layers * 2, seq_len)
        self.dnn = OurDNN(seq_len)
        self.fc = nn.Linear(2, 1)
        self.fc2= nn.Linear(2, 1)
    def forward(self, inputs):
        x = inputs
        y1 = self.lstm1(x)
        y3 = self.lstm2(x)
        x = x[:,:,0]
        y2 = self.dnn(x)
        #y4 = x[:,9].view(-1, 1)
        y = torch.cat((y1, y2), 1)
        z = self.fc(y)
        z = torch.cat((z, y3), 1)
        output = self.fc2(z)
        return output

def adjust_learning_rate(learning_rate, optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // 4))
    for group in optimizer.param_groups:
        group['lr'] = lr

def main():
    inDim = 7
    hiDim = 64
    outDim = 1
    numLayer = 2
    seq_len = 10
    predict_len = 20
    batch_size = 32
    learning_rate = 1e-4
    device = torch.device("cuda:0")
    epoch_num = 2
    iter_num = 10000
    cross_val = 1

    # Prepare model
    c_net = combine_net(inDim, hiDim, outDim, numLayer, seq_len)
    c_net = c_net.to(device)
    optimizer = optim.Adam(c_net.parameters(), lr = learning_rate)
    crit = nn.MSELoss()

    # Prepare dataset
    inputs_set, targets_set = initDatabase(batch_size, inDim, seq_len, predict_len)
    train_dataset = Data.TensorDataset(inputs_set,targets_set)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epoch_num):
        LOSS = 0
        b_num = 0
        adjust_learning_rate(learning_rate, optimizer, epoch)
        for i, (inputs, targets) in enumerate(train_loader):
            b_num += 1
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = c_net(inputs)
            optimizer.zero_grad()
            loss = crit(outputs.view(-1), targets)
            loss.backward()
            LOSS += loss.data.cpu()
            #acc += torch.sqrt(loss)
            optimizer.step()
            if i % 100 == 0:
                print("train epoch %d, batch %d, loss: %.10f" % \
                (epoch, i, LOSS / b_num))
        LOSS = 0
        b_num = 0
    test_set, means, stds = init_test_data()
    test_set = test_set.to(device)
    # Test
    with torch.no_grad():
        output = c_net(test_set)
        
        output = output.detach().cpu().view(-1).numpy()
        means = np.array(means)
        stds = np.array(stds)
    
        result = output * stds + means
        np.savetxt('combine_result.csv', result[142:])
        #np.savetxt('bm.csv', test_set[142:,9,0] * stds[0] + means[0], fmt = '%.6f')
    #torch.save(c_net, 'c_net.pkl')



if __name__ == "__main__":
    main()