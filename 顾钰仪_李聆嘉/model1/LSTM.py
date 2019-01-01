import torch
from torch.autograd import Variable


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, dropout = 1):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=6,
            dropout=0,
            bidirectional=False,
        )
        self.lstm2 = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=6,
            dropout=1,
            bidirectional=False,
        )
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size)
        self.bn1 = torch.nn.BatchNorm1d(hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 6)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # print(x.size())
        batch_size = x.size()[2]
        seq_length = x.size()[0]
        print(batch_size,seq_length)

        h0 = Variable(torch.zeros(6, 10, self.hidden_size))
        print(h0.size())
        c0 = Variable(torch.zeros(6, 10, self.hidden_size))

        # print(x.size(),h0.size(),c0.size())
        out1, (ht, ct) = self.lstm(x, (h0, c0))
        outputs, (ht, ct) = self.lstm2(out1, (h0, c0))

        print("outputs:",outputs.size())
        out = outputs[-1]
        print("out",out.size())
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = torch.nn.functional.dropout(out, training=self.training)
        out = self.fc2(out)

        return out