import torch
import torch.nn as nn
from torch.nn import init


class LSTMCell(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(LSTMCell, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        #self.FC = nn.Linear(in_dim+hid_dim, 4 * hid_dim)

        self.weight_ih = nn.Parameter(
            torch.FloatTensor(in_dim, 3 * hid_dim))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hid_dim, 3 * hid_dim))
        self.bias = nn.Parameter(torch.FloatTensor(3 * hid_dim))

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        init.orthogonal(self.weight_ih.data)
        weight_hh_data = torch.eye(self.hid_dim)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        self.weight_hh.data.set_(weight_hh_data)

        init.constant(self.bias.data, val = 0)
    
    def forward(self, x_t, c_prev, h_prev):

        # comb_feat = torch.cat([h_prev, x_t], 1)
        # comb_gates = self.FC(comb_feat)
        # i_gate = self.sigmoid(comb_gates[:, :self.hid_dim])
        # f_gate = self.sigmoid(comb_gates[:, self.hid_dim:2*self.hid_dim])
        # o_gate = self.sigmoid(comb_gates[:, 2*self.hid_dim:3*self.hid_dim])
        # g_gate = self.tanh(comb_gates[:, 3*self.hid_dim:])

        batch_size = h_prev.size(0)
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))

        m_h = torch.addmm(bias_batch, h_prev, self.weight_hh)
        m_i = torch.mm(x_t, self.weight_ih)

        i_gate, o_gate, g_gate = torch.split(m_h + m_i, split_size_or_sections = self.hid_dim, dim = 1)

        i_gate = self.sigmoid(i_gate)
        g_gate = self.tanh(g_gate)
        o_gate = self.sigmoid(o_gate)

        f_gate = 1 - i_gate

        c_t = f_gate * c_prev + i_gate * g_gate
        h_t = o_gate * self.tanh(c_t)
        return c_t, h_t
    

class LSTM(nn.Module):
    def __init__(self, in_dim, hid_dim, time_steps, device="cuda:0"):
        super(LSTM, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.T = time_steps
        self.device = device
        self.lstm_cell = LSTMCell(in_dim, hid_dim)

        # self.fc1 = nn.Linear(hid_dim, 1)


        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, 1)

        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(True)
        
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        prev_c = torch.zeros(x.size(0), self.hid_dim).float().to(self.device)
        prev_h = torch.zeros(x.size(0), self.hid_dim).float().to(self.device)
        for i in range(self.T):
            prev_c, prev_h = self.lstm_cell(x[:,i,:], prev_c, prev_h)

        internal = self.fc1(prev_h)
        internal = self.relu(internal)
        #internal = self.dropout(internal)
        score = self.fc2(internal)

        #score = self.fc1(prev_h)
        #return self.tanh(score).view(-1)
        return score.view(-1)

if __name__ == '__main__':
    device = "cuda:0"
    data = torch.randn(32, 10, 6).to(device)
    lstm = LSTM(6, 64, 10).to(device)
    output = lstm(data)
    print(output, output.shape)