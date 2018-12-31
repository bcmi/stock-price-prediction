from readfile import *
import torch
import torch.nn as nn

INPUT_SIZE = 6
HIDDEN_SIZE = 128
NUM_LAYERS = 3
OUTPUT_SIZE = 1
LEARNING_RATE = 0.001
EPOCH = 5


class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()
        self.input_size = INPUT_SIZE
        self.hidden_dim = HIDDEN_SIZE
        self.lstm = nn.LSTM(input_size=INPUT_SIZE,
                            hidden_size=HIDDEN_SIZE,
                            num_layers=NUM_LAYERS,
                            batch_first=True,
                            dropout=0.5
                            )
        self.hidden2out = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x, None)
        out = self.hidden2out(out[:, -1, :])
        return out


lstm = LSTM()
optimizer = torch.optim.Adam(lstm.parameters(), lr=LEARNING_RATE)
loss_func = nn.MSELoss()
loss_func = loss_func.cuda()
lstm = lstm.cuda()
lstm.train()
train_dataset, test_dataset, base_dataset = create_dsataset("data/train_data.csv", "data/test_data.csv")

for epoch in range(EPOCH):
    for index, (x, y) in enumerate(train_dataset):
        x, y = x.cuda(), y.cuda()
        output = lstm(x)
        loss = loss_func(output[:, 0], y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if index % 50 == 0:
            print('Epoch: ', epoch, '| train loss: ', loss.data.cpu().numpy())
torch.save(lstm, "lstm.pkl")

lstm.eval()
pred = []
for test in list(test_dataset):
    test = test[0].cuda()
    output = lstm(test)
    res = output[:, 0][0]
    pred.append(res.cpu())
    print(res.cpu())

ans = np.array(base_dataset) + np.array(pred)
with open("sample1.csv", 'w') as fout:
    fieldnames = ['caseid', 'midprice']
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(ans)):
        if i < 142:
            continue
        writer.writerow({'caseid': str(i + 1), 'midprice': float(ans[i])})
