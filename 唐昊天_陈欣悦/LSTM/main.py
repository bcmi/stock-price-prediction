# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from data_proc import HFTDataset, HFTTestDataset
# from LSTM import LSTM
# import csv
#
# def train():
#     lstm.train()
#     for epoch in range(NUM_EPOCHES):
#         loss_sum = 0
#         for idx, (data, gt) in enumerate(dataloader):
#             data, gt = data.to(device), gt.to(device).view(-1)
#             prediction = lstm(data) * 0.1 + 0.5*(data[:, 9, 2] + data[:, 9, 4])
#             loss = 100 * loss_fn(prediction, gt)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             # if idx % 50 == 0:
#             #     print('Epoch %d, Iteration %d: loss = %.4f'%(epoch+1, idx+1, loss.item()))
#
#             loss_sum += loss
#             if idx % 1000 == 0:
#                 print(prediction)
#                 print(gt)
#                 print()
#         print('Epoch %d, loss = %.4f' % (epoch + 1,  loss_sum))
#         if (epoch+1) % 10 == 0:
#             torch.save(lstm.state_dict(), "checkpoints/epoch_%s.pkl"%(epoch+1))
#
# def test(epoch=10):
#     lstm.load_state_dict(torch.load("checkpoints/epoch_%s.pkl"%epoch))
#     lstm.eval()
#     ans = []
#     for idx, data in enumerate(test_dataloader):
#         data = data.to(device)
#         prediction = lstm(data) * 0.1 + 0.5*(data[:, 9, 2] + data[:, 9, 4])
#         for i in range(len(data)):
#             ans.append(prediction[i].item())
#
#     with open('sample.csv','w') as fout:
#         fieldnames = ['caseid','midprice']
#         writer = csv.DictWriter(fout, fieldnames = fieldnames)
#         writer.writeheader()
#         for i in range(len(ans)):
#             writer.writerow({'caseid':str(i+1),'midprice':float(ans[i])})
#     print(ans)
#
#
# def lr_decay(optimizer, epoch, decay_rate, init_lr):
#     #lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
#     lr = init_lr * ((1-decay_rate)**epoch)
#     logger.info((" Learning rate is setted as:", lr))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return optimizer
#
# if __name__ == '__main__':
#     # device = 'cuda:0'
#     device = 'cpu'
#     NUM_EPOCHES = 50
#     dataset = HFTDataset()
#     dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
#     test_dataset = HFTTestDataset()
#     test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#     lstm = LSTM(6, 128, 10, device).to(device)
#     loss_fn = nn.L1Loss()
#     #optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001, betas=(0.9,0.999))
#     sgd_optimizer = optim.SGD(lstm.parameters(), lr = 0.015)
#     #train()
#     test(50)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_proc import HFTDataset, HFTTestDataset
from LSTM import LSTM
import csv

'''originally batch_size = 32, hidden_dim = 128'''

def train():
    lstm.train()

    for epoch in range(NUM_EPOCHES):
        print("========Start Epoch %d========" % (epoch + 1))
        val()

        loss_sum = 0
        #optimizer = get_lr_decay(sgd_optimizer, epoch, lr_decay, lr)

        for idx, (data, gt) in enumerate(dataloader):
            data, gt= data.to(device), gt.to(device).view(-1)
            """
            prediction_residual = 0.5*torch.mean(data[:, :, 2] + data[:, :, 4], dim=1)
            prediction = prediction_lstm + prediction_residual
            """


            prediction = lstm(data) * 0.1 + 0.5 * (data[:, 9, 2] + data[:, 9, 4])
            #prediction = lstm(data)
            #prediction = lstm(data) * 0.1 + data[:, 9, 0]
            #prediction = lstm(data[:, :, 1:]) * 0.1 + data[:, 9, 0]

            #loss = 100 * (loss_fn(prediction_f, gt1) + loss_fn(prediction_c, gt2))/2
            #loss = 100*loss_fn(prediction, gt)
            loss = 100 * loss_fn(prediction, gt)
            optimizer.zero_grad()
            loss.backward()


            if idx % 1000 == 0 and idx > 0:
                print("========Printing Gradients========")
                print("LSTM FC1")
                print(lstm.fc1.weight.grad.mean().abs().mean().item())
                print("LSTM FC2")
                print(lstm.fc2.weight.grad.abs().mean().item())
                print("LSTM CELL FC")
                print(lstm.lstm_cell.weight_hh.grad.abs().mean().item())
                print(lstm.lstm_cell.weight_ih.grad.abs().mean().item())
                #print(lstm.lstm_cell.weight_ih.grad.mean().item())
                print("========End Printing Gradients========")


            optimizer.step()


            loss_sum += loss.item()
            # if idx % 50 == 0:
            #     print('Epoch %d, Iteration %d: loss = %.4f' % (epoch + 1, idx + 1, loss.item()))
            if idx % 1000 == 0:
                print(prediction)
                print(gt)

        print('Epoch %d: loss = %.4f' % (epoch + 1, loss_sum))
        loss_sum = 0
        if (epoch + 1) % 10 == 0:
            torch.save(lstm.state_dict(), "checkpoints/2_epoch_%s.pkl" % (epoch + 1))

        print("========End Epoch %d========" % (epoch + 1))


def val():
    lstm.eval()
    dataset.phase = "val"
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_idx = 0
    total_loss = 0.
    for idx, (data, gt) in enumerate(dataloader):
        data, gt = data.to(device), gt.to(device).view(-1)
        prediction = lstm(data) * 0.1 + 0.5 * (data[:, 9, 2] + data[:, 9, 4])
        #prediction = lstm(data) * 0.1 + data[:, 9, 0]
        #prediction = lstm(data)
        #prediction = lstm(data[:, :, 1:]) * 0.1 + data[:, 9, 0]
        loss = torch.sqrt(torch.mean((prediction - gt) ** 2))
        total_loss += loss
        total_idx += 1

    avg_loss = total_loss / total_idx
    print("========Start Evaluation========")
    print("Evaluation on validation set: %.8f" % avg_loss.item())
    print("========End Evaluation========")
    lstm.train()
    dataset.phase = "train"
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


def test(epoch=10):
    lstm1.load_state_dict(torch.load("checkpoints/epoch_%s.pkl" % epoch))
    lstm1.eval()
    lstm2.load_state_dict(torch.load("checkpoints/2_epoch_%s.pkl" % epoch))
    lstm2.eval()


    ans = []
    for idx, data in enumerate(test_dataloader):
        data = data.to(device)
        prediction = lstm1(data) * 0.05 + lstm2(data) * 0.05 + 0.5 * (data[:, 9, 2] + data[:, 9, 4])
        #prediction = lstm(data) * 0.1 + data[:, 9, 0]
        #prediction = lstm(data)
        #prediction = lstm(data) * 0.1 + prev_midp
        """
        prediction_lstm = lstm(data) * 0.1
        prediction_residual = 0.5*torch.mean(data[:, :, 2] + data[:, :, 4], dim=1)
        prediction = prediction_lstm + prediction_residual
        """

        for i in range(len(data)):
            ans.append(prediction[i].item())

    with open('sample.csv', 'w') as fout:
        fieldnames = ['caseid', 'midprice']
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        #start from case 143
        for i in range(142,len(ans)):
            writer.writerow({'caseid':str(i+1),'midprice':float(ans[i])})
    #print(ans)

def get_lr_decay(optimizer, epoch, decay_rate, init_lr):
    #lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
    lr = init_lr * ((1-decay_rate)**epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

if __name__ == '__main__':
    torch.manual_seed(1)
    batch_size = 32
    hidden_dim = 128
    device = 'cpu'
    NUM_EPOCHES = 40
    feature_num = 6
    dataset = HFTDataset(split = "random")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataset = HFTTestDataset()
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    lstm = LSTM(feature_num, hidden_dim, 10, device).to(device)

    loss_fn = nn.L1Loss()



    #lr = 0.1
    #sgd_optimizer = torch.optim.SGD(lstm.parameters(), lr = lr)
    #lr_decay = 0.05
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001, betas=(0.9, 0.999))

    print("========Print Model========")
    print(lstm)
    print("========End Printing========")
    train()
    test(20)



    # f = open("all/test_data.csv", "r")
    # reader = csv.DictReader(f)
    # fieldnames = ['LastPrice']
    #
    # midPrices = []
    # dates = []
    # times = []
    # lastPrices = []
    # volumes = []
    # bidPrices = []
    # bidVolumes = []
    # askPrices = []
    # askVolumes = []
    # microPrices = []
    # ans = []
    # dic = {'LastPrice': lastPrices}
    # for row in reader:
    #     for field in fieldnames:
    #         dic[field].append(float(row[field]))
    #
    # for i in range(int(len(dic['LastPrice'])/10)-1):
    #     ans.append(0.5 * (dic['LastPrice'][i*10] + dic['LastPrice'][i*10+10]))
    # ans.append(dic['LastPrice'][-1])
    #
    # with open('sample.csv', 'w') as fout:
    #     fieldnames = ['caseid', 'midprice']
    #     writer = csv.DictWriter(fout, fieldnames=fieldnames)
    #     writer.writeheader()
    #     #start from case 143
    #     for i in range(142,len(ans)):
    #         writer.writerow({'caseid':str(i+1),'midprice':float(ans[i])})















#
#
# # import torch
# # import torch.nn as nn
# # from torch.utils.data import DataLoader
# # from data_proc import HFTDataset, HFTTestDataset
# # from LSTM import LSTM
# # import csv
# #
# # def train():
# #     lstm.train()
# #     for epoch in range(NUM_EPOCHES):
# #         loss_sum = 0
# #         for idx, (data, gt) in enumerate(dataloader):
# #             data, gt = data.to(device), gt.to(device).view(-1)
# #             prediction = lstm(data) * 0.1 + 0.5*(data[:, 9, 2] + data[:, 9, 4])
# #             loss = 100 * loss_fn(prediction, gt)
# #             optimizer.zero_grad()
# #             loss.backward()
# #             optimizer.step()
# #
# #             # if idx % 50 == 0:
# #             #     print('Epoch %d, Iteration %d: loss = %.4f'%(epoch+1, idx+1, loss.item()))
# #
# #             loss_sum += loss
# #             if idx % 1000 == 0:
# #                 print(prediction)
# #                 print(gt)
# #                 print()
# #         print('Epoch %d, loss = %.4f' % (epoch + 1,  loss_sum))
# #         if (epoch+1) % 10 == 0:
# #             torch.save(lstm.state_dict(), "checkpoints/epoch_%s.pkl"%(epoch+1))
# #
# # def test(epoch=10):
# #     lstm.load_state_dict(torch.load("checkpoints/epoch_%s.pkl"%epoch))
# #     lstm.eval()
# #     ans = []
# #     for idx, data in enumerate(test_dataloader):
# #         data = data.to(device)
# #         prediction = lstm(data) * 0.1 + 0.5*(data[:, 9, 2] + data[:, 9, 4])
# #         for i in range(len(data)):
# #             ans.append(prediction[i].item())
# #
# #     with open('sample.csv','w') as fout:
# #         fieldnames = ['caseid','midprice']
# #         writer = csv.DictWriter(fout, fieldnames = fieldnames)
# #         writer.writeheader()
# #         for i in range(len(ans)):
# #             writer.writerow({'caseid':str(i+1),'midprice':float(ans[i])})
# #     print(ans)
# #
# #
# # def lr_decay(optimizer, epoch, decay_rate, init_lr):
# #     #lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
# #     lr = init_lr * ((1-decay_rate)**epoch)
# #     logger.info((" Learning rate is setted as:", lr))
# #     for param_group in optimizer.param_groups:
# #         param_group['lr'] = lr
# #     return optimizer
# #
# # if __name__ == '__main__':
# #     # device = 'cuda:0'
# #     device = 'cpu'
# #     NUM_EPOCHES = 50
# #     dataset = HFTDataset()
# #     dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# #     test_dataset = HFTTestDataset()
# #     test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# #     lstm = LSTM(6, 128, 10, device).to(device)
# #     loss_fn = nn.L1Loss()
# #     #optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001, betas=(0.9,0.999))
# #     sgd_optimizer = optim.SGD(lstm.parameters(), lr = 0.015)
# #     #train()
# #     test(50)
#
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from data_proc import HFTDataset, HFTTestDataset
# from LSTM import LSTM
# import csv
#
# '''originally batch_size = 32, hidden_dim = 128'''
#
# def train():
#     lstm.train()
#
#     for epoch in range(NUM_EPOCHES):
#         print("========Start Epoch %d========" % (epoch + 1))
#         val()
#
#         loss_sum = 0
#         #optimizer = get_lr_decay(sgd_optimizer, epoch, lr_decay, lr)
#
#         for idx, (data, gt, mg) in enumerate(dataloader):
#             data, gt, mg= data.to(device), gt.to(device).view(-1), mg.to(device).view(-1)
#             """
#             prediction_residual = 0.5*torch.mean(data[:, :, 2] + data[:, :, 4], dim=1)
#             prediction = prediction_lstm + prediction_residual
#             """
#
#
#             #prediction = lstm(data) * 0.1 + 0.5 * (data[:, 9, 2] + data[:, 9, 4])
#             prediction = lstm(data) * 0.1 + mg
#             #prediction = lstm(data)
#             #prediction = lstm(data) * 0.1 + data[:, 9, 0]
#             #prediction = lstm(data[:, :, 1:]) * 0.1 + data[:, 9, 0]
#
#             #loss = 100 * (loss_fn(prediction_f, gt1) + loss_fn(prediction_c, gt2))/2
#             #loss = 100*loss_fn(prediction, gt)
#             loss = 100 * loss_fn(prediction, gt+mg)
#             optimizer.zero_grad()
#             loss.backward()
#
#
#             if idx % 1000 == 0 and idx > 0:
#                 print("========Printing Gradients========")
#                 print("LSTM FC1")
#                 print(lstm.fc1.weight.grad.mean().abs().mean().item())
#                 print("LSTM FC2")
#                 print(lstm.fc2.weight.grad.abs().mean().item())
#                 print("LSTM CELL FC")
#                 print(lstm.lstm_cell.weight_hh.grad.abs().mean().item())
#                 print(lstm.lstm_cell.weight_ih.grad.abs().mean().item())
#                 #print(lstm.lstm_cell.weight_ih.grad.mean().item())
#                 print("========End Printing Gradients========")
#
#
#             optimizer.step()
#
#
#             loss_sum += loss.item()
#             # if idx % 50 == 0:
#             #     print('Epoch %d, Iteration %d: loss = %.4f' % (epoch + 1, idx + 1, loss.item()))
#             if idx % 1000 == 0:
#                 print(prediction+mg)
#                 print(gt)
#
#         print('Epoch %d: loss = %.4f' % (epoch + 1, loss_sum))
#         loss_sum = 0
#         if (epoch + 1) % 10 == 0:
#             torch.save(lstm.state_dict(), "checkpoints/2_epoch_%s.pkl" % (epoch + 1))
#
#         print("========Ezzznd Epoch %d========" % (epoch + 1))
#
#
# def val():
#     lstm.eval()
#     dataset.phase = "val"
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     total_idx = 0
#     total_loss = 0.
#     for idx, (data, gt, mg) in enumerate(dataloader):
#         data, gt, mg = data.to(device), gt.to(device).view(-1), mg.to(device).view(-1)
#         #prediction = lstm(data) * 0.1 + 0.5 * (data[:, 9, 2] + data[:, 9, 4])
#         prediction = lstm(data) * 0.1 + mg
#         #prediction = lstm(data) * 0.1 + data[:, 9, 0]
#         #prediction = lstm(data)
#         #prediction = lstm(data[:, :, 1:]) * 0.1 + data[:, 9, 0]
#         loss = torch.sqrt(torch.mean((prediction - gt-mg) ** 2))
#         total_loss += loss
#         total_idx += 1
#
#     avg_loss = total_loss / total_idx
#     print("========Start Evaluation========")
#     print("Evaluation on validation set: %.8f" % avg_loss.item())
#     print("========End Evaluation========")
#     lstm.train()
#     dataset.phase = "train"
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
#
# def test(epoch=10):
#     lstm.load_state_dict(torch.load("checkpoints/epoch_%s.pkl" % epoch))
#     lstm.eval()
#
#
#
#     ans = []
#     for idx, (data, mg) in enumerate(test_dataloader):
#         data, mg = data.to(device), mg.to(device)
#         print(mg)
#         #prediction = lstm(data) * 0.1 + 0.5 * (data[:, 9, 2] + data[:, 9, 4])
#         prediction = lstm(data) * 0.1 + mg[0]
#         #prediction = lstm(data) * 0.1 + data[:, 9, 0]
#         #prediction = lstm(data)
#         #prediction = lstm(data) * 0.1 + prev_midp
#         """
#         prediction_lstm = lstm(data) * 0.1
#         prediction_residual = 0.5*torch.mean(data[:, :, 2] + data[:, :, 4], dim=1)
#         prediction = prediction_lstm + prediction_residual
#         """
#
#         for i in range(len(data)):
#             ans.append(prediction[i].item())
#
#     with open('sample.csv', 'w') as fout:
#         fieldnames = ['caseid', 'midprice']
#         writer = csv.DictWriter(fout, fieldnames=fieldnames)
#         writer.writeheader()
#         #start from case 143
#         for i in range(142,len(ans)):
#             writer.writerow({'caseid':str(i+1),'midprice':float(ans[i])})
#     #print(ans)
#
# def get_lr_decay(optimizer, epoch, decay_rate, init_lr):
#     #lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
#     lr = init_lr * ((1-decay_rate)**epoch)
#
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return optimizer
#
# if __name__ == '__main__':
#     torch.manual_seed(1)
#     batch_size = 32
#     hidden_dim = 128
#     device = 'cpu'
#     NUM_EPOCHES = 40
#     feature_num = 6
#     dataset = HFTDataset()
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     test_dataset = HFTTestDataset()
#     test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#     lstm = LSTM(feature_num, hidden_dim, 10, device).to(device)
#     #lstm2 = LSTM(feature_num, hidden_dim, 10, device).to(device)
#
#     loss_fn = nn.L1Loss()
#
#
#
#     #lr = 0.1
#     #sgd_optimizer = torch.optim.SGD(lstm.parameters(), lr = lr)
#     #lr_decay = 0.05
#     optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001, betas=(0.9, 0.999))
#
#     print("========Print Model========")
#     #print(lstm)
#     print("========End Printing========")
#     train()
#     test(20)
#
#
#
#     # f = open("all/test_data.csv", "r")
#     # reader = csv.DictReader(f)
#     # fieldnames = ['LastPrice']
#     #
#     # midPrices = []
#     # dates = []
#     # times = []
#     # lastPrices = []
#     # volumes = []
#     # bidPrices = []
#     # bidVolumes = []
#     # askPrices = []
#     # askVolumes = []
#     # microPrices = []
#     # ans = []
#     # dic = {'LastPrice': lastPrices}
#     # for row in reader:
#     #     for field in fieldnames:
#     #         dic[field].append(float(row[field]))
#     #
#     # for i in range(int(len(dic['LastPrice'])/10)-1):
#     #     ans.append(0.5 * (dic['LastPrice'][i*10] + dic['LastPrice'][i*10+10]))
#     # ans.append(dic['LastPrice'][-1])
#     #
#     # with open('sample.csv', 'w') as fout:
#     #     fieldnames = ['caseid', 'midprice']
#     #     writer = csv.DictWriter(fout, fieldnames=fieldnames)
#     #     writer.writeheader()
#     #     #start from case 143
#     #     for i in range(142,len(ans)):
#     #         writer.writerow({'caseid':str(i+1),'midprice':float(ans[i])})