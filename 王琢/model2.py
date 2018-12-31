import torch
from sys import platform
from tqdm import tqdm_notebook
import pandas as pd
import numpy as np

if platform != 'darwin':
    device_gpu = torch.device("cuda:1")
    device = device_gpu
    torch.cuda.set_device(device.index)
device_cpu = torch.device("cpu")
device = device_cpu
if torch.cuda.is_available():
    default_gpu_tensor_type = torch.cuda.FloatTensor
default_cpu_tensor_type = torch.FloatTensor
if device == device_cpu:
    torch.set_default_tensor_type(default_cpu_tensor_type)
else:
    torch.set_default_tensor_type(default_gpu_tensor_type)

class CSVReader:
    dataitems = ["MidPrice", "LastPrice", "Volume", "BidPrice1", "BidVolume1", "AskPrice1", "AskVolume1"]
    def __init__(self, training_set="./train_data.csv", testing_set="./test_data.csv"):
        self.Train = pd.read_csv(training_set,
                                 index_col="Date",
                                 usecols=[
                                     "Date", "Time",
                                     "MidPrice", "LastPrice",
                                     "Volume", "BidPrice1",
                                     "BidVolume1", "AskPrice1",
                                     "AskVolume1"
                                 ])

        self.Test = pd.read_csv(testing_set,
                                index_col="Date",
                                usecols=[
                                    "Date", "Time",
                                    "MidPrice", "LastPrice",
                                    "Volume", "BidPrice1",
                                    "BidVolume1", "AskPrice1",
                                    "AskVolume1"
                                ])


        self.Train = self.Train.sort_index()
        self.Test = self.Test.sort_index()
        def hour(s):
            q = [float(i) for i in s.split(":")]
            return q[0] + q[1] / 60

        self.Train["Hour"] = self.Train["Time"].map(hour)

        self.Test["Hour"] = self.Test["Time"].map(hour)

        TimeStampCount = self.Train["Time"].groupby("Date").count()
        TimeStampCount = TimeStampCount.sort_values()

        self.TrainDates = self.Train.index.unique().tolist()
        self.DangerousDates = TimeStampCount[TimeStampCount > 5000].index.tolist()
        self.FilteredDates = [date for date in self.TrainDates if date not in self.DangerousDates]

        self.TrainSet = {}

        self.AM = self.Train[self.Train["Hour"] < 11.70]
        self.PM = self.Train[self.Train["Hour"] > 12.70]

        for date in self.FilteredDates:
            self.TrainSet[f"{date}|AM"] = self.AM.loc[date]
            self.TrainSet[f"{date}|PM"] = self.PM.loc[date]

        # Splitting Testing Set
        self.TestingSet = []

        for begin in range(0, len(self.Test), 10):
            self.TestingSet.append(self.Test.iloc[begin: begin + 10])

    def training_dates(self):
        """
        Returning all training set dates
        """
        return self.FilteredDates

    def get_training_numpy(self, idx):
        """
        Returning training set at idx, also the am / pm / date infomation trailing it.
        [T, Feature]
        """
        key = list(self.TrainSet.keys())[idx]
        pandadb = self.TrainSet[key][self.dataitems]
        return key, pandadb.values

    def get_testing_numpy(self, idx):
        """
        Returning testing set at idx, also the am / pm / date infomation trailing it.
        [T, Feature]
        """
        pandadb = self.TestingSet[idx][self.dataitems]
        return pandadb.values

    def training_count(self):
        """
        Returning the count of all available sub training set, including morning and afternoon
        """
        return len(self.TrainSet)

    def testing_count(self):
        """
        Returning the count of all testing instance
        """
        return len(self.TestingSet)


class DataLoader:
    Q = torch.arange(0, 30, dtype=torch.long).unsqueeze(0)

    def __init__(self, csvreader=CSVReader()):
        self.Train = []
        self.Test = []

        self.reader = csvreader
        for idx in range(self.reader.training_count()):
            time, npdata = self.reader.get_training_numpy(idx)
            self.Train.append(torch.from_numpy(npdata).contiguous().to(device=device, dtype=torch.float32))

        for idx in range(self.reader.testing_count()):
            npdata = self.reader.get_testing_numpy(idx)
            self.Test.append(torch.from_numpy(npdata).contiguous().to(device=device, dtype=torch.float32))

        self.valid_idx = []
        self.train_idx = [_ for _ in range(0, len(self.Train))]
        self.set_validation_set([30, 31])

    def sample_batch(self, batch_size=32, source="train", full=False):
        if source == "train":
            choice = np.random.choice(self.train_idx)
        else:
            choice = np.random.choice(self.valid_idx)

        L = len(self.Train[choice])
        IDX = torch.randint(0, L - 30, (batch_size,))
        A = self.Train[choice]
        Q = self.Q
        P = IDX.unsqueeze(1)
        W = A[P + Q]
        M = torch.mean(W[:, 10:, 0], dim=1)
        if full:
            return W[:, :30, :], M
        else:
            return W[:, :10, :], M

    def sample_train(self, batch_size=32, full=False):
        return self.sample_batch(batch_size, full=full)

    def sample_valid(self, batch_size=32, full=False):
        return self.sample_batch(batch_size, source="valid", full=full)

    def get_test(self):
        # Returns all 1000 testing samples
        return torch.stack(self.Test, dim=0)

    def set_validation_set(self, lst):
        self.valid_idx = lst
        self.train_idx.clear()
        for idx in range(0, len(self.Train)):
            if idx not in lst:
                self.train_idx.append(idx)


class Trainer:
    def __init__(self, model, loader):
        self.model = model
        self.loader = loader
        self.loss = torch.nn.MSELoss()
        if issubclass(type(self.model), torch.nn.Module):
            self.optim = torch.optim.Adam(model.parameters(), lr=0.01)

    def train(self, round=20000, batch_size=128, lr=0.0001, validation=(10, 17), cp=0.1, penalty=0.1):
        valid = 1
        global_valid = 1
        assert issubclass(type(self.model), torch.nn.Module), "Can only train NN Models"
        self.optim.lr = lr
        bar = tqdm_notebook(iterable=range(round), desc="Training")
        self.loader.set_validation_set(validation)
        for i in bar:
            Trainset, Target = self.loader.sample_train(batch_size)
            self.optim.zero_grad()
            Estimation = self.model(Trainset)
            Loss = self.rmse_loss(Target, Estimation) + self.model.penalty()
            bar.set_postfix(loss=f"{Loss.detach().cpu():0.6f}", valid=valid, gbvalid=global_valid)
            Loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cp)
            self.optim.step()
            if i % 1024 == 1:
                valid = self.validate()
                global_valid = self.validate_all()
                self.loader.set_validation_set(validation)

        return Loss

    def validate(self, valid_size=256):
        Validset, Target = self.loader.sample_valid(valid_size)
        Estimation = self.model(Validset)
        Loss = self.rmse_loss(Target, Estimation)
        Loss = Loss.detach().cpu().item()
        return Loss

    def validate_all(self):
        p = []
        for i in range(self.loader.reader.training_count()):
            self.loader.set_validation_set([i])
            p.append(self.validate(valid_size=1024))

        return torch.mean(torch.tensor(p)).detach().cpu().item()

    def rmse_loss(self, target, estimate):
        return torch.sqrt(self.loss(target, estimate))

    def test(self):
        """
        Evaluate the model on the valid data
        Output the result to submission.csv
        """
        Testset = self.loader.get_test()
        print(Testset.shape)
        Estimation = self.model(Testset)
        self.printing(Estimation)
        print(Estimation.shape)

    def printing(self, result, begin=143):

        t = pd.DataFrame(result[begin - 1:].unsqueeze(1).detach().cpu().numpy(), columns=["midprice"],
                         index=np.arange(begin, result.size(0) + 1))
        t.index.name = 'caseid'
        t.to_csv("submission.csv")
        

