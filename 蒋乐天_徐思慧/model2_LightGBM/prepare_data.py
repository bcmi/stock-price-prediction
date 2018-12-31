import pandas as pd
import pickle

def time2min(time):
	list = time.split(':')
	h = int(list[0])
	m = int(list[1])
	return h*60 + m

def time2sec(time):
	list = time.split(':')
	h = int(list[0])
	m = int(list[1])
	s = int(list[2])
	return h*3600 + m*60 + s

X_train = []
y_train = []
X_test = []
m_test = []


#train
df = pd.read_csv("train_data.csv")

for i in range(0, len(df)-30, 5):
	if df.iloc[i].Date == df.iloc[i+29].Date and abs(time2sec(df.iloc[i].Time) - time2sec(df.iloc[i+9].Time)) == 27:
		X = []
		for j in range(1, 10):
			X.append(df.iloc[i+j].AskPrice1 - df.iloc[i+9].MidPrice)
			X.append(df.iloc[i+j].BidPrice1 - df.iloc[i+9].MidPrice)
			X.append(df.iloc[i+j].LastPrice - df.iloc[i+9].MidPrice)
			X.append(df.iloc[i+j].Volume - df.iloc[i+j-1].Volume)
			X.append(df.iloc[i+j].AskVolume1)
			X.append(df.iloc[i+j].BidVolume1)
		y = df.iloc[i+10:i+30].MidPrice.mean()  - df.iloc[i+9].MidPrice
		X_train.append(X)
		y_train.append(y)
	else:
		print(i)



#test
df = pd.read_csv("test_data.csv").dropna()

for i in range(1000): #df.shape[0] - 30
	if i < 142:
		continue
	print(i+1)
	i *= 10
	X = []
	for j in range(1, 10):
		X.append(df.iloc[i+j].AskPrice1 - df.iloc[i+9].MidPrice)
		X.append(df.iloc[i+j].BidPrice1 - df.iloc[i+9].MidPrice)
		X.append(df.iloc[i+j].LastPrice - df.iloc[i+9].MidPrice)
		X.append(df.iloc[i+j].Volume - df.iloc[i+j-1].Volume)
		X.append(df.iloc[i+j].AskVolume1)
		X.append(df.iloc[i+j].BidVolume1)
	X_test.append(X)
	m_test.append(df.iloc[i+9].MidPrice)

#pickle
with open("X_train.pkl", "wb") as f:
	pickle.dump(X_train, f)

with open("y_train.pkl", "wb") as f:
	pickle.dump(y_train, f)

with open("X_test.pkl", "wb") as f:
	pickle.dump(X_test, f)

with open("m_test.pkl", "wb") as f:
	pickle.dump(m_test, f)