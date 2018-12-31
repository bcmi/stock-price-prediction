import pandas as pd
import pickle

def loaddata(filename = 'data/train_data.csv'):
	df = pd.read_csv(filename)
	train_data = df.iloc[:,1:10]
	return train_data

def get_time(time_str):
    t = time_str.split(":")
    return 3600*int(t[0]) + 60*int(t[1]) + int(t[2])

def cut_data(dataset):
    pieces = []
    current_date = dataset.iloc[0, 0]
    current_t = get_time(dataset.iloc[0, 1])
    a_piece = []
    for i in range(1, len(dataset)):
        date = dataset.iloc[i, 0]
        t = get_time(dataset.iloc[i, 1])
        if current_date == date:
            if t - current_t < 3:
                continue
            elif t - current_t == 3:
                tmp = dataset.iloc[i, 2:].tolist()
                tmp.append(dataset.iloc[i, 4] - dataset.iloc[i-1, 4])
                if tmp[-1]<0:
                    tmp[-1] = 0
                a_piece.append(tmp)
                current_date = date
                current_t = t
                continue        
        if len(a_piece)>= 30:
            pieces.append(a_piece)
        tmp = dataset.iloc[i, 2:].tolist()
        tmp.append(dataset.iloc[i, 4] - dataset.iloc[i-1, 4]) 
        if tmp[-1] < 0:
            tmp[-1] = 0
        a_piece = [tmp]
        current_date = date
        current_t = t

    if len(a_piece)>= 30:
        pieces.append(a_piece)
    
    return pieces

a = loaddata()

print(a[:3])
print(type(a.iloc[0,1]))
time_a = get_time(a.iloc[1, 1])
print(time_a)

pieces = cut_data(a)
count = 0
b = 0
for p in pieces:
    count += len(p)
    b += len(p) - 29
print(count)
print(b)
print(len(pieces), len(pieces[0]), len(pieces[0][0]))
# f = open("data/cut_data", 'wb')
# pickle.dump(pieces, f)
# print(pieces[0])
# print(pieces[1])