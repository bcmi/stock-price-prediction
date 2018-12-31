import csv
import pickle
# file=csv.reader(open("train_data.csv",'r'))
# f=open("train_data.pkl","wb")
# train_data=[]
# count=0
# pre=""
# for i in file:
#   if count==0:
#      count=1
#      continue
#   if count==1:
#       count+=1
#       pre=i[2]
#       x=[]
#       for index in range(3,len(i)):
#         x.append(float(i[index]))
#         train_data.append(x)
#   if i[2]!=pre:
#     x=[]
#     pre=i[2]
#     for index in range(3,len(i)):
#       x.append(float(i[index]))
#     train_data.append(x)

# print(len(train_data))
# pickle.dump(train_data,f)
# f.close()
file=csv.reader(open("train_data.csv",'r'))
f=open("train_data.pkl","wb")
train_data=[]
count=0
pre_data="2018-06-01"
pre_time='09:30:01'
#pre_volume=0
day_index=[0]
index=1
for row in file:
  if count==0:
    count=1
    continue
  if row[1]!=pre_data:
    day_index.append(index)
    #print(pre_data,row[1])
    pre_time=row[2]
    pre_data=row[1]
    x=[ float(tmp) for tmp in row[3:]]
    #pre_volume=x[2]
    train_data.append(x)
    index+=1
  else:
    if row[2]!=pre_time:
      pre_time=row[2]
      pre_data=row[1]
      x=[float(tmp) for tmp in row[3:]]
      #tmp=pre_volume
      #pre_volume=x[2]
      #x[2]-=tmp
      train_data.append(x)
      index+=1
day_index.append(len(train_data))
pickle.dump(train_data,f)
pickle.dump(day_index,f)

f.close()


