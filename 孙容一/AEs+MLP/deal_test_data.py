import csv
import pickle

file1=csv.reader(open("test_data.csv",'r'))
f=open("test_data.pkl",'wb')
test_data=[]
count=0
for i in file1:
  if count==0:
    count+=1
    continue
  if i==[]:
    continue
  else:
    x=[float(tmp) for tmp in i[3: ]]
    test_data.append(x)




start=0
while start<len(test_data)-10:
  end=start+10
  pre=test_data[start][2]
  i=start+1
  while i<end:
    tmp=pre
    pre=test_data[i][2]
    test_data[i][2]-=tmp
    i+=1
  test_data[start][2]=test_data[start+1][2]
  start=end

pickle.dump(test_data,f)   
f.close()

for i in test_data:
  print(i[0])