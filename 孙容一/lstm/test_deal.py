import numpy as np
import csv
import pickle

file=open("result.pkl","rb")
ans=pickle.load(file)
file.close()
print(ans.shape)

out=open("result4.csv",'w',newline='')
csv_write=csv.writer(out,dialect='excel')
ans.tolist()
stu=['caseid','midprice']
csv_write.writerow(stu)
for i in range(142,len(ans)):
   stu=[i+1,ans[i][0]]
   csv_write.writerow(stu)
