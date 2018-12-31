import csv
from functools import reduce
#训练数据预处理
#430039
a=[]
b=[]

def str2float(s):

    def fn(x,y):

        return x*10+y

    n=s.index('.')

    s1=list(map(int,[x for x in s[:n]]))

    s2=list(map(int,[x for x in s[n+1:]]))

    return reduce(fn,s1)+reduce(fn,s2)/(10**len(s2))#乘幂


sign=0
sum=0


with open("train_data.csv") as file:
    for line in  file:
        if('D' not in line):
            tmp=line.split(',')
            if (sign%30<10):
                a.append(float(tmp[6]))
                a.append(float(tmp[8]))
                a.append(float(tmp[7])/float(tmp[9]))
                sign=sign+1
            elif(sign%30<29):
                sum=sum+str2float(tmp[3])
                sign=sign+1
            elif(sign%30==29):
                sum=sum+str2float(tmp[3])
                ave=sum/20
                a.append(ave)
                b.append(a)
                sum=0
                a=[]
                sign=sign+1
                if(sign==430020):
                    break
print(len(b))


'''                
for i in range(9999):
    if(i%10==9):
        b.append(0.67*a[i]+0.33*a[i+1])
b.append(a[9999])
'''

out=open('tmp1.csv','a',newline='')
imf=[]
csv_write=csv.writer(out,dialect='excel')
for i in range(31):
    imf.append(i)
csv_write.writerow(imf)
imf=[]
for i in range(14334):
    for j in range(31):
        imf.append(b[i][j])
    csv_write.writerow(imf)
    imf=[]
out.close()
        
