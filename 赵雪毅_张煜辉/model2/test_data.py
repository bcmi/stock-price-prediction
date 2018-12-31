import csv
from functools import reduce
#测试数据预处理
#1000


def str2float(s):

    def fn(x,y):

        return x*10+y

    n=s.index('.')

    s1=list(map(int,[x for x in s[:n]]))

    s2=list(map(int,[x for x in s[n+1:]]))

    return reduce(fn,s1)+reduce(fn,s2)/(10**len(s2))#乘幂


sign=0
a=[]
b=[]

with open("test_data.csv") as file:
    for line in  file:
        if(line[0]=='4'):
            tmp=line.split(',')

            a.append(float(tmp[6]))
            a.append(float(tmp[8]))
            a.append(float(tmp[7])/float(tmp[9]))
            sign=sign+1
            if(sign%10==0):
                b.append(a)
                a=[]
print(len(b))


'''                
for i in range(9999):
    if(i%10==9):
        b.append(0.67*a[i]+0.33*a[i+1])
b.append(a[9999])
'''

out=open('tmp2.csv','a',newline='')
imf=[]
csv_write=csv.writer(out,dialect='excel')
for i in range(30):
    imf.append(i)
csv_write.writerow(imf)
imf=[]
for i in range(1000):
    for j in range(30):
        imf.append(b[i][j])
    csv_write.writerow(imf)
    imf=[]
out.close()
        

