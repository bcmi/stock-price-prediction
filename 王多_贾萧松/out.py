import sys
import csv

resfile = "\\2018-12-24-20_20_16\res_2018-12-24-20_20_16.csv"




with open("test_data.csv") as ts:
    with open(resfile) as rf:
        reader1 = csv.reader(rf)
        title1 = next(reader1)
        reader2 = csv.reader(ts)
        title2 = next(reader2)
        prev = []
        with open(resfile.rpartition('.')[0]+'_out.csv','w') as of:
            for t in title1[:-1]:
                of.write(t+',')
            of.write(title1[-1]+'\n')
            i = 0
            cnt = 0
            j = 1
            total = 0
            for row2 in reader2:
                j += 1
                if not row2[0]:
                    total = 0
                    continue

                i += 1
                total += float(row2[3])
                if i%10 != 0:
                    continue
                #print(j)
                row1 = next(reader1)
                if prev and (prev[2].startswith("11:2") and (row2[2].startswith("12") or row2[2].startswith("13"))):
                    of.write(row1[0]+','+row2[3]+'\n')
                    #print(j)
                    cnt += 1
                else:
                    of.write(row1[0]+','+str((float(row1[1])+total/10.0))+'\n')#str(float(row2[3])*(float(row1[1])))+'\n')
                    # of.write(row1[0]+','+str(float(row1[1])*(max_p-min_p)+min_p)+'\n')
                prev = row2


print(cnt)