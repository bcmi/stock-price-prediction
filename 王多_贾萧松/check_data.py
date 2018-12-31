import csv
import time
def get_unix(row):
    return float(time.mktime(time.strptime(row[1]+' '+row[2],"%Y-%m-%d %H:%M:%S")))#time.mktime(time.strptime(row[1]+' '+row[2],"%Y/%m/%d %H:%M:%S")))

# # with open("test_data.csv") as f:
# #     content = f.readlines()
# #     set_cnt = 1
# #     prev = []
# #     cnt = 0
# #     for i in range(1, len(content)):
# #         line = content[i]
# #         if(line.startswith(",,")):
# #             set_cnt += 1
# #             prev = []
# #             continue
# #         line = line.strip("\n").split(",")
# #         if prev and set_cnt>=143 and line[2] != "11:29:59":
# #             if(get_unix(line)-get_unix(prev) != 3.0):
# #                 cnt += 1
# #                 print(i, prev[2], line[2])
# #         prev = line
# #     print(cnt)

with open("train_data.csv") as f:
    content = f.readlines()
    set_cnt = 1
    prev = []
    cnt = 0
    for i in range(1, len(content)):
        line = content[i]
        line = line.strip("\n").split(",")
        if prev and line[2] != "11:29:59":
            if(get_unix(line)-get_unix(prev) != 3.0):
                cnt += 1
                print(i, prev[2], line[2])
        prev = line
    print(cnt)