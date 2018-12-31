import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
'''
important Notice!
This part is data preprocessing part of our project.
However, here the train set and test set preprocessing are incompatible.
You can find that training part is more complicated than testing part, for 
we have retain some failed attempts here in training part.
Anyone want to regenerate the model should modify training part to be compatible 
with test part.


'''
totalsum = 0
def getSum(row):
    global totalsum
    totalsum += row["MidPrice"]
    return totalsum

def getTime(row):
    pdtime = pd.to_datetime(row["Date"] + " " + row["Time"])
    return pdtime

def getDayofWeek(row):
    day = row["Timestamp"].dayofweek
    return day

def getDayofmonth(row):
    day = row["Timestamp"].month
    return day

def getNoon(row):
    noontime = pd.Timestamp(row["Date"] + " 12:00:00")
    tdl = row["Timestamp"] - noontime
    if (tdl.total_seconds() > 0):
        return 0
    else:
        return 1

def duration(row):
    if (row["BeforeNoon"] == 1):
        stand = pd.Timestamp(row["Date"] + " 9:30:00")
    else:
        stand = pd.Timestamp(row["Date"] + " 13:00:00")
    tdl = row["Timestamp"] - stand
    return tdl.total_seconds()

def bidvminusaskv(row):
    return row["BidVolume1"] - row["AskVolume1"]

def bidvdivideaskv(row):
    return row["BidVolume1"] / row["AskVolume1"]

def bidpminusaskp(row):
    return row["BidPrice1"] - row["AskPrice1"]

def bidpdivideaskp(row):
    return row["BidPrice1"] / row["AskPrice1"]
    
def process_train_test():
    f = open('train_data.csv')
    df = pd.read_csv(f)

    df_duplicate = df.duplicated(subset = ["Time", "Date"], keep = False)
    dat = df_duplicate.data

    total = 0
    for i in range(len(dat)):
        if dat[i] == True:
            total = total + 1
            nam = df.loc[i, "Unnamed: 0"]
            df.drop(nam, inplace = True)

    df.drop(labels = 'Unnamed: 0', axis = 1, inplace = True)


    df["Timestamp"] = df.apply(lambda row:getTime(row), axis = 1)
    #df["DayofWeek"] = df.apply(lambda row:getDayofWeek(row), axis = 1)
    #df["DayofMonth"] = df.apply(lambda row:getDayofmonth(row), axis = 1)
    df["BeforeNoon"] = df.apply(lambda row:getNoon(row), axis = 1)
    df["Duration"] = df.apply(lambda row:duration(row), axis = 1)
    df["BidV-AskV"] = df.apply(lambda row:bidvminusaskv(row), axis = 1)
    #df["BidV/AskV"] = df.apply(lambda row:bidvdivideaskv(row), axis = 1)
    df["BidP-AskP"] = df.apply(lambda row:bidpminusaskp(row), axis = 1)
    #df["BidP/AskP"] = df.apply(lambda row:bidpdivideaskp(row), axis = 1)

    df.drop(labels = 'Volume', axis = 1, inplace = True)
    df.drop(labels = 'Date', axis = 1, inplace = True)
    df.drop(labels = 'Time', axis = 1, inplace = True)
    #df.drop(labels = 'MidPrice', axis = 1, inplace = True)
    df.drop(labels = 'Timestamp', axis = 1, inplace = True)


    preserve = pd.DataFrame(df[["MidPrice", "Duration","BidVolume1", "AskVolume1", "BidV-AskV","BidP-AskP"]])

    # Polynomial attempts
    # #min_max_scaler = MinMaxScaler()
    # #x_minmax = min_max_scaler.fit_transform(df.values)
    # poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    # X_ploly = poly.fit_transform(x_minmax)
    # X_ploly_df = pd.DataFrame(X_ploly, columns=poly.get_feature_names(df.columns))
    # df = X_ploly_df

    col_name = []
    exist_name = df.columns.values.tolist()

    for i in range(1, 11):
        for name in exist_name:
            col_name.append(name + str(i))

    col_name.append("MidPrice1dif")
    col_name.append("MidPrice2dif")
    col_name.append("MidPrice3dif")
    col_name.append("MidPrice5dif")
    col_name.append("MidPrice8dif")
    
    col_name.extend(["BV1dif","BV2dif","BV3dif", "BV5dif","BV8dif"])

    col_name.extend(["AV1dif","AV2dif","AV3dif", "AV5dif","AV8dif"])

    col_name.append("MidPriceAverage")
    col_name.append("MidPriceVar")
    col_name.append("MidPriceSkew")

    col_name.append("VdiffAverage")
    col_name.append("VdiffVar")

    col_name.append("PdiffAverage")
    col_name.append("PdiffVar")

    tmp_row = []
    refined_data = []

    for row in range(len(df) - 30):
        if (subrow == 30):
            subrow = 0
            pricesum = 0
            for i in range(20):
                pricesum += preserve.iloc[row - 1 - i]["MidPrice"]

            tmp_row.append(pricesum / 20 - preserve.iloc[row - 20]["MidPrice"])

            lastindex = row - 20
            lastmidprice = preserve.iloc[lastindex]["MidPrice"]
            lastbidvolume = preserve.iloc[lastindex]["BidVolume1"]
            lastaskvolume = preserve.iloc[lastindex]["AskVolume1"]

            for d in [1, 2, 3, 5, 8]:
                tmp_row.append(lastmidprice - preserve.iloc[lastindex - d]["MidPrice"])
            for d in [1, 2, 3, 5, 8]:
                tmp_row.append(lastbidvolume - preserve.iloc[lastindex - d]["BidVolume1"])
            for d in [1, 2, 3, 5, 8]:
                tmp_row.append(lastaskvolume - preserve.iloc[lastindex - d]["AskVolume1"])

            miu = np.mean(tmp_row[0 :len(exist_name) * 10: 10])
            var = np.var(tmp_row[0 :len(exist_name) * 10: 10])

            dif  = 0
            tmpmid  = tmp_row[0:len(exist_name) * 10:10]
            for i in range(len(tmpmid)):
                dif +=  (tmpmid[i] - miu) ** 3 / var ** 1.5
            skew = dif / len(tmpmid)

            tmp_row.append(miu)
            tmp_row.append(var)
            tmp_row.append(skew)

            vmiu = np.mean(tmp_row[8:len(exist_name) * 10:10])
            vvar = np.var(tmp_row[8:len(exist_name) * 10:10])
            tmp_row.append(vmiu)
            tmp_row.append(vvar)

            pmiu = np.mean(tmp_row[9:len(exist_name) * 10:10])
            pvar = np.var(tmp_row[9:len(exist_name) * 10:10])
            tmp_row.append(pmiu)
            tmp_row.append(pvar)
                
            refined_data.append(tmp_row)

            tmp_row = []
            row -= 10

        cur_time = preserve.iloc[row]["Duration"]
        follow_time = preserve.iloc[row + 1]["Duration"]

        if (follow_time - cur_time - 3 < 2e-7 and subrow < 10):
            tmp_row.extend(df.iloc[row].values)
            subrow += 1
        elif (follow_time - cur_time - 3 > 2e-7):
            tmp_row = []
            subrow = 0
        else:
            subrow += 1


    min_max_scaler = MinMaxScaler()
    x_minmax = min_max_scaler.fit_transform(refined_data)

    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    X_ploly = poly.fit_transform(x_minmax)
    
    X_ploly_df = pd.DataFrame(X_ploly, columns=poly.get_feature_names(col_name))

    X_ploly_df.to_csv('xgbtrain.csv')




    # test set treatment
    f = open('test_data.csv')
    df = pd.read_csv(f)
    df.drop(labels = 'Unnamed: 0', axis = 1, inplace = True)

    df["Timestamp"] = df.apply(lambda row:getTime(row), axis = 1)
    #df["DayofWeek"] = df.apply(lambda row:getDayofWeek(row), axis = 1)
    #df["DayofMonth"] = df.apply(lambda row:getDayofmonth(row), axis = 1)
    df["BeforeNoon"] = df.apply(lambda row:getNoon(row), axis = 1)
    df["Duration"] = df.apply(lambda row:duration(row), axis = 1)
    df["BidV-AskV"] = df.apply(lambda row:bidvminusaskv(row), axis = 1)
    df["BidV/AskV"] = df.apply(lambda row:bidvdivideaskv(row), axis = 1)
    df["BidP-AskP"] = df.apply(lambda row:bidpminusaskp(row), axis = 1)
    df["BidP/AskP"] = df.apply(lambda row:bidpdivideaskp(row), axis = 1)


    df.drop(labels = 'Volume', axis = 1, inplace = True)
    df.drop(labels = 'Date', axis = 1, inplace = True)
    df.drop(labels = 'Time', axis = 1, inplace = True)
    #df.drop(labels = 'MidPrice', axis = 1, inplace = True)
    df.drop(labels = 'Timestamp', axis = 1, inplace = True)

    preserve = pd.DataFrame(df[["MidPrice", "Duration"]])

    for i in range(len(df)):
        if (len(df.iloc[i]) < 13):
            print(df.iloc[i])

    df = df[1420:]
    preserve = preserve[1420:]

    col_name = []

    exist_name = df.columns.values.tolist()
    for i in range(1, 11):
        for name in exist_name:
            col_name.append(name + str(i))


    subrow = 0
    
    tmp_row = []
    refined_data = []
    for row in range(len(df)):
        if (row == len(df) - 1):
            refined_data.append(tmp_row)
        if (subrow == 10):
            subrow = 0

            refined_data.append(tmp_row)
            tmp_row = []

        tmp_row.extend(df.iloc[row].values)
        subrow += 1

    newdf = pd.DataFrame(data = refined_data, columns = col_name)

    newdf.to_csv('xgbtest.csv')
    preserve.to_csv('preserve.csv')



#process_train_test()
