import pandas as pd
data = pd.read_csv('test_data.csv')
#data = data.dropna()
for i in range(1000):
    midprice_dict = {'MidPrice':[],'BidPrice1':[],"BidVolume1":[],"AskVolume1":[]}
    for j in range(10):
        MidPrice = data['MidPrice'][i*11+j]
        midprice_dict['MidPrice'].append(MidPrice)

        BidPrice1 = data['BidPrice1'][i*11+j]
        midprice_dict['BidPrice1'].append(BidPrice1)

        BidVolume1 = data['BidVolume1'][i*11+j]
        midprice_dict['BidVolume1'].append(BidVolume1)

        AskVolume1 = data['AskVolume1'][i*11+j]
        midprice_dict['AskVolume1'].append(AskVolume1)

    midprice_dict['MidPrice'].append(MidPrice)
    midprice_dict['BidPrice1'].append(BidPrice1)
    midprice_dict['BidVolume1'].append(BidVolume1)
    midprice_dict['AskVolume1'].append(AskVolume1)
    data_frame = pd.DataFrame(midprice_dict)
    data_frame.to_csv("minidata"+str(i)+'.csv',index=False,sep=',')

