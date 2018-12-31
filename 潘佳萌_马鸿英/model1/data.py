import pandas as pd
data = pd.read_csv('test_data.csv')

data = data.drop(data.index[:1420])
data.to_csv("s.csv")