此项目由两个py文件组成，分别是test0.py和test1.py

test0.py用于数据预处理，去除数据的Date和Time列，运行时注意修改读取文件名和写入文件名

test1.py用于建立数据集、建立模型、训练模型以及推理
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True)函数用于将文件中读取的数据转换成可供模型训练和推理用的数据
        data为从文件读取的数据
        n_in表示前序序列长度，此实验中设为10
        n_out表示输出维度，此实验只需要预测midprice，所以设为1

    def rmse(X, y)函数用于计算验证集经模型推理后得到的结果的RMSE
        X为验证集数据
        y为验证集正确结果

    model.add(LSTM(500, input_shape=(train_X.shape[1], train_X.shape[2])))语句用于添加LSTM层到模型，500为神经元个数，可以调整

    history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
        可以设置epochs和batch_size

    最终输出一个名为result.csv的文件

result1.csv 和 result2.csv为提交的两个结果 