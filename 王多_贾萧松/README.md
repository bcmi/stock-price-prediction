本组Public Board上的最好成绩为0.00131 提交的为0.00131和0.00132的两组，因为Stack的模型有随机性，所以我们提交了两组由Stack产生的结果

最终Private Board上两组的成绩均是0.00125（Lucky Board Gap~） 仍然比本地cross validation的结果差一些(Bad Local Gap~)

下面是执行流程及代码说明：

1.数据清洗 python process_raw_data.py

做的事情有：将训练数据按天且按上下午隔开（也就是说数据不会有跨天或跨中午的预测）

同时洗去连续的两组完全相同的数据（减少了大量噪声，因为测试数据中观察可以发现，两组完全一样的情况是很少的）+洗去12点的数据

给train 和 test 增加feature：一周的周几，一天的第几个小时（由于这种离散变量需要One hot编码，所以test数据中12点的那一组，我们令其值为11） + http://www.infosec-wiki.com/?p=69667  中提到的特征

得到清洗后的数据train_data_2.csv test_data_2.csv

2.特征工程 python generate_features.py

根据论文：

Modeling high-frequency limit order book dynamics with support vector machines

Using Deep Learning for price prediction by exploiting stationary limit order book features

增加特征

并将十个时间点的数据放在一行，同时得到训练数据的目标Y

得到最终数据train_data_3.csv  test_data_3.csv

3. 训练模型
python xgboost+lr+stack.py

a. GridSearch找出最优Xgboost超参数（已经设置好，已注释）

b. 根据数据及超参数，通过Xgboost 中的feature_importances_参数，判断不同参数数目下的最优结果，以求获得参数数目和结果间的平衡（已经设置好，已注释）

c. Xgboost + linear regression

这是根据Facebook的论文：

Practical Lessons from Predicting Clicks on Ads at Facebook 

改编的一种方法

d. Stack是Kaggle比赛中的常见技巧，这里第一层：我采用了36个Xgboost+linear regression 每个给定在一定范围内随机数目的feature以及随机扰动的超参数进行训练 然后第二层使用Linear regression进行回归的得到最终结果

模型融合具体原理可以参考：

http://www.dcjingsai.com/common/bbs/topicDetails.html?tid=348

https://www.kaggle.com/c/otto-group-product-classification-challenge/discussion/14335

https://blog.csdn.net/a358463121/article/details/53054686

结果输出在以执行起始时间为名字的文件夹里: xxx.csv

4. 产生输出

由于我回归的任务是：求相对前10个点的平均值，后20个时间点的平均值的变化量，所以现在要将预测的变化量加上前10个点的平均值。

流程：

进入out.py将 resfile变量的文件名改为输入文件名xxx.csv，获得输出xxx_out.csv

然后python out.py

删掉1-142case 提交

trick：经过反复观察数据，我们发现11:29:5x(12:xx:xx)的值趋向于稳定不变，因此对于11:29以及12：xx的预测，我们直接输出最后一个时刻的值!

