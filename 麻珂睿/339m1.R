setwd("C:/Users/simon_zsy/Desktop/")
library(dplyr)
train1.df = read.table("C:/Users/simon_zsy/Desktop/339/train_data11.csv",sep=",",header=TRUE)
str(train1.df)
summary(train1.df)
#train1.df = train1.df[-c(4771, 4772,4773),]
#LastPrice = train.df$LastPrice[-1]
#LastPrice = class.ind(LastPrice)
##LastPrice = c(LastPrice, NA)
##LastPrice.df = data.frame(LastPrice1 = LastPrice)
#train.df = bind_cols(train.df,LastPrice.df)
conbine.df = data.frame()
#for(a in 1:30){
train.df = train1.df[115771:425970,]
train.df = select(train.df,X ,MidPrice, LastPrice, Volume, BidPrice1, BidVolume1, AskPrice1, AskVolume1)
d0 = filter(train.df, X %% 30 == 0)
d1 = filter(train.df, X %% 30 == 1)
d2 = filter(train.df, X %% 30 == 2)
d3 = filter(train.df, X %% 30 == 3)
d4 = filter(train.df, X %% 30 == 4)
d5 = filter(train.df, X %% 30 == 5)
d6 = filter(train.df, X %% 30 == 6)
d7 = filter(train.df, X %% 30 == 7)
d8 = filter(train.df, X %% 30 == 8)
d9 = filter(train.df, X %% 30 == 9)
trainer.df = bind_cols(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)
trainer.df = select(trainer.df, -starts_with("X"))
#trainer.df = select(trainer.df, -starts_with("MidPrice"))
#trainer.df = select(trainer.df, -starts_with("LastPrice"))
train1.df = train1.df[-1,]

d10 = filter(train.df, X %% 30 == 10)
d11 = filter(train.df, X %% 30 == 11)
d12 = filter(train.df, X %% 30 == 12)
d13 = filter(train.df, X %% 30 == 13)
d14 = filter(train.df, X %% 30 == 14)
d15 = filter(train.df, X %% 30 == 15)
d16 = filter(train.df, X %% 30 == 16)
d17 = filter(train.df, X %% 30 == 17)
d18 = filter(train.df, X %% 30 == 18)
d19 = filter(train.df, X %% 30 == 19)
d20 = filter(train.df, X %% 30 == 20)
d21 = filter(train.df, X %% 30 == 21)
d22 = filter(train.df, X %% 30 == 22)
d23 = filter(train.df, X %% 30 == 23)
d24 = filter(train.df, X %% 30 == 24)
d25 = filter(train.df, X %% 30 == 25)
d26 = filter(train.df, X %% 30 == 26)
d27 = filter(train.df, X %% 30 == 27)
d28 = filter(train.df, X %% 30 == 28)
d29 = filter(train.df, X %% 30 == 29)
Mid.df = bind_cols(d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29)
Mid.df = select(Mid.df, starts_with("MidPrice"))
MidMean = rowMeans(Mid.df)
MidMean = data.frame(MidPointMean = MidMean)
conbine1.df = bind_cols(MidMean, trainer.df)
#conbine1.df = conbine1.df[-c(159, 317, 474,627,784,943,1099,1258,1414,1571,1729,1888,2047,2205,2363,2521,2679,	2838,	2997,	3156,	3315,	3474	,3632	,3791	,3950	,4109	,4267	,4426,	4585,	4744,	4899,	5058	,5217,	5376,	5535,	5693,	5852,6011,	6170,	6328	,6487	,6645	,6804	,6964	,7123	,7282,	7442,	7601	,7760,	7919,	8078	,8237	,8397,	8556	,8715	,9034,	9352,	9511,	9830,	9988,	10307,	10465,	10625	,10786	,10947,	11106	,11265,	11424,	11583,	11742,	11901,	12060,	12219,	12379,	12538,	12730,	12889	,13050,	13250,	13410	,13612,	13833),]
conbine.df = bind_rows(conbine.df, conbine1.df)
#}


#train.df = select(train.df, LastPrice = LastPrice1, Volume, BidPrice1, BidVolume1, AskPrice1, AskVolume1)
#install.packages("nnet")
#library(nnet)
#netPrice = nnet(LastPrice ~ BidPrice1 + BidVolume1 + AskPrice1 + AskVolume1, data = train.df)
#netPrice = nnet(train.df, LastPrice, size = 5)

#install.packages("neuralnet")
# library(neuralnet)
# maxs = apply(conbine.df, 2, max)
# mins = apply(conbine.df, 2, min)
# cbd = as.data.frame(scale(conbine.df, center = mins, scale = maxs - mins))
# f = as.formula(MidPointMean~MidPrice+LastPrice+Volume+BidPrice1+BidVolume1+AskPrice1+AskVolume1+MidPrice1+LastPrice1+Volume1+BidPrice11+BidVolume11+AskPrice11+AskVolume11+MidPrice2+LastPrice2+Volume2+BidPrice12+BidVolume12+AskPrice12+AskVolume12+MidPrice3+LastPrice3+Volume3+BidPrice13+BidVolume13+AskPrice13+AskVolume13+MidPrice4+LastPrice4+Volume4+BidPrice14+BidVolume14+AskPrice14+AskVolume14+MidPrice5+LastPrice5+Volume5+BidPrice15+BidVolume15+AskPrice15+AskVolume15+MidPrice6+LastPrice6+Volume6+BidPrice16+BidVolume16+AskPrice16+AskVolume16+MidPrice7+LastPrice7+Volume7+BidPrice17+BidVolume17+AskPrice17+AskVolume17+MidPrice8+LastPrice8+Volume8+BidPrice18+BidVolume18+AskPrice18+AskVolume18+MidPrice9+LastPrice9+Volume9+BidPrice19+BidVolume19+AskPrice19+AskVolume19)
# nePrice = neuralnet(f, data = cbd, hidden = 200, linear.output=T, threshold=0.002, lifesign = "full", learningrate = 0.1, stepmax = 1e+07, algorithm = "rprop+", err.fct = "sse", act.fct = "tanh")
# summary(nePrice)
# plot(nePrice)
# 
# library(gee)
# geePrice = gee(MidPointMean~., data = conbine.df)
# summary(geePrice)
# 
# #install.packages("RSNNS")
# library(RSNNS)
# rsPrice = mlp(conbine.df[,2:71], conbine.df[,1], size = 5 )
# 
# f2 = as.formula(MidPointMean~MidPrice+BidPrice1+AskPrice1+AskVolume1+LastPrice1+Volume1+BidPrice11+BidVolume11+AskPrice11+LastPrice2+BidPrice12+BidVolume12+AskVolume12+MidPrice3+LastPrice3+Volume3+BidPrice13+AskPrice13+MidPrice4+LastPrice4+Volume4+BidVolume14+AskPrice14+AskVolume14+MidPrice5+LastPrice5+Volume5+BidPrice15+AskPrice15+AskVolume15+MidPrice6+LastPrice6+Volume6+BidPrice16+BidVolume16+AskPrice16+AskVolume16+MidPrice7+LastPrice7+Volume7+BidPrice17+AskPrice17+MidPrice8+LastPrice8+Volume8+BidPrice18+BidVolume18+AskPrice18+AskVolume18+MidPrice9+LastPrice9+Volume9+BidPrice19+BidVolume19+AskPrice19+AskVolume19)
# glmPrice = glm(MidPointMean~., data = conbine.df)
# summary(glmPrice)
# 
# lmPrice = lm(MidPointMean~., data = conbine.df)
# summary(lmPrice)
# 
# #install.packages("rpart")
# library(rpart)
# rpPrice = rpart(MidPointMean~., data = conbine.df)
# 
# #install.packages("RWeka")
# library(RWeka)
# rwPrice = M5P(MidPointMean~., data = conbine.df)
# 
# #install.packages("AMORE")
# library(AMORE)
# maxs = apply(conbine.df, 2, max)
# mins = apply(conbine.df, 2, min)
# cbd = as.data.frame(scale(conbine.df, center = mins, scale = maxs - mins))
# net<-newff(n.neurons=c(70,40,1),learning.rate.global=1e-4,momentum.global=0.001, error.criterium="LMS", Stao=NA, hidden.layer="tansig", output.layer="purelin", method="ADAPTgd")
# amPrice = train(net, cbd[,2:71], cbd[,1], show.step=1000000, n.shows=5)
# 
# library(nnet)
# maxs = apply(conbine.df, 2, max)
# mins = apply(conbine.df, 2, min)
# cbd = as.data.frame(scale(conbine.df, center = mins, scale = maxs - mins))
# nnPrice = nnet(MidPointMean~., cbd, size = 20, maxit = 10000000, linout = T, MaxNWts = 10000, abstol = 0.5) 
# summary(nnPrice)
# 
# #install.packages("e1071")
# library(e1071)
# svmPrice = svm(MidPointMean~., conbine.df)
# 
# #install.packages("earth")
# library(earth)
# eaPrice = earth(MidPointMean~., conbine.df)
# 
# #install.packages("glmnet")
# library(glmnet)
# gnPrice = glmnet(as.matrix(conbine.df[,2:71]), conbine.df[,1], family = "gaussian")

library(mgcv)
library(lme4)
f0 = as.formula(MidPointMean~s(LastPrice)+s(	Volume)+s(	BidPrice1)+s(	BidVolume1)+s(	AskPrice1)+s(	AskVolume1)+s(	LastPrice1)+s(	Volume1)+s(	BidPrice11)+s(	BidVolume11)+s(	AskPrice11)+s(	AskVolume11)+s(	LastPrice2)+s(	Volume2)+s(	BidPrice12)+s(	BidVolume12)+s(	AskPrice12)+s(	AskVolume12)+s(	LastPrice3)+s(	Volume3)+s(	BidPrice13)+s(	BidVolume13)+s(	AskPrice13)+s(	AskVolume13)+s(	LastPrice4)+s(	Volume4)+s(	BidPrice14)+s(	BidVolume14)+s(	AskPrice14)+s(	AskVolume14)+s(	LastPrice5)+s(	Volume5)+s(	BidPrice15)+s(	BidVolume15)+s(	AskPrice15)+s(	AskVolume15)+s(	LastPrice6)+s(	Volume6)+s(	BidPrice16)+s(	BidVolume16)+s(	AskPrice16)+s(	AskVolume16)+s(	LastPrice7)+s(	Volume7)+s(	BidPrice17)+s(	BidVolume17)+s(	AskPrice17)+s(	AskVolume17)+s(	LastPrice8)+s(	Volume8)+s(	BidPrice18)+s(	BidVolume18)+s(	AskPrice18)+s(	AskVolume18)+s(	LastPrice9)+s(	Volume9)+s(	BidPrice19)+s(	BidVolume19)+s(	AskPrice19)+s(	AskVolume19))
f1 = as.formula(MidPointMean~s(LastPrice)+s(MidPrice9)+s(Volume)+s(BidPrice1)+s(BidVolume1)+s(AskPrice1)+s(AskVolume1)+s(LastPrice1)+s(Volume1)+s(BidPrice11)+s(BidVolume11)+s(AskPrice11)+s(AskVolume11)+s(LastPrice2)+s(BidPrice12)+s(BidVolume12)+s(AskPrice12)+s(AskVolume12)+s(LastPrice3)+s(Volume3)+s(BidPrice13)+s(BidVolume13)+s(AskPrice13)+s(AskVolume13)+s(LastPrice4)+s(Volume4)+s(Volume5)+s(BidVolume16)+s(AskPrice16)+s(AskVolume16)+s(LastPrice7)+s(Volume7)+s(BidPrice17)+s(BidVolume17)+s(AskPrice17)+s(AskVolume17)+s(Volume9)+s(BidPrice19)+s(BidVolume19)+s(AskPrice19)+s(AskVolume19))
#f1 = as.formula(MidPointMean~s(LastPrice)+s(MidPrice9)+s(Volume)+s(BidPrice1)+s(BidVolume1)+s(AskPrice1)+s(AskVolume1)+s(LastPrice1)+s(Volume1)+s(BidPrice11)+s(BidVolume11)+s(AskPrice11)+s(AskVolume11)+s(LastPrice2)+s(BidPrice12)+s(BidVolume12)+s(AskPrice12)+s(AskVolume12)+s(LastPrice3)+s(Volume3)+s(BidPrice13)+s(BidVolume13)+s(AskPrice13)+s(AskVolume13)+s(LastPrice4)+s(Volume4)+s(Volume5)+s(BidVolume16)+s(AskPrice16)+s(AskVolume16)+s(LastPrice7)+s(Volume7)+s(BidPrice17)+s(BidVolume17)+s(AskPrice17)+s(AskVolume17)+s(Volume9)+s(BidPrice19)+s(BidVolume19)+s(AskPrice19)+s(AskVolume19))
gamPrice = gam(f0,data = conbine.df)
summary(gamPrice)
plot(gamPrice,se=T,resid=T,pch=16)

# install.packages("deepnet")
# library(deepnet)
# dpPrice = nn.train(conbine.df[,2:71], conbine.df[,1], hidden=c(50,20), activationfun="sigm", learningrate=0.8, momentum=0.5, learningrate_scale=1, output="sigm", numepochs=3, batchsize=100, hidden_dropout=0, visible_dropout=0)
# 
# install.packages("darch")
# library(deepnet)
# 
# install.packages("elmNN")
# library(elmNN)
# 
# install.packages("forecast")
# install.packages("tseries")
# library(zoo)
# library(forecast)
# library(tseries)
# 
# #install.packages("ggthemes")
# library(keras)
# library(ggplot2)
# library(ggthemes)
# library(lubridate)
# set.seed(7)





predict.df = read.table("C:/Users/simon_zsy/Desktop/test_data.csv",sep=",",header=TRUE)
predict.df = select(predict.df, MidPrice, LastPrice, Volume, BidPrice1, BidVolume1, AskPrice1, AskVolume1)
X = c(0:9999)
X = data.frame(X = X)
predict.df = bind_cols(X, predict.df)
d0 = filter(predict.df, X %% 10 == 0)
d1 = filter(predict.df, X %% 10 == 1)
d2 = filter(predict.df, X %% 10 == 2)
d3 = filter(predict.df, X %% 10 == 3)
d4 = filter(predict.df, X %% 10 == 4)
d5 = filter(predict.df, X %% 10 == 5)
d6 = filter(predict.df, X %% 10 == 6)
d7 = filter(predict.df, X %% 10 == 7)
d8 = filter(predict.df, X %% 10 == 8)
d9 = filter(predict.df, X %% 10 == 9)
predict.df = bind_cols(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)
predict.df = select(predict.df, -starts_with("X"))
#predict.df = select(predict.df, -starts_with("MidPrice"))
#predict.df = select(predict.df, -starts_with("LastPrice"))

# answer = predict(glmPrice, predict.df)
# answer = answer[-(1:142)]
# 
# answer = predict(lmPrice, predict.df)
# answer = answer[-(1:142)]
# 
# small = predict.df[20,]
# detach("package:dplyr")
# maxs = apply(predict.df, 2, max)
# mins = apply(predict.df, 2, min)
# pd = as.data.frame(scale(predict.df, center = mins, scale = maxs - mins))
# answer = compute(nePrice, pd)
# answer = answer$net.result*(max(conbine.df$MidPointMean)-min(conbine.df$MidPointMean))+min(conbine.df$MidPointMean)
# answer = answer[143:1000,]
# 
# answer = predict(rpPrice, predict.df)
# answer = answer[-(1:142)]
# 
# answer = predict(rwPrice, predict.df)
# answer = answer[-(1:142)]
# 
# answer = predict(rsPrice, predict.df[,1:70])
# answer = answer[-(1:142)]
# 
# maxs = apply(predict.df, 2, max)
# mins = apply(predict.df, 2, min)
# pd = as.data.frame(scale(predict.df, center = mins, scale = maxs - mins))
# answer = sim(amPrice$net, pd[,1:70])
# answer = answer*(max(conbine.df$MidPointMean)-min(conbine.df$MidPointMean))+min(conbine.df$MidPointMean)
# answer = answer[-(1:142)]
# 
# maxs = apply(predict.df, 2, max)
# mins = apply(predict.df, 2, min)
# pd = as.data.frame(scale(predict.df, center = mins, scale = maxs - mins))
# answer = predict(nnPrice, pd)
# answer = answer*(max(conbine.df$MidPointMean)-min(conbine.df$MidPointMean))+min(conbine.df$MidPointMean)
# answer = answer[-(1:142)]
# 
# answer = predict(svmPrice, predict.df)
# answer = answer[-(1:142)]
# 
# answer = predict(eaPrice, predict.df)
# answer = answer[-(1:142)]

answer = predict(gamPrice, predict.df)
answer = answer[-(1:142)]

# answer = predict(gnPrice, as.matrix(predict.df))
# answer = answer[-(1:142)]

#answer = (answer1 + answer2)/2

final = data.frame(caseid = c(143:1000), midprice = answer)

write.csv(final,"C:/Users/simon_zsy/Desktop/answer.csv",row.names = FALSE)

