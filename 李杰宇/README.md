### Requirements
+ python == 3.6.3
+ torch == 0.4.1
+ gensim == 3.6.0
+ xgboost == 0.8.1
+ sklearn == 0.0
+ numpy == 1.13.3
+ tensorboardX == 1.5
+ tensorflow == 1.7.0
+ minepy == 1.2.2
+ nltk == 3.3

### Training
Firstly, you need to choose parameters in config.py.
And run trainer.train() in train.py.
+ model1
  - task_type : R
  - use_attention : False
  - use_RNN : True
  - you don't neet to change the other parameters
  - result on public leaderboard is 0.00149
+ model2
  - task_type : C
  - use_attention : True
  - you don't neet to change the other parameters
  - result on public leaderboard is 0.00150

### Test
You only need to run trainer.Diagenerate(n) in train.py.
n is the index of the parameters.
The result will be stored in the prediction.csv.
