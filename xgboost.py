import xgboost as xgb
from sklearn.metrics import accuracy_score
import os
print(os.getcwd())
dtrain = xgb.DMatrix(r'.\personal stuff\Practice\data\agaricus.txt.train')
dtest = xgb.DMatrix(r'.\personal stuff\Practice\data\agaricus.txt.test')

print(dtrain.num_col(), dtest.num_col())
print(dtrain.num_row(), dtest.num_row())

param = {'max_depth': 2, 'eta': 1, 'silent': 0, 'objective': 'binary:logistic'}
print(param)
num_round = 2
import time
starttime = time.perf_counter()
bst = xgb.train(param, dtrain, num_round)
endtime = time.perf_counter()
print(endtime - starttime)

train_preds = bst.predict(dtrain)
print(train_preds)
train_predictions = [round(value) for value in train_preds]
print(train_predictions)
y_train = dtrain.get_label()
print(y_train)
train_acuraccy = accuracy_score(y_train, train_predictions)
print(train_acuraccy)

preds = bst.predict(dtest)
predictions = [round(value) for value in preds]
y_test = dtest.get_label()
test_accuracy = accuracy_score(y_test, predictions)
print(test_accuracy)

import graphviz
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.figure(figsize = (32, 24))
xgb.plot_tree(bst, num_trees = 0, rankdir = 'LR')
plt.show()

#sklearn接口
from xgboost import XGBClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

my_workpath = r'.\personal stuff\Practice\data'

X_train, y_train = load_svmlight_file(my_workpath + r'\agaricus.txt.train')
X_test, y_test = load_svmlight_file(my_workpath +r'\agaricus.txt.test')
print(X_train.shape)
print(X_test.shape)

num_round = 2
bst = XGBClassifier(
    max_depth = 2, learning_rate = 1, n_estimators = num_round, 
    silent = True, objective = 'binary:logistic')
bst.fit(X_train, y_train)

train_preds = bst.predict(X_train)
train_predictions = [round(value) for value in train_preds]
train_accuracy = accuracy_score(y_train, train_predictions)
print(train_accuracy)

preds = bst.predict(X_test)
predictions = [round(value) for value in preds]
test_accuracy = accuracy_score(y_test, predictions)
print(test_accuracy)

#使用sklearn的cross_val_score接口
from xgboost import XGBClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

my_workpath = r'.\personal stuff\Practice\data'
X_train, y_train = load_svmlight_file(my_workpath + r'\agaricus.txt.train')
X_test, y_test = load_svmlight_file(my_workpath + r'\agaricus.txt.test')
print(X_train.shape)
print(X_test.shape)

num_round = 2
param = {'max_depth': 2, 'eta': 1, 'silent': 0, 'n_estimators': num_round, 
        'learning_rate': 0.1,'objective': 'binary:logistic'}
print(param)
bst = XGBClassifier(**param)

kfold = StratifiedKFold(n_splits = 10, random_state = 7)
results = cross_val_score(bst, X_train, y_train, cv = kfold)
print(results)
print(results.mean())

#使用sklearn的GridSearchcv接口
from xgboost import XGBClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

my_workpath = r'.\personal stuff\Practice\data'
X_train, y_train =load_svmlight_file(my_workpath + r'\agaricus.txt.train')
X_test, y_test = load_svmlight_file(my_workpath +r'\agaricus.txt.test')
print(X_train.shape)
print(X_test.shape)

params = {'max_depth': 2, 'eta': 1, 'silent':0,
        'learning_rate': 0.1, 'objective': 'binary:logistic'}

bst = XGBClassifier(**params)
param_set = {'n_estimators': range(1, 51, 1)}
clf = GridSearchCV(estimator = bst, param_grid = param_set, scoring = 'accuracy', cv = 5)
clf.fit(X_train, y_train)
print(clf.cv_results_, clf.best_params_, clf.best_score_)

preds = clf.predict(X_test)
predictions = [round(value) for value in preds]
test_accuracy = accuracy_score(y_test, predictions)
print(test_accuracy)

#使用sklearn的early_stopping_rounds参数
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns

my_workpath = r'.\personal stuff\Practice\data'
X_train, y_train = load_svmlight_file(my_workpath + r'\agaricus.txt.train')
X_test, y_test = load_svmlight_file(my_workpath + r'\agaricus.txt.test')
print(X_train.shape, X_test.shape)

seed = 7
test_size = 0.33
X_train_part, X_validate, y_train_part, y_validate = train_test_split(X_train, y_train, 
                                                test_size = test_size, random_state = seed)
num_round = 100
param = {'n_estimators': num_round, 'max_depth': 2, 
        'learning_rate': 0.1, 'silent': 0, 'objective': 'binary:logistic'}
bst = XGBClassifier(**param)
eval_set = [(X_validate, y_validate)]
bst.fit(X_train_part, y_train_part, early_stopping_rounds = 10, 
        eval_metric = 'error', eval_set = eval_set, verbose = True)

results = bst.evals_result()
print(results)
epochs = len(results['validation_0']['error'])
print(epochs)
sns.lineplot(x = range(epochs), y = results['validation_0']['error'])
plt.title('XGBoost Early Stop')
plt.ylabel('Error')
plt.xlabel('epochs')
plt.legend('Test')

preds = bst.predict(X_test)
predictions = [round(value) for value in preds]
test_accuracy = accuracy_score(y_test, predictions)
print(test_accuracy)

#sklearn总例子
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns

my_workpath = r'.\personal stuff\Practice\data'
X_train, y_train = load_svmlight_file(my_workpath + r'\agaricus.txt.train')
X_test, y_test = load_svmlight_file(my_workpath + r'\agaricus.txt.test')
print(X_train.shape, X_test.shape)

seed = 7
test_size = 0.33
X_train_part, X_validate, y_train_part, y_validate = train_test_split(X_train, y_train,      
            test_size = test_size, random_state = seed)
print(X_train_part.shape, X_validate.shape)
num_round = 100
params = {'max_depth': 2, 'learning_rate': 0.1, 'n_estimators': num_round, 'silent': False,
        'objective': 'binary:logistic'}
bst = XGBClassifier(**params)
eval_set = [(X_train_part, y_train_part), (X_validate, y_validate)]
bst.fit(X_train_part, y_train_part, eval_metric = ['error', 'logloss'], eval_set = eval_set, verbose = True)

results = bst.evals_result()
epochs = len(results['validation_0']['logloss'])
sns.lineplot(x = range(epochs), y = results['validation_0']['logloss'])
plt.legend('validation_0')
sns.lineplot(x = range(epochs), y = results['validation_1']['logloss'])
plt.legend('validation_1')
plt.title('XGBoost Logloss')
plt.show()

sns.lineplot(x = range(epochs), y = results['validation_0']['error'])
plt.legend('validation_0')
sns.lineplot(x = range(epochs), y = results['validation_1']['error'])
plt.legend('validation_1')
plt.title('XGBoost Error')
plt.show()

preds = bst.predict(X_test)
prediction = [round(value) for value in preds]
test_accuracy = accuracy_score(y_test, prediction)
print(test_accuracy)

#Xgboost的交叉验证
import numpy as np
import xgboost as xgb
import os
print(os.getcwd())
import pandas as pd
train = pd.read_csv(r'.\personal stuff\Practice\Xgboost使用\Lect3_code\data\higgsboson_training.csv', 
    na_values = -999.0, converters = {'Label': lambda x: int(x == 's')})
data = train.iloc[:, 1: 31]
dtrain = xgb.DMatrix(data, label = train.Label, missing = -999.0, weight = train.Weight)
param = {'objective': 'binary:logitraw', 'eta': 0.1, 'max_depth': 6, 'silent': 0, }

def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0) / np.sum(label == 1))
    param['scale_pos_weight'] = ratio
    wtrain = dtrain.get_weight()
    wtest = dtest.get_weight()
    sum_weight = sum(wtrain) + sum(wtest)
    wtrain *= sum_weight / sum(wtrain)
    wtest *= sum_weight / sum(wtest)
    dtrain.set_weight(wtrain)
    dtest.set_weight(wtest)
    return (dtrain, dtest, param)