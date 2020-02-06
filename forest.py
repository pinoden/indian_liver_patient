import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from data_load import load_csv
from sklearn import preprocessing

test = load_csv('test')
train = load_csv('train')
# y = LabelBinarizer().fit_transform(y)
lb = preprocessing.LabelBinarizer()

train.Sex = lb.fit_transform(train.Sex)
# print(train.Sex)
# print(train.shape, train.head())

features = ['Pclass','Sex','SibSp','Parch','SibSp','Parch']
# Create target object and call it y
y = train.Survived
# Create X
X = train[features]
print(X.shape)
X = X.dropna()
print(X.shape)
# print('before',X.shape)
# print('after',X.shape)

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y,test_size=0.1, random_state=1)
# print(val_y.loc[val_y == 0])
# Specify Model
forest_model = RandomForestRegressor(random_state=2)
forest_model.fit(train_X, train_y)
# temp_res = pd.DataFrame({'PassengerId': val_X.PassengerId, 'Results': tit_pred[:]})
# print(temp_res.shape)
# temp_res = temp_res.merge(train, how='left', on='PassengerId')
# # stats = stats.merge(clust, on='address_id', how='left')
# print(temp_res.shape)
# trsh = 3
for trsh in range(60,90):
    tit_pred = forest_model.predict(val_X)
    tit_pred[tit_pred >= trsh/100] = 1
    tit_pred[tit_pred < trsh/100] = 0

    temp_res = pd.DataFrame({'Survived': val_y, 'Results': tit_pred.astype(int)})
    temp_res['error'] = temp_res['Survived']^temp_res['Results']
    # print(temp_res.shape)
    print(trsh,temp_res.error.sum())
    # print(temp_res.loc[(temp_res.Survived == 0) & (temp_res.Results == 0)])



# Create test_X
# test_X = test[features]
# final_pred = forest_model.predict(test_X)
# trsh = 0.4
#
# final_pred[final_pred >= trsh] = 1
# final_pred[final_pred < trsh] = 0
#
# # Exportind the results
# results = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': final_pred[:]})
# results.to_csv('data/results.csv',index = None)