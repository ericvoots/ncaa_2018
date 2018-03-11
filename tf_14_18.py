#this program will be for using tensorflow for the 2014-2017  seasons in part 1 of the competition
#also test log regression
from sklearn import model_selection

#import tensorflow
import pandas as pd
import numpy as np
import gc

from sklearn.linear_model import LogisticRegressionCV
from sklearn.utils import shuffle

from sklearn.ensemble import RandomForestClassifier

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold


train_all_df = pd.read_csv('input\\training_data.csv')
train_all_df = shuffle(train_all_df)

os_df = train_all_df.loc[train_all_df['Season'] > 2016]
train_all_df = train_all_df.loc[train_all_df['Season'] < 2017]

train_y_df = train_all_df['Result']
y_os_df = os_df['Result']

train_x_df = train_all_df.drop(['Unnamed: 0', 'full_conf', 'TeamID', 'Opp_ID', 'Opp_full_conf', 'Season', 'Score',\
                                'Result', 'Opp_Score', 'Opp_Conf', 'DayNum', 'Conf', 'NumOT'], axis=1)

x_os_df = os_df.drop(['Unnamed: 0', 'full_conf', 'TeamID', 'Opp_ID', 'Opp_full_conf', 'Season', 'Score',\
                                'Result', 'Opp_Score', 'Opp_Conf', 'DayNum', 'Conf', 'NumOT'], axis=1)

del train_all_df, os_df
gc.collect()

clf_log = LogisticRegressionCV(fit_intercept=False)


clf_log.fit(train_x_df, train_y_df)

print('R^2 of the logistic regression', clf_log.score(train_x_df, train_y_df))

log_pred = clf_log.predict(train_x_df)

log_pred = pd.DataFrame(log_pred)

log_pred.to_csv('input\\log_preds.csv')

column_list = train_x_df.columns


column_list = pd.DataFrame(column_list, columns=['Feature'])
coef = clf_log.coef_
coef = pd.DataFrame(coef.reshape(-1, len(coef)), columns=['Coefficients'])

features_log_df = pd.merge(column_list, coef, left_index=True, right_index=True)

del column_list, coef
gc.collect()
print(features_log_df)

#cross validation of log regression

num_samples = 100
test_size = .80
num_instances = len(train_x_df)
seed = 4
kfold = model_selection.ShuffleSplit(n_splits=5, test_size=test_size, random_state=seed)
model = LogisticRegressionCV()
results = model_selection.cross_val_score(model, train_x_df, train_y_df, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))

#out of sample log - but log over fitting on probabilities

print('out of sample r^2 score', clf_log.score(x_os_df, y_os_df))


log_prob_df = clf_log.predict_proba(x_os_df)[:, 1]
log_prob_df = pd.DataFrame(log_prob_df, columns=['probability'])
log_prob_df.to_csv('submissions\\log_probabilities.csv')

rclf = RandomForestClassifier(n_estimators=25, min_samples_split=60, min_samples_leaf=30)
rclf_fit = rclf.fit(train_x_df, train_y_df)
rclf_score = rclf.score(train_x_df, train_y_df)
rclf_predict = rclf.predict_proba(train_x_df)[:,1]
rclf_predict_df = pd.DataFrame(rclf_predict, columns=['rf_proba'])

print(rclf_predict_df)
print(rclf_score)

#tensorflow

model = Sequential()
model.add(Dense(120, input_dim=train_x_df.shape[1], activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(train_x_df.values, train_y_df.values, epochs=150, batch_size=10)
# evaluate the model
scores = model.evaluate(train_x_df.values, train_y_df.values)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


