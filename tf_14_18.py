#this program will be for using tensorflow for the 2014-2017  seasons in part 1 of the competition
#also test ridge regression

#import tensorflow
import pandas as pd
import numpy as np
import gc

from sklearn.linear_model import LogisticRegression

train_all_df = pd.read_csv('input\\training_data.csv')

train_y_df = train_all_df['Result']

train_x_df = train_all_df.drop(['Unnamed: 0', 'full_conf', 'TeamID', 'Opp_ID', 'Opp_full_conf', 'Season', 'Score',\
                                'Result', 'Opp_Score', 'Opp_Conf', 'DayNum', 'Conf'], axis=1)

clf_log = LogisticRegression(C=10, fit_intercept=False)


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