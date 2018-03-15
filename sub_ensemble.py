from sklearn import model_selection

#import tensorflow
import pandas as pd
import numpy as np
import gc
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegressionCV, Ridge
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
#from keras.models import Sequential
#from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import xgboost
from catboost import CatBoostClassifier

from sklearn.utils import shuffle

train_all_df = pd.read_csv('input\\training_data.csv')

#test_df = train_all_df.loc[train_all_df['Season'] > 2017]
train_all_df = train_all_df.loc[train_all_df['Season'] <= 2017]

train_y_df = train_all_df['Result']
#y_test_df = test_df['Result']
print(train_y_df.head(5))
train_x_df = train_all_df.drop(['full_conf', 'Conf', 'TeamID', 'TeamID_Opp', 'Season', 'Result', 'full_conf_Opp',\
                                'Conf_Opp', 'seed_hist_pct', 'seed_hist_pct_Opp'], axis=1)
train_x_df = train_x_df[['Seed', 'Seed_Opp', 'seed_diff', 'Conf_Rank', 'Conf_Rank_Opp', 'conf_diff']]
#x_test_df = test_df.drop(['full_conf', 'Conf', 'TeamID', 'TeamID_Opp', 'Season', 'Result', 'full_conf_Opp', 'Conf_Opp'], axis=1)

print('training columns, post drop\n', train_x_df.columns)
del train_all_df
gc.collect()

clf_log = LogisticRegressionCV(fit_intercept=False, solver='newton-cg')


clf_log.fit(train_x_df, train_y_df)

print('R^2 of the logistic regression', clf_log.score(train_x_df, train_y_df))

log_pred = clf_log.predict(train_x_df)

log_pred = pd.DataFrame(log_pred)

log_pred.to_csv('input\\log_preds_train.csv')

column_list = train_x_df.columns


column_list = pd.DataFrame(column_list, columns=['Feature'])
coef = clf_log.coef_
coef = pd.DataFrame(coef.reshape(-1, len(coef)), columns=['Coefficients'])

features_log_df = pd.merge(column_list, coef, left_index=True, right_index=True)
print('coefficients log\n', features_log_df)
del column_list, coef
gc.collect()
#print(features_log_df)

#cross validation of log regression

num_samples = 100
test_size = .80
num_instances = len(train_x_df)
seed = 4
kfold = model_selection.ShuffleSplit(n_splits=5, test_size=test_size, random_state=seed)

results = model_selection.cross_val_score(clf_log, train_x_df, train_y_df, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))

#out of sample log - but log over fitting on probabilities


#log_prob_df = clf_log.predict_proba(x_test_df)[:, 1]
#log_prob_df = pd.DataFrame(log_prob_df, columns=['probability'])
#log_prob_df.to_csv('submissions\\log_probabilities.csv')

rclf = RandomForestClassifier(n_estimators=25, min_samples_split=60, min_samples_leaf=30, max_depth=5)
rclf_fit = rclf.fit(train_x_df, train_y_df)
rclf_score = rclf.score(train_x_df, train_y_df)
rclf_predict_df = rclf.predict_proba(train_x_df)[:,1]
rclf_predict_df = pd.DataFrame(rclf_predict_df, columns=['rf_proba'])
rclf_pred = rclf.predict(train_x_df)
rclf_pred = pd.DataFrame(rclf_pred)
rclf_pred.to_csv('input\\rclf_preds_train.csv')


#tensorflow - shows signs of overfitting - try norm - still 0 or 1 similar to log
'''
scaler = MinMaxScaler()
train_x_df_norm = scaler.fit_transform(train_x_df)

model = Sequential()
model.add(Dense(512, input_dim=train_x_df_norm.shape[1], activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='hard_sigmoid'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(train_x_df_norm, train_y_df, epochs=150, batch_size=25)
# evaluate the model
scores = model.evaluate(train_x_df_norm, train_y_df.values)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

tf_pred_df = model.predict(train_x_df_norm)
tf_pred_df = pd.DataFrame(tf_pred_df)
tf_pred_df.to_csv('submissions\\tf_pred_train.csv')
'''
#xgboost and stratified kfold, best results around 5 depth, 50 weight

model_xgb = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=50, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, seed=27)

model_xgb.fit(train_x_df, train_y_df)

kfold = StratifiedKFold(n_splits=10, random_state=7)
results = cross_val_score(model_xgb, train_x_df, train_y_df, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#results_os = model_xgb.score(x_test_df, y_test_df)
#print("Accuracy: %.2f%%" % (results_os.mean()*100))

xgb_pred_df = model_xgb.predict_proba(train_x_df)
xgb_pred_df = pd.DataFrame(xgb_pred_df)

#lightgbm, using to dart to change type of boosting, similar issue as tf and log regression all close to or 1, i hate light gbm anyways


params = {"objective": "binary",
          "boosting_type": "dart",
          "learning_rate": 0.1,
          "num_leaves": 31,
          "max_bin": 256,
          "feature_fraction": 0.8,
          "verbosity": 0,
          "min_data_in_leaf": 10,
          "min_child_samples": 10,
          "subsample": 0.8
          }
X_train, X_val, y_train, y_val = train_test_split(train_x_df, train_y_df, test_size=0.1, random_state=0)
dtrain = lgb.Dataset(X_train, y_train)
dvalid = lgb.Dataset(X_val, y_val, reference=dtrain)
bst = lgb.train(params, dtrain, 1000, valid_sets=dvalid, verbose_eval=50)

test_pred = bst.predict(
    train_x_df, num_iteration=bst.best_iteration)

print(test_pred)

'''
#catboost
X_train, X_validation, y_train, y_validation = train_test_split(train_x_df, train_y_df, train_size=0.7, random_state=1234)
clf_cat = CatBoostClassifier(iterations=10, depth=5, learning_rate=0.1, loss_function='Logloss')
clf_cat.fit(X_train, y_train, eval_set=(X_validation, y_validation), plot=True)

#knn



#pca - rf



#ica - rf





#combine predictions together with submission format



#control

'''