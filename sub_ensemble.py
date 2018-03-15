from sklearn import model_selection

#import tensorflow
import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV, Ridge
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import xgboost
from sklearn.decomposition import IncrementalPCA
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from functools import reduce

train_all_df = pd.read_csv('input\\training_data.csv')
test_df = pd.read_csv('input\\test_data.csv', delimiter=';')
#test_df = train_all_df.loc[train_all_df['Season'] > 2017]
train_y_df = train_all_df['Result']
#y_test_df = test_df['Result']
print(train_y_df.head(5))
train_x_df = train_all_df.drop(['full_conf', 'Conf', 'TeamID', 'TeamID_Opp', 'Season', 'Result', 'full_conf_Opp',\
                                'Conf_Opp', 'seed_hist_pct', 'seed_hist_pct_Opp'], axis=1)
print('test_df\n', test_df.head(5))
test_x_df = test_df.drop(['Conf', 'TeamID', 'TeamID_Opp', 'Season', 'Conf_Opp', 'ID'], axis=1)

#x_test_df = test_df.drop(['full_conf', 'Conf', 'TeamID', 'TeamID_Opp', 'Season', 'Result', 'full_conf_Opp', 'Conf_Opp'], axis=1)

print('training columns, post drop\n', train_x_df.columns)
del train_all_df
gc.collect()


clf_log = LogisticRegressionCV(fit_intercept=False, solver='newton-cg')


clf_log.fit(train_x_df, train_y_df)
log_t_pred = clf_log.predict_proba(test_x_df)[:, 1]
log_t_pred = pd.DataFrame(log_t_pred, columns=['log_pred'])

print('R^2 of the logistic regression', clf_log.score(train_x_df, train_y_df))

log_t_pred.to_csv('input\\log_preds.csv')

column_list = train_x_df.columns

column_list = pd.DataFrame(column_list, columns=['Feature'])
coef = clf_log.coef_
coef = pd.DataFrame(coef.reshape(-1, len(coef)), columns=['Coefficients'])

features_log_df = pd.merge(column_list, coef, left_index=True, right_index=True)
#print('coefficients log\n', features_log_df)
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
rclf_pred = rclf.predict_proba(test_x_df)[:, 1]
rclf_pred = pd.DataFrame(rclf_pred, columns=['rf_pred'])
rclf_pred.to_csv('input\\rclf_preds.csv')


#tensorflow - shows signs of overfitting - try norm - still 0 or 1 similar to log
#tensorflow crashes with extra fields, lowering number

train_x_df_min = train_x_df[['Seed', 'Seed_Opp', 'seed_diff', 'Conf_Rank', 'Conf_Rank_Opp', 'conf_diff']]

test_x_df_min = test_x_df[['Seed', 'Seed_Opp', 'seed_diff', 'Conf_Rank', 'Conf_Rank_Opp', 'conf_diff']]

model = Sequential()
model.add(Dense(1024, input_dim=train_x_df_min.shape[1], activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='hard_sigmoid'))
model.add(Dense(128, activation='hard_sigmoid'))
model.add(Dense(64, activation='hard_sigmoid'))
model.add(Dense(1, activation='hard_sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(train_x_df_min.values, train_y_df.values, epochs=250, batch_size=200)
# evaluate the model
scores = model.evaluate(train_x_df_min.values, train_y_df.values)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

tf_pred_df = model.predict(test_x_df_min.values)
tf_pred_df = pd.DataFrame(tf_pred_df, columns=['tf_pred'])
tf_pred_df.to_csv('submissions\\tf_pred.csv')

#xgboost and stratified kfold, best results around 5 depth, 50 weight

model_xgb = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,
 min_child_weight=50, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, seed=27)

model_xgb.fit(train_x_df, train_y_df)

kfold = StratifiedKFold(n_splits=10, random_state=7)
results = cross_val_score(model_xgb, train_x_df, train_y_df, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#results_os = model_xgb.score(x_test_df, y_test_df)
#print("Accuracy: %.2f%%" % (results_os.mean()*100))
test_x_df = test_x_df[train_x_df.columns]
xgb_pred_df = model_xgb.predict_proba(test_x_df)[:, 1]

xgb_pred_df = pd.DataFrame(xgb_pred_df, columns=['xgb_pred'])

#lightgbm

params = {"objective": "binary",
          "boosting_type": "dart",
          "learning_rate": 0.1,
          "num_leaves": 31,
          "max_depth": 4,
          "max_bin": 256,
          "feature_fraction": 0.8,
          "verbosity": 0,
          "min_data_in_leaf": 90,
          "min_child_samples": 90,
          "subsample": 0.8
          }
X_train, X_val, y_train, y_val = train_test_split(train_x_df, train_y_df, test_size=0.1, random_state=0)
dtrain = lgb.Dataset(X_train, y_train)
dvalid = lgb.Dataset(X_val, y_val, reference=dtrain)
bst = lgb.train(params, dtrain, 1000, valid_sets=dvalid, verbose_eval=50)

test_pred_gbm = bst.predict(
    test_x_df, num_iteration=bst.best_iteration)

test_pred_gbm = pd.DataFrame(test_pred_gbm, columns=['gbm_pred'])

#knn
clf_knn = KNeighborsClassifier(n_neighbors=50)
clf_knn.fit(train_x_df, train_y_df)

print('\nKnn score\n', clf_knn.score(train_x_df, train_y_df))

knn_pred = clf_knn.predict_proba(test_x_df)[:, 1]
knn_pred = pd.DataFrame(knn_pred, columns=['knn_pred'])

#pca - rf
ipca = IncrementalPCA(n_components=7)
ipca.fit(train_x_df)
print(ipca.explained_variance_ratio_)

ipca_train_x_df = ipca.transform(train_x_df)
ipca_test_x_df = ipca.transform(test_x_df)

#log ipca
clf_log.fit(ipca_train_x_df, train_y_df)
print('ipca score', clf_log.score(ipca_train_x_df, train_y_df))

log_ica_pred = clf_log.predict_proba(ipca_test_x_df)[:, 1]
log_ica_pred = pd.DataFrame(log_ica_pred, columns=['log_ica_pred'])

#ica - rf - only doing these two due to time constraints

rclf.fit(ipca_train_x_df, train_y_df)
rclf_ica_pred = rclf.predict_proba(ipca_test_x_df)[:, 1]
rclf_ica_pred = pd.DataFrame(rclf_ica_pred, columns=['rclf_pred_ica'])

df_final = test_df.join(log_t_pred).join(log_ica_pred).join(rclf_pred).join(rclf_ica_pred).join(xgb_pred_df).join(knn_pred).join(tf_pred_df)
df_final['Wins_pred'] = df_final[(df_final.log_pred >= 0.5) & (df_final.log_ica_pred >= 0.5) & (df_final.rf_pred >= 0.5) &\
                            (df_final.rclf_pred_ica >= 0.5) & (df_final.xgb_pred >= 0.5) & (df_final.tf_pred >= 0.5)&\
                            (df_final.knn_pred >= 0.5)].count()

df_final['Loss_pred'] = 7 - df_final['Wins_pred']
df_final['Avg_pred'] = df_final[['log_pred', 'log_ica_pred', 'rf_pred', 'rclf_pred_ica', 'xgb_pred', 'tf_pred',\
                                 'knn_pred']].mean(axis=1)
#after 4 seeds no more correction - correction didn't work damn it manual fix last minute 1 am
df_final['Avg_pred'] = df_final.loc[(df_final.Seed == 1) & (df_final.Seed_Opp == 16), 'Avg_pred'] = 1
df_final['Avg_pred'] = df_final.loc[(df_final.Seed == 16) & (df_final.Seed_Opp == 1), 'Avg_pred'] = 0
df_final['Avg_pred'] = df_final.loc[(df_final.Seed == 2) & (df_final.Seed_Opp == 15), 'Avg_pred'] = 1
df_final['Avg_pred'] = df_final.loc[(df_final.Seed == 15) & (df_final.Seed_Opp == 2), 'Avg_pred'] = 0
df_final['Avg_pred'] = df_final.loc[(df_final.Seed == 3) & (df_final.Seed_Opp == 14) & (df_final.Avg_pred > 0.5), 'Avg_pred'] = 1
df_final['Avg_pred'] = df_final.loc[(df_final.Seed == 14) & (df_final.Seed_Opp == 3) & (df_final.Avg_pred < 0.5), 'Avg_pred'] = 0
df_final['Avg_pred'] = df_final.loc[(df_final.Seed == 4) & (df_final.Seed_Opp == 13) & (df_final.Avg_pred > 0.5), 'Avg_pred'] = 1
df_final['Avg_pred'] = df_final.loc[(df_final.Seed == 13) & (df_final.Seed_Opp == 4) & (df_final.Avg_pred < 0.5), 'Avg_pred'] = 0


df_final.to_csv('input\\final_with_subs.csv')

#combine predictions together with submission format



#control

