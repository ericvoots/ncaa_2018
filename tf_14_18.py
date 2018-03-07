#this program will be for using tensorflow for the 2014-2017  seasons in part 1 of the competition
#also test ridge regression

#import tensorflow
import pandas as pd
import numpy as np
import gc


train_all_df = pd.read_csv('input\\training_data.csv')

train_y_df = train_all_df['Result']

train_x_df = train_all_df.drop(['Unnamed: 0', 'full_conf', 'TeamID', 'Opp_ID', 'Opp_full_conf'], axis=1)

print(train_all_df.columns)

print(train_x_df.columns)