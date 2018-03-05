#this program will be for using tensorflow for the 2014-2017  seasons in part 1 of the competition

#import tensorflow
import pandas as pd
import numpy as np

t_seeds_df = pd.read_csv('input\\datafiles\\NCAATourneySeeds.csv')

details_df = pd.read_csv('input\\datafiles\\NCAATourneyDetailedResults.csv')

teams_conf_df = pd.read_csv('input\\datafiles\\TeamConferences.csv')

print(details_df.head(5))
