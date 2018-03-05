#this program will be for using tensorflow for the 2014-2017  seasons in part 1 of the competition

#import tensorflow
import pandas as pd
import numpy as np
import gc

t_seeds_df = pd.read_csv('input\\datafiles\\NCAATourneySeeds.csv')

details_df = pd.read_csv('input\\datafiles\\NCAATourneyDetailedResults.csv')

teams_conf_df = pd.read_csv('input\\datafiles\\TeamConferences.csv')

seed_hist_df = pd.read_csv('input\\seed_historical.csv')

conf_rank_df = pd.read_csv('input\\conference_rank.csv', delimiter=';')

#need to join all data to details

#get winning team seed information
details2_df = pd.merge(details_df, t_seeds_df, how='left', left_on=['WTeamID', 'Season'], right_on=['TeamID', 'Season'])
details2_df = details2_df.drop('TeamID', axis=1)
details2_df = details2_df.rename(columns={'Seed': 'WSeed'})
details2_df['WSeed'] = details2_df['WSeed'].str.lstrip('YXWZba').str.rstrip('YXWbZba')

#lossing team seed information
details2_df = pd.merge(details2_df, t_seeds_df, how='left', left_on=['LTeamID', 'Season'], right_on=['TeamID', 'Season'])
details2_df = details2_df.drop('TeamID', axis=1)
details2_df = details2_df.rename(columns={'Seed': 'LSeed'})
details2_df['LSeed'] = details2_df['LSeed'].str.lstrip('YXWZba').str.rstrip('YXWbZba')

del t_seeds_df, details_df
gc.collect()

print(details2_df.head(25))



