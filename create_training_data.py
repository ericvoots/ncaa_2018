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

details2_df[['WSeed', 'LSeed']] = details2_df[['WSeed', 'LSeed']].astype(float)

del t_seeds_df, details_df
gc.collect()

#get conference of the team

#winning teams conference
details3_df = pd.merge(details2_df, teams_conf_df, how='left', left_on=['WTeamID', 'Season'], right_on=['TeamID', 'Season'])
details3_df = details3_df.drop('TeamID', axis=1)
details3_df = details3_df.rename(columns={'ConfAbbrev': 'WConf'})

#losing team conference
details3_df = pd.merge(details3_df, teams_conf_df, how='left', left_on=['LTeamID', 'Season'], right_on=['TeamID', 'Season'])
details3_df = details3_df.drop('TeamID', axis=1)
details3_df = details3_df.rename(columns={'ConfAbbrev': 'LConf'})

del teams_conf_df, details2_df
gc.collect()

#get seed matching up information - play in games with two 16s will be a 0% due to data but its 16 seed so...
details4_df = pd.merge(details3_df, seed_hist_df, how='left', left_on=['WSeed', 'LSeed'], right_on=['Seed1', 'Seed2'])
details4_df = details4_df.drop(['Wins', 'Losses', 'Seed1', 'Seed2'], axis=1)

del details3_df, seed_hist_df
gc.collect()

print(details4_df.head(25))



