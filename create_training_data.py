#this program will be for using tensorflow for the 2014-2017  seasons in part 1 of the competition

#import tensorflow
import pandas as pd
import numpy as np
import gc
from sklearn.utils import shuffle

#

t_seeds_df = pd.read_csv('input\\datafiles\\NCAATourneySeeds.csv')

details_df = pd.read_csv('input\\datafiles\\NCAATourneyDetailedResults.csv')

teams_conf_df = pd.read_csv('input\\datafiles\\TeamConferences.csv')

seed_hist_df = pd.read_csv('input\\seed_historical.csv')

conf_rank_df = pd.read_csv('input\\conference_rank.csv', delimiter=';')

reg_2018_df = pd.read_csv('input\\datafiles\\RegularSeasonDetailedResults.csv')
matchup_2018_df = pd.read_csv('submissions\\SampleSubmissionStage2_SampleTourney2018.csv')
#regular season 2018
matchup_2018_df['Season'] = matchup_2018_df['ID'].str.split('_').str[0]
matchup_2018_df['TeamID1'] = matchup_2018_df['ID'].str.split('_').str[1]
matchup_2018_df['TeamID2'] = matchup_2018_df['ID'].str.split('_').str[2]

#WScore	LTeamID	LScore	WLoc	NumOT	WFGM	WFGA	WFGM3	WFGA3	WFTM	WFTA	WOR	WDR	WAst	WTO	WStl	WBlk	WPF	LFGM	LFGA	LFGM3	LFGA3	LFTM	LFTA	LOR	LDR	LAst	LTO	LStl	LBlk	LPF
reg_2018_df = reg_2018_df.drop(['LTeamID', 'DayNum'], axis=1)
eos_df_med = reg_2018_df.groupby(['WTeamID', 'Season']).median().add_suffix('_med').reset_index()
eos_df_std = reg_2018_df.groupby(['WTeamID', 'Season']).std().add_suffix('_std').reset_index()
eos_df = pd.merge(left=eos_df_med, right=eos_df_std, how='inner', on=('WTeamID', 'Season'))
eos_df = eos_df.rename(columns={'WTeamID': 'TeamID'})
del reg_2018_df
gc.collect()

print('\neos df\n', eos_df.head(5))

details_df = details_df[['Season', 'WTeamID', 'LTeamID']]

#eos_df['WTeamID'].replace(r'\s+', np.nan, regex=True)

#create season averages



#need to join all data to details



'''
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
details4_df = details4_df.rename(columns={'Win%': 'seed_hist_pct'})
details4_df = pd.merge(details4_df, seed_hist_df, how='left', left_on=['LSeed', 'WSeed'], right_on=['Seed1', 'Seed2'])
details4_df = details4_df.drop(['Wins', 'Losses', 'Seed1', 'Seed2'], axis=1)
details4_df = details4_df.rename(columns={'Win%': 'Opp_seed_hist_pct'})

del details3_df, seed_hist_df
gc.collect()

#get conf_rankings
#get winning teams conference  rank
details5_df = pd.merge(details4_df, conf_rank_df, how='left', left_on=['WConf', 'Season'], right_on=['conference', 'year'])
details5_df = details5_df.drop(['W', 'L', 'PCT', 'Non_Conference_RPI', 'Overall_RPI', 'conference', 'year'], axis=1)
details5_df = details5_df.rename(columns={'Rank': 'WConf_Rank', 'fullconference': 'Wfull_conf'})

#get lossing teams conference rank
details5_df = pd.merge(details5_df, conf_rank_df, how='left', left_on=['LConf', 'Season'], right_on=['conference', 'year'])
details5_df = details5_df.drop(['W', 'L', 'PCT', 'Non_Conference_RPI', 'Overall_RPI', 'conference', 'year'], axis=1)
details5_df = details5_df.rename(columns={'Rank': 'LConf_Rank', 'fullconference': 'Lfull_conf'})

del details4_df, conf_rank_df
gc.collect()

#save intermediate step before creating set for input and output, used for 2018 matching of team data for predictions
#details5_df.to_csv('input\\details_conf_hist.csv')

details5_df = details5_df.drop(['WLoc'],axis=1)
details5_df_temp = details5_df.copy()

#match up losing teams to create 0 Win values
#thx to https://www.kaggle.com/juliaelliott/basic-starter-kernel-ncaa-men-s-dataset for inspiration on next part

#rename W columns first then L columns
df_win = details5_df.rename(columns={'WTeamID': 'TeamID', 'WScore': 'Score', 'WFGM': 'FGM', 'WFGA': 'FGA',\
                                     'WFGM3': 'FGM3', 'WFTM': 'FTM', 'WFTA': 'FTA', 'WOR': 'OR', 'WDR': 'DR', 'WAst':'Ast',\
                                     'WTO': 'TO', 'WFGA3': 'FGA3', 'WStl': 'Stl', 'WBlk': 'Blk', 'WPF': 'PF',\
                                     'WSeed': 'Seed', 'WConf': 'Conf', 'Wfull_conf': 'full_conf', 'WConf_Rank': 'Conf_Rank'})

df_win = df_win.rename(columns={'LTeamID': 'Opp_ID', 'LScore': 'Opp_Score', 'LFGM': 'Opp_FGM', 'LFGA': 'Opp_FGA',\
                                'LFGM3': 'Opp_FGM3', 'LFGA3': 'Opp_FGA3', 'LFTM': 'Opp_FTM', 'LFTA': 'Opp_FTA', 'LOR': \
                                'Opp_OR', 'LDR': 'Opp_DR', 'LAst': 'Opp_Ast', 'LTO': 'Opp_TO', 'LStl': 'Opp_Stl',\
                                'LBlk': 'Opp_Blk', 'LPF': 'Opp_PF', 'LSeed': 'Opp_Seed', 'LConf': 'Opp_Conf',\
                                'Lfull_conf': 'Opp_full_conf', 'LConf_Rank': 'Opp_Conf_Rank'})

df_win['Result'] = 1

df_loss = details5_df_temp.rename(columns={'LTeamID': 'TeamID', 'LScore': 'Score', 'LFGM': 'FGM', 'LFGA': 'FGA',\
                                     'LFGM3': 'FGM3', 'LFTM': 'FTM', 'LFTA': 'FTA', 'LOR': 'OR', 'LDR': 'DR', 'LAst':'Ast',\
                                     'LTO': 'TO', 'LFGA3': 'FGA3', 'LStl': 'Stl', 'LBlk': 'Blk', 'LPF': 'PF',\
                                     'LSeed': 'Seed', 'LConf': 'Conf', 'Lfull_conf': 'full_conf', 'LConf_Rank': 'Conf_Rank'})

df_loss = df_loss.rename(columns={'WTeamID': 'Opp_ID', 'WScore': 'Opp_Score', 'WFGM': 'Opp_FGM', 'WFGA': 'Opp_FGA',\
                                'WFGM3': 'Opp_FGM3', 'WFGA3': 'Opp_FGA3', 'WFTM': 'Opp_FTM', 'WFTA': 'Opp_FTA', 'WOR': \
                                'Opp_OR', 'WDR': 'Opp_DR', 'WAst': 'Opp_Ast', 'WTO': 'Opp_TO', 'WStl': 'Opp_Stl',\
                                'WBlk': 'Opp_Blk', 'WPF': 'Opp_PF', 'WSeed': 'Opp_Seed', 'WConf': 'Opp_Conf',\
                                'Wfull_conf': 'Opp_full_conf', 'WConf_Rank': 'Opp_Conf_Rank'})

df_loss['Result'] = 0

frames = [df_win, df_loss]
del df_win, df_loss
gc.collect()
total_df = pd.concat(frames)
total_df.reset_index()

#next is variable creation

total_df['seed_diff'] = total_df['Seed'] - total_df['Opp_Seed']
total_df['conf_diff'] = total_df['Conf_Rank'] - total_df['Opp_Conf_Rank']
total_df['asst_to'] = total_df['Ast'] / total_df['TO']
total_df['opp_asst_to'] = total_df['Opp_Ast'] / total_df['Opp_TO']

total_df = shuffle(total_df)
#save data to create a model
total_df.to_csv('input\\training_data.csv', index=False)
'''

