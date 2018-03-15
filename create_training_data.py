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


#WScore	LTeamID	LScore	WLoc	NumOT	WFGM	WFGA	WFGM3	WFGA3	WFTM	WFTA	WOR	WDR	WAst	WTO	WStl	WBlk	WPF	LFGM	LFGA	LFGM3	LFGA3	LFTM	LFTA	LOR	LDR	LAst	LTO	LStl	LBlk	LPF
reg_2018_df = reg_2018_df.drop(['LTeamID', 'DayNum'], axis=1)
eos_df_med = reg_2018_df.groupby(['WTeamID', 'Season']).median().add_suffix('_med').reset_index()
#eos_df_std = reg_2018_df.groupby(['WTeamID', 'Season']).std().add_suffix('_std').reset_index()
#eos_df = pd.merge(left=eos_df_med, right=eos_df_std, how='inner', on=('WTeamID', 'Season'))
eos_df_med = eos_df_med.rename(columns={'WTeamID': 'TeamID'})
del reg_2018_df
gc.collect()

details_df = details_df[['Season', 'WTeamID', 'LTeamID']]

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
details4_df = details4_df.rename(columns={'Win%': 'seed_hist_pct_Opp'})

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

details5_df = details5_df.rename(columns={'WTeamID': 'TeamID', 'LTeamID': 'TeamID_Opp', 'WSeed': 'Seed',\
                                          'LSeed': 'Seed_Opp', 'WConf': 'Conf', 'LConf': 'Conf_Opp', 'Wfull_conf':\
                                          'full_conf', 'WConf_Rank': 'Conf_Rank', 'Lfull_conf': 'full_conf_Opp',\
                                          'LConf_Rank': 'Conf_Rank_Opp'})

details5_df_temp = details5_df.copy()

#match up losing teams to create 0 Win values
#thx to https://www.kaggle.com/juliaelliott/basic-starter-kernel-ncaa-men-s-dataset for inspiration on next part

#rename W columns first then L columns
df_win = details5_df

df_win['Result'] = 1

df_loss = details5_df_temp

df_loss['Result'] = 0

del details5_df, details5_df_temp
gc.collect()

#rename L vars to D for defense
eos_df_med = eos_df_med.rename(columns={'LTeamID': 'TeamID_Opp', 'LFGM_med': 'DFGM_med', 'LFGA_med': 'DFGA_med', 'LFGM3_med': 'DFGM3', \
                                        'LFGA3_med': 'DFGA3_med', 'LFTM_med': 'DFTM_med', 'LFTA_med': 'DFTA_med',\
                                        'LOR_med': 'DOR_med', 'LDR_med': 'DDR_med', 'LAst_med': 'DAst_med',\
                                        'LTO_med': 'DTO_med', 'LStl_med': 'DStl_med', 'LBlk_med': 'DBlk_med',\
                                        'LPF_med': 'DPF_med', 'LScore_med': 'DScore_med', 'Lfull_conf': 'full_conf_Opp',\
                                        'LConf': 'Conf_Opp'})

#remove W from variables
eos_df_med = eos_df_med.rename(columns={'WTeamID': 'TeamID', 'WScore_med': 'Score_med', 'WFGM_med': 'FGM_med', 'WFGA_med': 'FGA_med',\
                                        'WFGM3_med': 'FMG3_med', 'WFGA3_med': 'FGA3_med', 'WFTM_med': 'FTM_med',\
                                        'WFTA_med': 'FTA_med', 'WOR_med': 'OR_med', 'WDR_med': 'DR_med',\
                                        'WAst_med': 'Ast_med', 'WTO_med': 'TO_med', 'WStl_med': 'Stl_med',\
                                        'WBlk_med': 'Blk_med', 'WPF_med': 'PF_med'})

eos_df_med = eos_df_med.drop(['NumOT_med'], axis=1)
print('eos df columns\n', eos_df_med.columns)
#issues with matchup 2018 data, running out of time sending to csv to use
eos_2018 = pd.DataFrame()
eos_2018 = eos_df_med.loc[(eos_df_med['Season'] == 2018)]
eos_2018.to_csv("input\\eos_2018.csv", index=False)

eos_df_med_copy = eos_df_med.copy()

final_df_win = pd.merge(left=df_win, right=eos_df_med, how='left', left_on=('Season', 'TeamID'), right_on=('Season', 'TeamID'))
final_df_win = pd.merge(left=final_df_win, right=eos_df_med_copy, how='left', left_on=('Season', 'TeamID_Opp'), right_on=('Season', 'TeamID'), suffixes=('', '_Opp'))
final_df_win = final_df_win.loc[:, ~final_df_win.columns.duplicated()]
final_df_win.reset_index()

del eos_df_med, eos_df_med_copy
gc.collect()

#create test set


final_df_loss = pd.DataFrame()
final_df_loss[['TeamID', 'TeamID_Opp', 'Season']] = final_df_win[['TeamID_Opp', 'TeamID', 'Season']]

final_df_loss[['Seed', 'Conf', 'seed_hist_pct', 'full_conf', 'Conf_Rank', 'Score_med', 'DScore_med', 'FGM_med',\
            'FGA_med', 'FMG3_med', 'FGA3_med', 'FTM_med', 'FTA_med', 'OR_med', 'DR_med', 'Ast_med', 'TO_med',\
            'Stl_med', 'Blk_med', 'PF_med', 'DFGM_med', 'DFGA_med', 'DFGM3', 'DFGA3_med', 'DFTM_med', 'DFTA_med',\
            'DOR_med', 'DDR_med', 'DAst_med', 'DTO_med', 'DStl_med', 'DBlk_med', 'DPF_med',\
            'Seed_Opp', 'Conf_Opp', 'seed_hist_pct_Opp', 'full_conf_Opp', 'Conf_Rank_Opp',\
            'Score_med_Opp', 'DScore_med_Opp', 'FGM_med_Opp', 'FGA_med_Opp', 'FMG3_med_Opp', 'FGA3_med_Opp',\
            'FTM_med_Opp', 'FTA_med_Opp', 'OR_med_Opp', 'DR_med_Opp', 'Ast_med_Opp', 'TO_med_Opp','Stl_med_Opp',\
            'Blk_med_Opp', 'PF_med_Opp', 'DFGM_med_Opp', 'DFGA_med_Opp', 'DFGM3_Opp', 'DFGA3_med_Opp', 'DFTM_med_Opp',\
            'DFTA_med_Opp', 'DOR_med_Opp', 'DDR_med_Opp', 'DAst_med_Opp', 'DTO_med_Opp', 'DStl_med_Opp',\
            'DBlk_med_Opp', 'DPF_med_Opp']] = final_df_win[[\
            'Seed_Opp', 'Conf_Opp', 'seed_hist_pct_Opp', 'full_conf_Opp', 'Conf_Rank_Opp',\
            'Score_med_Opp', 'DScore_med_Opp', 'FGM_med_Opp', 'FGA_med_Opp', 'FMG3_med_Opp', 'FGA3_med_Opp',\
            'FTM_med_Opp', 'FTA_med_Opp', 'OR_med_Opp', 'DR_med_Opp', 'Ast_med_Opp', 'TO_med_Opp', 'Stl_med_Opp',\
            'Blk_med_Opp', 'PF_med_Opp', 'DFGM_med_Opp', 'DFGA_med_Opp', 'DFGM3_Opp', 'DFGA3_med_Opp', 'DFTM_med_Opp',\
            'DFTA_med_Opp', 'DOR_med_Opp', 'DDR_med_Opp', 'DAst_med_Opp', 'DTO_med_Opp', 'DStl_med_Opp',\
            'DBlk_med_Opp', 'DPF_med_Opp',\
            'Seed', 'Conf', 'seed_hist_pct', 'full_conf', 'Conf_Rank', 'Score_med', 'DScore_med', 'FGM_med',\
            'FGA_med', 'FMG3_med', 'FGA3_med', 'FTM_med', 'FTA_med', 'OR_med', 'DR_med', 'Ast_med', 'TO_med', \
            'Stl_med', 'Blk_med', 'PF_med', 'DFGM_med', 'DFGA_med', 'DFGM3', 'DFGA3_med', 'DFTM_med', 'DFTA_med', \
            'DOR_med', 'DDR_med', 'DAst_med', 'DTO_med', 'DStl_med', 'DBlk_med', 'DPF_med']]

final_df_loss['Result'] = 0
frames = [final_df_win, final_df_loss]
final_df = pd.concat(frames)

final_df['seed_diff'] = final_df['Seed'] - final_df['Seed_Opp']
final_df['conf_diff'] = final_df['Conf_Rank'] - final_df['Conf_Rank_Opp']

final_df.to_csv('input\\training_data.csv', index=False)

final_df_loss.to_csv('input\\training_losses_temp.csv', index=False)
final_df_loss.to_csv('input\\training_wins_temp.csv', index=False)

#test set creation
t_seeds_df = pd.read_csv('input\\datafiles\\NCAATourneySeeds.csv')

teams_conf_df = pd.read_csv('input\\datafiles\\TeamConferences.csv')

seed_hist_df = pd.read_csv('input\\seed_historical.csv')

conf_rank_df = pd.read_csv('input\\conference_rank.csv', delimiter=';')

reg_2018_df = pd.read_csv('input\\datafiles\\RegularSeasonDetailedResults.csv')

matchup_2018_df = pd.read_csv('submissions\\substage2.csv', delimiter=';')
print(matchup_2018_df)
#regular season 2018
matchup_2018_df['Season'] = matchup_2018_df['ID'].str.split('_').str[0]
matchup_2018_df['TeamID'] = matchup_2018_df['ID'].str.split('_').str[1]
matchup_2018_df['TeamID_Opp'] = matchup_2018_df['ID'].str.split('_').str[2]

matchup_2018_df.to_csv('input\\matchup_2018.csv')