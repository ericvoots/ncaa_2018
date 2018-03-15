from bracketeer import build_bracket

import numpy as np
import pandas as pd

from hashlib import sha256

#doesn't work for 2018
datapath = 'input\\datafiles\\'
files = {
    #'season_compact': "RegularSeasonCompactResults.csv",
    #'tourney_compact': "NCAATourneyCompactResults.csv",
    #'season_detail': "RegularSeasonDetailedResults.csv",
    #'tourney_detail': "NCAATourneyDetailedResults.csv",
    'sample_sub': "SampleSubmissionStage1.csv",
    'seeds': "NCAATourneySeeds.csv",
    #'seasons': "Seasons.csv",
    'slots': "NCAATourneySlots.csv",
    'teams': "Teams.csv",
}
files = {key: datapath + value for key, value in files.items()}

dfs = {key: pd.read_csv(value) for key, value in files.items()}
dfs['teams'] = dfs['teams'].drop(['FirstD1Season', 'LastD1Season'], axis=1)

"""
teamsPath='../input/DataFiles/Teams.csv',
seedsPath='../input/DataFiles/NCAATourneySeeds.csv',
slotsPath='../input/DataFiles/NCAATourneySlots.csv',
submissionPath='../output/submission_seeddiff_only.csv',
year=year,
outputPath=f'../output/sub_
""" and None

#for key in dfs.keys():
#    print(f"{key}: {dfs[key].columns}")


teams = dfs['teams'].copy().reset_index()
teams['index'] = teams['index'] * 314
teams['new_id'] = teams['index'].astype(str)# + teams['TeamID'].apply(
       # lambda x: sha256(str(x).encode('utf-8')).hexdigest()[:10])

teams['new_teamname'] = teams['TeamName'].apply(
    lambda x: sha256(str(x).encode('utf-8')).hexdigest()[:10])
teamid_map = dict(zip(teams['TeamID'].values, teams['new_id'].values))


seeds = dfs['seeds'].copy()
seeds['new_teamid'] = seeds['TeamID'].map(teamid_map)
seeds['new_seed'] = seeds['Seed'].apply(lambda x: sha256(str(x).encode('utf-8')).hexdigest()[:6])
seed_map = dict(zip(seeds['Seed'].values, seeds['new_seed'].values))

slots = dfs['slots'].copy()
slots['new_slot'] = slots['Slot'].apply(lambda x: sha256(str(x).encode('utf-8')).hexdigest()[:10])
slots.loc[slots['Slot']=='R6CH', 'new_slot'] = 'R6CH'

slot_map = dict(zip(slots['Slot'].values, slots['new_slot'].values))
slot_map.update(seed_map)
slot_map.update({"R6CH": "R6CH"})

slots['new_slot'] = slots['Slot'].map(slot_map)
slots['new_strongseed'] = slots['StrongSeed'].map(slot_map)
slots['new_weakseed'] = slots['WeakSeed'].map(slot_map)

sub = dfs['sample_sub'].copy()
sub[['season', 'id1', 'id2']] = sub['ID'].str.split('_', expand=True)

sub['id1'] = sub['id1'].astype(int).map(teamid_map)
sub['id2'] = sub['id2'].astype(int).map(teamid_map)
sub['new_id'] = sub['season'] + '_' + sub['id1'] + '_' + sub['id2']

sub['id1'] = sub['id1'].apply(lambda x: int('0x' + x, 0))
sub['id2'] = sub['id2'].apply(lambda x: int('0x' + x, 0))
sub['new_pred'] = sub['id1'] - sub['id2']
sub['new_pred'] = sub['new_pred'] - sub['new_pred'].min()
sub['new_pred'] = sub['new_pred']/sub['new_pred'].max()

teamsout = (
    teams[['new_id', 'new_teamname']]
    .rename(columns={'new_id': 'TeamID', 'new_teamname': 'TeamName'})
)
seedsout = (
    seeds[['Season', 'new_seed', 'new_teamid']]
    .rename(columns={'new_seed': 'Seed', 'new_teamid': 'TeamID'})
)
slotsout = (
    slots[['Season', 'new_slot', 'new_strongseed', 'new_weakseed']]
    .rename(columns={'new_slot': 'Slot', 'new_strongseed': 'StrongSeed', 'new_weakseed': 'WeakSeed'})
)
subout = (
    sub[['new_id', 'new_pred']]
    .rename(columns={'new_id': 'ID', 'new_pred': 'Pred'})
)

teamsout.to_csv('input\\teams.csv', index=False)
seedsout.to_csv('input\\seeds.csv', index=False)
slotsout.to_csv('input\\slots.csv', index=False)
subout.to_csv('input\\sub.csv', index=False)

b = build_bracket(
        outputPath='output.png',
        teamsPath='input\\teams.csv',
        seedsPath='input\\seeds.csv',
        submissionPath='submissions\\ensemble_pred.csv',
        slotsPath='input\\slots.csv',
        year=2018
)