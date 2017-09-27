import os.path
import pandas as pd
import numpy as np

DATASET_DIR = '../datasets/'
DATA_FILES = ['epl-2015-2016.csv', 'epl-2016-2017.csv', 'epl-2017-2018.csv']
IMPORTANT_FEATURES = []
CURR_SEASON_DATA = os.path.join(DATASET_DIR, DATA_FILES[-1])


def load_data():
    for d_file in DATA_FILES:
        d_file = os.path.join(DATASET_DIR, d_file)
        data1 = pd.read_csv(d_file)
        print(data1.head())


def get_remaining_features(home, away):
    df = pd.read_csv(CURR_SEASON_DATA)
    # Home team and Away team
    home_team = df['HomeTeam'].values
    away_team = df['AwayTeam'].values
    # Get the indexes for home and away team
    home_idx = get_index(home_team.tolist(), home)
    away_idx = get_index(away_team.tolist(), away)
    # Drop string columns
    df.drop(['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTR', 'HTR', 'Referee'], axis=1, inplace=True)
    # Get rows where the home and away team shows up respectively
    home_data = df.values[home_idx]
    away_data = df.values[away_idx]
    return np.average(home_data, axis=0), np.average(away_data, axis=0)


def get_index(teams, value):
    indexes = []
    for i, team in enumerate(teams):
        value = value.title()
        if team == value:
            indexes.append(i)
    return indexes


home_data, away_data = get_remaining_features(home='arsenal', away='chelsea')
print(home_data, '\n')
print(away_data)
