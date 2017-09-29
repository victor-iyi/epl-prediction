"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  Created on 04 September, 2017 @ 9:58 PM.
  Copyright (c) 2017. Victor. All rights reserved.
"""
import os.path
import warnings
from glob import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# !- constants
DATASET_DIR = '../datasets/'
SAVE_CSV_PATH = os.path.join(DATASET_DIR, 'combined/epl-no-labels.csv')
DATA_FILES = glob(os.path.join(DATASET_DIR, '*.csv'))
USELESS_ROWS = ['Div', 'Date', 'Referee']


def load_data():
    data = []
    for d_file in DATA_FILES:
        d = pd.read_csv(d_file)
        d.drop(USELESS_ROWS, axis=1, inplace=True)
        data.append(d)
    dataset = pd.concat(data)
    return dataset


def get_x_team(home_or_away):
    pd.read_csv()


def process_to_features(home, away):
    df = pd.read_csv(SAVE_CSV_PATH)
    # !- Home team and Away team
    home_team = df['HomeTeam'].values
    away_team = df['AwayTeam'].values
    # !- Get the indexes for home and away team
    home_idx = get_index(home_team.tolist(), home)
    away_idx = get_index(away_team.tolist(), away)
    # !- Drop string columns
    # df.drop(['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTR', 'HTR', 'Referee'], axis=1, inplace=True)
    df.drop(USELESS_ROWS, axis=1, inplace=True)
    # !- Get rows where the home and away team shows up respectively
    home_data = df.values[home_idx]
    away_data = df.values[away_idx]
    # !- Find the average across all records and return
    return np.average(home_data, axis=0), np.average(away_data, axis=0)


def get_index(teams, value):
    value = value.title()
    indexes = [i for i, team in enumerate(teams) if team == value]
    return indexes


def handle_non_numeric(df):
    """
    Processes a dataframe in order to handle for non-numeric data

    :type df: pd.DataFrame
    :param df:
            the dataframe containing the data
    :return df: pd.DataFrame
            Clean numeric DataFrame object
    """
    columns = df.columns.values

    def convert(val):
        return text_digit[val]

    for col in columns:
        text_digit = {}  # {"Female": 0}
        if df[col].dtype != np.int64 and df[col].dtype != np.float64:
            uniques = set(df[col].values.tolist())
            x = 0
            for unique in uniques:
                if unique not in text_digit:
                    text_digit[unique] = x
                    x += 1
            df[col] = list(map(convert, df[col]))
    return df


def process(filename=None, test_size=None, train_size=None, save_csv=False):
    """
    Process data into training and testing set.

    :param filename: str or None (default is None)
            The path to the `csv` file which contains the dataset. If
            set to None, it will load all the datasets.
    :param test_size: float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If
        int, represents the absolute number of test samples. If None,
        the value is automatically set to the complement of the train size.
        If train size is also None, test size is set to 0.25.
    :param train_size: float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
    :param save_csv: bool (default is False)
        Save the processed file into a csv file.
    :return: X_train, X_test, y_train, y_test
            `np.ndarray` o
    """
    if filename:
        data = pd.read_csv(filename)
    else:
        data = load_data()
    # !- FTR = full time result
    X_all = data.drop(['FTR'], axis=1)
    y_all = data['FTR']
    # !- Clean out non numeric data
    X_all = handle_non_numeric(X_all)
    X_all.fillna(0, inplace=True)  # because the model is seeing some NaN values
    # !- Save processed data to csv
    if save_csv:
        X_all.to_csv(SAVE_CSV_PATH)
    # !- Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,
                                                        test_size=test_size, train_size=train_size,
                                                        random_state=42, stratify=y_all)
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def main():
    X_train, X_test, y_train, y_test = process(filename=None, save_csv=True)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)


if __name__ == '__main__':
    main()
