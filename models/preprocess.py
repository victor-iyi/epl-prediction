"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  Created on 04 September, 2017 @ 9:58 PM.
  Copyright (c) 2017. Victor. All rights reserved.
"""
import os.path
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

DATASET_DIR = '../datasets/'
DATA_FILES = ['epl-2015-2016.csv', 'epl-2016-2017.csv', 'epl-2017-2018.csv']
CURR_SEASON_DATA = os.path.join(DATASET_DIR, DATA_FILES[-1])
USELESS_ROWS = ['Referee', 'Div', 'Date', 'HomeTeam', 'AwayTeam']


def load_data():
    # dataset = pd.read_csv(CURR_SEASON_DATA)
    # dataset.drop(USELESS_ROWS, axis=1, inplace=True)
    data = []
    for i, d_file in enumerate(DATA_FILES):
        print(d_file)
        d_file = os.path.join(DATASET_DIR, d_file)
        d = pd.read_csv(d_file)
        d.drop(USELESS_ROWS, axis=1, inplace=True)
        data.append(d)
    dataset = pd.concat(data)
    return dataset


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


def process(filename=None, test_size=None, train_size=None):
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
    :return: X_train, X_test, y_train, y_test
            `np.ndarray` o
    """
    if filename:
        data = pd.read_csv(filename)
    else:
        data = load_data()
    # FTR = full time result
    X_all = data.drop(['FTR'], axis=1)
    y_all = data['FTR']
    X_all = handle_non_numeric(X_all)
    X_all.fillna(0, inplace=True)  # because the model is seeing some NaN values
    # X_all.to_csv('X_all.csv')
    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,
                                                        test_size=test_size, train_size=train_size,
                                                        random_state=42, stratify=y_all)
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def main():
    X_train, X_test, y_train, y_test = process(filename=None)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)


if __name__ == '__main__':
    main()
