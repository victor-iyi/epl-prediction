import os.path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import preprocess

DATASET_DIR = '../datasets/'
DATA_FILES = ['epl-2015-2016.csv', 'epl-2016-2017.csv', 'epl-2017-2018.csv']
CURR_SEASON_DATA = os.path.join(DATASET_DIR, DATA_FILES[-1])
USELESS_ROWS = ['Referee', 'Div', 'Date', 'HomeTeam', 'AwayTeam']


def load_data():
    dataset = pd.read_csv(CURR_SEASON_DATA)
    dataset.drop(USELESS_ROWS, axis=1, inplace=True)
    for d_file in DATA_FILES[:-1]:
        d_file = os.path.join(DATASET_DIR, d_file)
        data = pd.read_csv(d_file)
        data.drop(USELESS_ROWS, axis=1, inplace=True)
        dataset = pd.concat([dataset, data])
    return dataset


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
    value = value.title()
    indexes = [i for i, team in enumerate(teams) if team == value]
    return indexes


def preprocess_features(X):
    # init new output dataframe
    """
    Cleans up any non-numeric data.

    :param X:
        Features to be cleaned.

    :return: output `pd.DataFrame`
        A new pandas DataFrame object with clean numeric values.
    """
    output = pd.DataFrame(index=X.index)
    # investigate each feature col for data
    for col, col_data in X.iteritems():
        # if data is categorical, convert to dummy variables
        if col_data.dtype == object:
            print('obj lets get dummies')
            col_data = pd.get_dummies(col_data, prefix=col)
        # collect the converted cols
        output = output.join(col_data)
    return output


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
    print(data.columns.values)
    # FTR = full time result
    X_all = data.drop(['FTR'], axis=1)
    y_all = data['FTR']
    X_all = preprocess.process_data(X_all)
    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,
                                                        test_size=test_size, train_size=train_size,
                                                        random_state=42, stratify=y_all)
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


if __name__ == '__main__':
    home_data, away_data = get_remaining_features(home='arsenal', away='chelsea')
    print(home_data, '\n')
    print(away_data)
    # data = load_data()
    # print(data.tail(3))
    # X_train, X_test, y_train, y_test = process(filename=None)
    # print(X_train.shape, y_train.shape)
    # print(X_test.shape, y_test.shape)
