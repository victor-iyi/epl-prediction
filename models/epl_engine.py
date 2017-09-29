import os.path
import pickle
import time

SAVE_DIR = 'trained'
if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)


def train(clf, X_train, y_train):
    start = time.time()
    clf.fit(X_train, y_train)
    print('Training took {:.04f}secs'.format(time.time() - start))
    return clf


def test(clf, X_test, y_test):
    accuracy = clf.score(X_test, y_test)
    return accuracy


def predict(clf, X):
    return clf.predict(X)


def train_predict(clf, X_train, y_train, X):
    train(clf, X_train, y_train)
    return predict(clf, X)


def save_classifier(clf, filename=None, force=False):
    if not filename:
        model_name = str(clf)
        filename = model_name[:model_name.index('(')]
    filename = os.path.join(SAVE_DIR, '{}.pkl'.format(filename))
    if force or not os.path.exists(filename):
        f = open(filename, 'wb')
        pickle.dump(clf, f)
        f.close()
        print('Successfully saved!')
    else:
        raise (Exception('File already exist!'))


def main():
    from sklearn.ensemble import AdaBoostClassifier
    from .preprocess import process, process_to_features

    # Load all the datasets
    home_team = 'arsenal'
    away_team = 'stoke'

    X_train, X_test, y_train, y_test = process(filename=None, test_size=0.1, save_csv=True)
    pred_features = process_to_features(home=home_team, away=away_team)

    print('Training: ', X_train.shape, y_train.shape)
    print('Testing:  ', X_test.shape, y_test.shape)

    try:
        # !- TODO: Try a bunch of classifiers to get better accuracy
        clf = AdaBoostClassifier(n_estimators=500, learning_rate=1e-2)
        train(clf, X_train, y_train)
        accuracy = test(clf, X_test, y_test)
        pred = predict(clf, pred_features)

        print('Soccer prediction accuracy = {:.02%}\n'.format(accuracy))
        print('Prediction: ', pred)

        save_classifier(clf, force=True)
    except Exception as e:
        import sys

        sys.stderr.write(str(e))
        sys.stderr.flush()


if __name__ == '__main__':
    main()
