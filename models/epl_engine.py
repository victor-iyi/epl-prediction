import time
import pickle
import os.path

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
        print('File already exist!')


if __name__ == '__main__':
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import AdaBoostClassifier

    from process_data import process, CURR_SEASON_DATA

    X_train, X_test, y_train, y_test = process(CURR_SEASON_DATA)

    clf = LogisticRegression()
    train(clf, X_train, y_train)
    accuracy = test(clf, X_test, y_test)
    pred = predict(clf, [X_test[0]])
    print(pred, y_test[0])
    print('LogisticRegression Accuracy = {:.02%}\n'.format(accuracy))

    clf = DecisionTreeClassifier()
    train(clf, X_train, y_train)
    accuracy = test(clf, X_test, y_test)
    pred = predict(clf, [X_test[0]])
    print(pred, y_test[0])
    print('DecisionTreeClassifier Accuracy = {:.02%}\n'.format(accuracy))

    clf = AdaBoostClassifier()
    train(clf, X_train, y_train)
    accuracy = test(clf, X_test, y_test)
    pred = predict(clf, [X_test[0]])
    print(pred, y_test[0])
    print('AdaBoostClassifier Accuracy = {:.02%}\n'.format(accuracy))

    save_classifier(clf, force=True)
