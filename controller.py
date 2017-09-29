import pickle
from flask import Flask, render_template, request
from models.epl_engine import train_predict

app = Flask(__name__)
CLF_PATH = 'models/trained/AdaBoostClassifier.pkl'


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        home_team = request.form['home_team']
        away_team = request.form['away_team']
        if home_team and away_team:
            f = open(file=CLF_PATH, mode='rb')
            clf = pickle.load(f)
            train_predict(clf, home_team, away_team)
    return render_template('index.html')
