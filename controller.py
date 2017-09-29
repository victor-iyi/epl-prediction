import os.path
import pickle

from flask import Flask, render_template, request, jsonify

from models.epl_engine import predict
from models.preprocess import get_all_teams, process_to_features

app = Flask(__name__)
CLF_PATH = os.path.join('models/trained/', 'AdaBoostClassifier.pkl')


@app.route('/', methods=['GET', 'POST'])
def index():
    all_teams = get_all_teams()
    return render_template('index.html', all_teams=all_teams)


@app.route('/__predict', methods=['POST'])
def __predict():
    data = {
        'status': False,
        'response': {
            'data': None
        },
    }
    home_team = request.json['home_team']
    away_team = request.json['away_team']
    if home_team and away_team:
        f = open(CLF_PATH, 'rb')
        clf = pickle.load(f)
        pred_features = process_to_features(home=home_team, away=away_team)
        pred = predict(clf, pred_features)  # ['H', 'A'] H = Home team A = Away team, D = Draw
        result_construct = {
            'result': pred[0]
        }
        data['status'] = True
        data['response']['data'] = result_construct
    return jsonify(data)
