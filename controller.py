from flask import Flask, render_template, request

app = Flask(__name__)
CLF_PATH = 'models/trained/AdaBoostClassifier.pkl'


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/predict/', methods=['POST'])
def __predict():
    home_team = request.args['home_team']
    away_team = request.args['away_team']
    if home_team and away_team:
        pass
