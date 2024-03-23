# English Premier Leauge Prediction

The scope of this project is to predict the outcome of an English Premier Leauge
match, based on historic data.

For more information about the data used, check [notes.txt].

[notes.txt]: ./datasets/notes.txt

## Installation

Requirements:

- [Poetry]
- [Python 3.10+][python.org]
- `pip install virtualenv`

[Poetry]: https://python-poetry.org/docs/
[python.org]: https://www.python.org/downloads/

Create a virutal environment:

```sh
virtualenv .venv
```

Activate the environment:

```sh
source .venv/bin/activate
```

Install dependencies:

```sh
poetry install
```

## Setup

Setup involves a 2 stage process.

- Run the Flask app
- Run training script and save the trained model.

> **Note:** These steps are independent of each other as there is already a pre-trained
model saved in [`AdaBoostClassifier.pkl`]

[`AdaBoostClassifier.pkl`]: ./models/trained/AdaBoostClassifier.pkl

### Run the Flask app

To run the application simply run the following command:

```sh
python __init__.py
```

The server should hopefully start running in <http://127.0.0.1:5000>

![Home Page][homepage]

[homepage]: ./static/images/index.png

### Train the model

To train the model, simply run the following command:

```sh
python -m models.epl_engine
```

#### Change the default model

The default model used is `AdaBoostClassifier`, however you can change this to
which ever `sklearn` model you desire by updating the `main` function in [`epl_engine.py`]

[`epl_engine.py`]: ./models/epl_engine.py

```py
def main() -> None:
    # Change to the sklearn model you wish to use.
    from sklearn.ensemble import AdaBoostClassifier
    
    # ...code snippet

    # Update the following line accordingly to the imported model.
    clf = AdaBoostClassifier(n_estimators=500, learning_rate=1e-2)
```

> **Note:** If you decide to change the model, be also sure to update the loaded
classifier in the [`controller.py`]

```py
# Change "AdaBoostClassifier.pkl" to whichever model you which to used
# saved in "models/trained/" folder.
CLF_PATH: str = os.path.join('models/trained/', 'AdaBoostClassifier.pkl')
```

[`controller.py`]: ./controller.py

## Credits

- Victor I. Afolabi - [victor-iyi]
- Tomiiide - [tomiiide]

[victor-iyi]: https://github.com/victor-iyi
[tomiiide]: https://github.com/tomiiide

## Contribution

You are very welcome to modify and use them in your own projects.

Please keep a link to the [original repository]. If you have made a fork with
substantial modifications that you feel may be useful, then please [open a new
issue on GitHub][issues] with a link and short description.

## License (MIT)

This project is opened under the [MIT][license] which allows very
broad use for both private and commercial purposes.

A few of the images used for demonstration purposes may be under copyright.
These images are included under the "fair usage" laws.

[original repository]: https://github.com/victor-iyi/epl-prediction
[issues]: https://github.com/victor-iyi/epl-prediction/issues
[license]: ./LICENSE
