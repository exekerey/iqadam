# transport complaints classifier

## setup

```shell
git clone https://github.com/exekerey/iqadam
cd iqadam
python -m venv .venv
pip install -r requirements.txt 
```

And put your Mistral API key into `.env` file as in `.env.example`
(It's used for labeling the data, so no need to label that by hand)

## Usage

- you may replace the dataset in `data/raw/` folder (`AI_dataset.xlsx`)

- then you need to go through `label_data` and `preprocess_x` notebooks to preprocess it. This notebooks label data(as
  there are no Y labels for dataset) and preprocesses text.

- run `train.py`.
  This will train the model(logistic regression), vectorizer(tfidf) and save them as .pkl files

- run `evaluate.py` to eval the model. See precision, recall, f1 and confusion matrix

- you may predict new reviews using trained model running `predict.py`

## main libs used:

- scikit-learn
- pandas
- pymorphy3
- nltk
- matplotlib
- mistralai

## thoughts

The dataset is imbalanced and has only 4 positive examples, compared to 140+ negative ones.
It would be much reasonable to use LLMs with JSON mode for the classification(as I did when labeling data)
