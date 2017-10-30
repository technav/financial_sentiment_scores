from model_selection import pre_processor, tokenizer, \
            tokenizer_snowball, tokenizer_lancaster
from model_selection import get_best_model
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np

'''
Kicks off the model selection process to tune various parameters
in an exhaustive grid search and prints out different metrics to
help with final model selection
'''

if __name__ == '__main__':
    df = pd.read_csv('~/prj_prep/data/phrase_bank_train.csv',  \
                names = ['headline', 'label'])
    X = df['headline'].tolist()
    y = df['label'].tolist()

    #Remove after testing
    X = X[:1000]
    y = y[:1000]

    X_train, X_test, y_train, y_test = train_test_split(X, y, \
                            test_size=0.2, random_state = 1)

    models = get_best_model(X_train, y_train)
