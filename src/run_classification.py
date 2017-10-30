from build_model import fit_model
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np

'''
This kick starts the model building by tuning various vectorizer and
classifier parameters and proceeds to feature selection and printing
classification report
'''
if __name__ == '__main__':
    output_path = '~/prj_prep/src/'
    df = pd.read_csv('~/prj_prep/data/phrase_bank_train.csv',
                names = ['headline', 'label'])
    X = df['headline'].tolist()
    y = df['label'].tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, \
                            test_size=0.2, random_state = 1)

    # Fits the best model using GSCV and prints out classification_report
    # and top 10 important features per class
    fit_model(X_train, y_train, X_test, y_test, output_path)
