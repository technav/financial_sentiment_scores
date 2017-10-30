import pandas as pd
import numpy as np
import os
import pickle

'''
Loads the model that was pre-fit and saved and uses the model to
annotate the test data
'''

if __name__ == '__main__':
    # Load test data that needs to be annotated
    t_df = pd.read_csv('R2000_50.csv', names = ['ticker', 'pub_dt', 'headline'])
    t_df['publish_date'] = pd.to_datetime(t_df.pub_dt, errors = 'coerce')
    t_df = t_df.drop('pub_dt',1)
    # Drop duplication headlines for the same ticker on same date
    t_df = t_df.drop_duplicates(['ticker', 'publish_date', 'headline'])

    # Load the model
    model_file = open("clf_sgdc_1.sav", "rb")
    sgd_clf = pickle.load(model_file)
    model_file.close()

    # Use the model to label or predict sentiment on the test data
    t_df['label'] = sgd_clf.predict(t_df['headline'])
