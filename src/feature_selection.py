import pandas as pd
import numpy as np


def get_important_features(model, n=10):
    ''' Prints out n most important 'Bullish' (from positive class)
    and 'Bearish' (from negative class) features from the output
    of pipeline (model: Assumption that it is a logistic clf )
    INPUT : model
    OUTPUT: None
    '''
    #Intialize the classifier and vectorizer from the model
    clf = model.named_steps['clf']
    vctrz = model.named_steps['vect']

    #Get the coeffs vectors
    coef_vec = clf.coef_

    # Join feature_names and coeffs using zip and sort
    sorted_coefs = sorted(zip(coef_vec[0], vctrz.get_feature_names()),
                            key=itemgetter(0), reverse=True)

    top_ftrs = []
    top_n = zip(sorted_coefs[:n], sorted_coefs[:-(n+1):-1])

    #Read positive and negative class coeffs separately to get
    #top features in both classes
    for (pos_coef, neg_ftr), (neg_coef, pos_ftr) in top_n:
        top_ftrs.append("{:0.4f}{: >15}    {:0.4f}{: >15}".\
                        format(pos_coef, neg_ftr, neg_coef, pos_ftr))

    print "\n".join(top_ftrs)
