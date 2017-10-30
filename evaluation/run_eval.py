import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_pred(y_true, y_pred):
    ''' Return cosine similarity after evaluating the predicted values
    '''
    cosine_smty = cosine_similarity(np.array(y_pred).reshape(1, -1), \
                          np.array(y_true).reshape(1, -1))[0][0]

    return cosine_smty
