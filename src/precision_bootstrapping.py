from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.utils import shuffle
from operator import itemgetter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report as clr
from sklearn.model_selection import GridSearchCV
from pre_processor import TextPreprocessor
from sklearn.metrics import accuracy_score, precision_score, recall_score
from collections import defaultdict

'''
    Self-training iteratively expands the labeled data by using
    the classifier to predict on unlabeled data in each iteration.
    Classifier is trained in each iteration and evaluated using
    one set of gold labels. Loop of iteration stops when the required
    balance between precision/recall is reached
'''

def override(x):
    ''' Template function to override the 'preprocessor' and
    'tokenizer' steps inside sklearn TfidfVectorizer
    '''
    return x

def fit_model(X_train, y_train, X_test, y_test):
    ''' Run gridsearch to find best models for lasso, linear SVM,
    Random forest classifier and adaboost classifier
    '''
    tfidf = TfidfVectorizer(lowercase=False, decode_error = 'ignore',
                            preprocessor = override,
                            tokenizer = override,
                            max_features=5000, ngram_range = (1,2))

    #Initialize parameter grid for SGDC
    param_sgd = {'clf__alpha': (0.00001, 0.000001),
                'clf__penalty': ('l2', 'elasticnet'),
                'clf__max_iter': (1000,),
                'clf__tol': (1e-3,),
                'clf__loss':('modified_huber',)}

    pipeline = Pipeline([('pre_process', TextPreprocessor()),
                         ('vect', tfidf),
                         ('clf', SGDClassifier())])

    gscv = GridSearchCV(pipeline, param_sgd
                            ,scoring='accuracy',
                            cv=8, verbose=0, n_jobs=-1)
    gscv.fit(X_train, y_train)
    model = gscv.best_estimator_
    print "Best params: {}".format(gscv.best_params_)
    print "Best CV score: %0.3f" % gscv.best_score_
    return model



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


def train_data(train_df, itr, unlabeled_df, evaluation_df):
    ''' Trains a new model on the new set of training data
    using the same features and labels a new set of
    previously unlabeled data which will be added to the
    training data for the next loop
    '''
    df = shuffle(train_df)
    X = df['headline'].tolist()
    y = df['label'].tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, \
                            test_size=0.2, random_state = 1)

    model = fit_model(X_train, y_train, X_test, y_test)

    # Print out most important features
    print 'Most important features for iteration: {}'.format(itr)
    get_important_features(model)

    #Get predictions on test split
    y_pred = model.predict(X_test)
    print 'Classification Report for test data for iteration: {}'.format(itr)
    print clr(y_test, y_pred)

    #Evaluate the model using evaluation_df which alread has labels
    eval_data_pred = model.predict(evaluation_df['headline'])
    print 'Classification Report for EVALUATION data for iteration: {}'.format(itr)
    print clr(evaluation_df['label'], eval_data_pred)

    #Annotate unlabeled data
    unlabeled_df['label'] = model.predict(unlabeled_df['headline'])
    itr += 1

    return model, unlabeled_df, itr




if __name__ == '__main__':
    #Read the first set of gold labels to start the training
    df = pd.read_csv('gold1.csv', names = ['headline', 'label'])

    #Read the second set of heldout gold labels for evaluation at each loop
    g2_df = pd.read_csv('gold2.csv', names = ['headline', 'label'])

    #Read all unlabeled data in chunks
    pr1_df = pd.read_csv('prc_ds1.csv', names = ['headline'])
    pr2_df = pd.read_csv('prc_ds2.csv', names = ['headline'])
    pr3_df = pd.read_csv('prc_ds3.csv', names = ['headline'])
    pr4_df = pd.read_csv('prc_ds4.csv', names = ['headline'])

    # initialize i to track iterations:
    i=1
    #Start the first training
    model1, res1_df, i = train_data(df, i, pr1_df, g2_df)

    #Concat all the available labeled dataframes and use it as train
    # Next rounds of training
    # 2
    df = pd.concat([df, res1_df])
    model2, res2_df, i = train_data(df, i, pr2_df, g2_df)

    # 3
    df = pd.concat([df, res2_df])
    model3, res3_df, i = train_data(df, i, pr3_df, g2_df)

    # 4 and this gives our final model
    df = pd.concat([df, res3_df])
    model4, res4_df, i = train_data(df, i, pr4_df, g2_df)

    # Read the last set of heldout gold labels and using the best model
    # from above iterations get final evaluation
    g3_df = pd.read_csv('gold3.csv', names = ['headline', 'label'])
    g3_pred = model3.predict(g3_df['headline'])
    print 'Classification Report for FINAL EVALUATION data: '
    print clr(g3_df['label'], g3_pred)
