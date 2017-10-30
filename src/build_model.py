import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report as clr
from sklearn.model_selection import GridSearchCV
from pre_processor import TextPreprocessor
from collections import defaultdict
from feature_selection import get_important_features

'''
This module fits the final model (selected from 'run_model_selection')
to the data, prints out important features per class, classification
report and finally saves the model after tuning different parameters.
Import this functionality in run_classification.
'''

def override(x):
    ''' Template function to override the 'preprocessor' and
    'tokenizer' steps inside sklearn TfidfVectorizer
    '''
    return x

def fit_model(X_train, y_train, X_test, y_test, output_path):
    ''' Run gridsearch to find best models for lasso, linear SVM,
    Random forest classifier and adaboost classifier
    '''
    tfidf = TfidfVectorizer(lowercase=False, decode_error = 'ignore',
                            preprocessor = override,
                            tokenizer = override,
                            max_features=5000)

    #Initialize parameter grid for MultinomialNB/SGDC
    # param_mnb = { 'vect__max_df': (0.5, 0.75, 1.0),
    #               'vect__ngram_range': [(1, 1), (1, 2)],
    #               'vect__norm':['l1', 'l2']}

    param_sgd = {'vect__max_df': (0.5, 0.75, 1.0),
                'vect__ngram_range': [(1, 1), (1, 2)],
                'vect__norm':['l1', 'l2'],
                'vect__max_features':[5000, 8000, 10000],
                'clf__alpha': (0.00001, 0.000001),
                'clf__penalty': ('l2', 'elasticnet'),
                'clf__max_iter': (1000,),
                'clf__tol': (1e-3,),
                'clf__class_weight' : [{0:0.1, 1:0.3, -1:0.6},
                                {0:0.2, 1:0.4, -1:0.4}],
                'clf__loss':('modified_huber',)}

    pipeline = Pipeline([('pre_process', TextPreprocessor()),
                         ('vect', tfidf),
                         ('clf', SGDClassifier())])
                         #('clf', MultinomialNB())])
    gscv = GridSearchCV(pipeline, param_mnb, scoring='accuracy',
                                   cv=8, verbose=0, n_jobs=-1)
    gscv.fit(X_train, y_train)
    model = gscv.best_estimator_
    print "Best params: {}".format(gscv.best_params_)
    print "Best CV score: %0.3f" % gscv.best_score_

    #Get predictions and print out classification report
    y_pred = model.predict(X_test)
    print 'Classification Report: ',clr(y_test, y_pred)
    print clr(y_test, y_pred)

    # Print out most important features in 'bullish'/'bearish' class
    print 'Top 10 important features for bullish/bearish classes:'
    get_important_features(model)

    #Save the model to use it later to annotate actual test data
    if output_path:
        os.chdir(output_path)
        with open('clf_model_1.sav', 'wb') as f:
            pickle.dump(model, f)

        print 'Model written to: {}'.format(output_path)
