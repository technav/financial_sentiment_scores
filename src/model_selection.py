from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from nltk.stem import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk import pos_tag
import re

'''
Module meant for model selection by looking at various tokenization /
vectorization and classifier parameters
'''

def pre_processor(text):
    ''' Clean text to remove non-alpha characters
    INPUT: str
    OUTPUT: str
    '''
    text = re.sub('<[^>]*>', '', text)
    #emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower())
    return text

def tokenizer(text):
    ''' Regular tokenizer
    INPUT: str
    OUTPUT: list
    '''
    sw = stopwords.words('english')
    text = pre_processor(text)
    return [word for word in text.split() if word not in sw]

def tokenizer_snowball(text):
    ''' SnowballStemmer to transform word into its root form
    INPUT: str
    OUTPUT: list
    '''
    sw = stopwords.words('english')
    snowball = SnowballStemmer('english')
    text = pre_processor(text)
    return [snowball.stem(word) for word in text.split() \
                if word not in sw]

def tokenizer_lancaster(text):
    ''' Lancaster stemmer to transform word into its root form
    INPUT: str
    OUTPUT: list
    '''
    sw = stopwords.words('english')
    lcstr = LancasterStemmer()
    text = pre_processor(text)
    return [lcstr.stem(word) for word in text.split() \
                if word not in sw]

def lemmatize(text):
    '''
    Get tokens, POS tags and lemmatizes tokens using WordNetLemmatizer.
    INPUT: str
    OUTPUT: list
    '''
    tag = {'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(pos_tag[0], wn.NOUN)

    text = pre_processor(text)
    cleaned_tokens = []

    # Get associated POS tag before lemmatizing
    pos_tk_hl = pos_tag(text.split())
    for token, tag in pos_tk_hl:
        token = wn.lemmatize(token, tag)
        cleaned_tokens.append(token)

    cleaned_tokens = [tk for tk in cleaned_tokens if tk != '']
    return cleaned_tokens


def get_best_model(X, y):
    ''' Run gridsearch to find best models for lasso, linear SVM,
    Random forest classifier and adaboost classifier
    INPUT: list of train headlines
            list of train labels
    OUTPUT: list of models
            model from GSCV
    '''
    sw = stopwords.words('english')

    tfidf = TfidfVectorizer(lowercase=False, decode_error = 'ignore',\
                            max_features=5000)

    log_reg = {}
    log_reg['model'] = 'LogReg'
    log_reg['pipe'] = Pipeline([('vect', tfidf),\
                         ('clf', LogisticRegression(random_state=0))])
    log_reg['param'] = {'vect__ngram_range': [(1, 1), (1, 2)],
                        'vect__tokenizer': [tokenizer, tokenizer_lancaster, \
                                             tokenizer_snowball, lemmatize],
                        'vect__norm':['l1', 'l2'],
                        'clf__penalty': ['l1', 'l2'],
                        'clf__C': [1.0, 10.0, 100.0]}

    rf_clf = {}
    rf_clf['model'] = "RandomForestClf"
    rf_clf['pipe'] = Pipeline([('vect', tfidf),\
                         ('clf', RandomForestClassifier(random_state=0))])
    rf_clf['param'] = {'vect__ngram_range': [(1, 1), (1, 2)],
                   'vect__tokenizer': [tokenizer, tokenizer_lancaster, \
                                        tokenizer_snowball, lemmatize],
                    'clf__max_features':[None, "sqrt"],
                    'clf__min_samples_split':[0.6, 0.7],
                    'clf__n_estimators':[100, 500, 1000],
                   'vect__norm':['l1', 'l2']}

    ada_clf = {}
    ada_clf['model'] = "AdaBoostClf"
    ada_clf['pipe'] = Pipeline([('vect', tfidf),\
                         ('clf', AdaBoostClassifier(random_state=0))])
    ada_clf['param'] = {'vect__ngram_range': [(1, 1), (1, 2)],
                        'vect__tokenizer': [tokenizer, tokenizer_lancaster, \
                                        tokenizer_snowball, lemmatize],
                        'vect__norm':['l1', 'l2']}

    mnb_clf = {}
    mnb_clf['model'] = "MultinomialNB"
    mnb_clf['pipe'] = Pipeline([('vect', tfidf),\
                         ('clf', MultinomialNB())])
    mnb_clf['param'] = {'vect__ngram_range': [(1, 1), (1, 2)],
                        'vect__tokenizer': [tokenizer, tokenizer_lancaster, \
                                        tokenizer_snowball, lemmatize],
                        'vect__norm':['l1', 'l2']}

    sgd_clf = {}
    sgd_clf['model'] = "SGDC"
    sgd_clf['pipe'] = Pipeline([('vect', tfidf),\
                         ('clf', SGDClassifier())])
    sgd_clf['param'] = {'vect__max_df': (0.5, 0.75, 1.0),
                        'vect__ngram_range': [(1, 1), (1, 2)],
                        'vect__tokenizer': [tokenizer, tokenizer_lancaster, \
                                            tokenizer_snowball, lemmatize],
                        'vect__norm':['l1', 'l2'],
                        'clf__alpha': (0.00001, 0.000001),
                        'clf__penalty': ('l2', 'elasticnet'),
                        'clf__max_iter': (1000,),
                        'clf__tol': (1e-3,),
                        'clf__loss':('modified_huber',)}

    models = defaultdict(list)

    #for model in [rf_clf, ada_clf, log_reg, mnb_clf, sgd_clf]:
    for model in [ada_clf]:
        models['modelName'].append(model['model'])
        gscv = GridSearchCV(model['pipe'], model['param'], scoring='accuracy',\
                                   cv=10, verbose=0, n_jobs=-1)
                # scoring = make_scorer(mean_squared_log_error,\
                # greater_is_better=False),cv = 10, n_jobs =-1)
        gscv.fit(X, y)
        print "Best params for {}: {}".format(models['modelName'], gscv.best_params_)
        models['estimators'].append(gscv.best_estimator_)
        models['scores'].append(gscv.best_score_)
    return models, gscv.best_estimator_
