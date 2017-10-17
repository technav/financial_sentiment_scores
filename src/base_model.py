from __future__ import unicode_literals
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, \
    recall_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def get_balanced_sample(df, class_label):
    ''' Create a dataframe with balanced samples
    '''
    g = df.groupby(class_label)
    new_df = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
    return new_df

def lemmatize_hl(X_train):
    lem = WordNetLemmatizer()
    lemmatize = lambda d: " ".join(lem.lemmatize(word) for word in d.split())
    return [lemmatize(hl.decode('utf8',errors='ignore')) for hl in X_train]


def get_vectorizer(X_train, num_features=5000):
    vect = TfidfVectorizer(max_features=num_features, stop_words='english',\
                            decode_error='ignore')
    return vect.fit(X_train)


def run_model(model, X_train, X_test, y_train, y_test):
    m = Model()
    m.fit(X_train, y_train)
    y_predict = m.predict(X_test)
    return accuracy_score(y_test, y_predict), \
        f1_score(y_true=y_test, y_pred=y_predict, average='weighted'), \
        precision_score(y_true=y_test, y_pred=y_predict, average='weighted'), \
        recall_score(y_true=y_test, y_pred=y_predict, average='weighted'), \


def fit_model(model, X_train, X_test, y_train, y_test):
    vect = get_vectorizer(X_train)
    X_train = vect.transform(X_train).toarray()
    X_test = vect.transform(X_test).toarray()

    print "acc\tf1\tprec\trecall"
    acc, f1, prec, rec = run_model(model, X_train, X_test, y_train, y_test)
    print "%.4f\t%.4f\t%.4f\t%.4f\t%s" % (acc, f1, prec, rec, name)

if __name__ == '__main__':
    tr_df = pd.read_csv('phrase_bank_train.csv',  \
                names = ['headline', 'label'])
    #Get balanced sample
    # cls_lbl = 'label'
    # bl_df = get_balanced_sample(tr_df, cls_lbl)
    #
    # X = bl_df['headline'].values
    # y = bl_df['label'].values

    X = tr_df['headline'].tolist()
    y = tr_df['label'].tolist()

    #Get train/test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, \
                            test_size=0.2, random_state = 1)

    mnb_model = MultinomialNB()
    #Fit model using train data
    fit_model(mnb_model, lemmatize_hl(X_train),lemmatize_hl(X_test),\
                y_train, y_test)
