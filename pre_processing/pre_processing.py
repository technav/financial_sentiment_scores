# Preprocessing steps
from __future__ import unicode_literals
import pandas as pd
import string
from nltk import wordpunct_tokenize
from nltk import sent_tokenize, pos_tag
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from spacy.en import English
'''txt = txt.decode('utf8', errors='ignore')
'''
class TextPreProcessor(TransformerMixin):

    def __init__(self):
        pass
        # self.stopwords = set(sw.words('english') + \
        #             ['pct', 'news', 'GMT', 'AM', 'PM', 'Reuters', \
        #             'reuters',  'visit', 'click', \
        #             'Yahoo', 'suggest', 'feedback', 'alert', 'email', \
        #             'client', 'link', 'site', 'report', 'reports', \
        #             'reporting', 'said', 'data', 'market', 'markets', \
        #             'time', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', \
        #             'Friday', 'Saturday', 'Sunday', 'people', 'likely'])
        # self.symbols =  " ".join(string.punctuation).split(" ") + \
        #             ["\n", "\n\n", "", " ", "---", "..."]
        # self.parser = English()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        all_tkns = [custom_tokenize(hl.decode('utf8',errors='ignore')) for hl in X]
        #hdlns = [self.custom_tokenize(hl) for hl in X]
        return all_tkns


def custom_tokenize(headline):
    stopwords = set(sw.words('english') + \
            ['pct', 'news', 'GMT', 'AM', 'PM', 'Reuters', \
            'reuters',  'visit', 'click', \
            'Yahoo', 'suggest', 'feedback', 'alert', 'email', \
            'client', 'link', 'site', 'report', 'reports', \
            'reporting', 'said', 'data', 'market', 'markets', \
            'time', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', \
            'Friday', 'Saturday', 'Sunday', 'people', 'likely'])
    symbols =  " ".join(string.punctuation).split(" ") + \
            ["\n", "\n\n", "", " ", "---", "..."]
    parser = English()
    tokens = parser(headline)
    tokens = [tok.lemma_.lower() for tok in tokens if (tok not in \
                stopwords or tok not in symbols)]
    #print len(tokens)
    return tokens

if __name__ == '__main__':
    df = pd.read_csv('phrase_bank_train.csv',  \
                names = ['headline', 'label'])
    X = df['headline'].tolist()
    y = df['label'].tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    #Remove after testing
    X_train = X_train[:100]
    y_train = y_train[:100]

    #Initialize vectorizer
    #vectorizer = CountVectorizer(tokens)


    #Initialize pipeline
    pipe = Pipeline([("preprocess", TextPreProcessor()),
                    ('vectorizer', CountVectorizer()),
                    ('classifier', MultinomialNB())])
    pipe.fit(X_train, y_train)
