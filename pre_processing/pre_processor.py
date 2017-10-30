from __future__ import unicode_literals
import string
from operator import itemgetter
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import WordNetLemmatizer
import re
import numpy as np
from textblob import TextBlob
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """ Custom sklearn transformer to preprocess data.
    Process, clean headlines and transforms data by using
    Textblob tokenization, lemmatization and other filtering techniques.
    """

    def __init__(self, stopwords=None, punct=None):
        """
        Instantiates the Text processor
        """
        self.stopwords = set(sw.words('english') + \
                    ['pct', 'news', 'GMT', 'AM', 'PM', 'Reuters', \
                    'reuters',  'visit', 'click', \
                    'Yahoo', 'suggest', 'feedback', 'alert', 'email', \
                    'client', 'link', 'site', 'report', 'reports', \
                    'reporting', 'said', 'data', 'market', 'markets', \
                    'time', 'monday', 'tuesday', 'wednesday', 'thursday', \
                    'friday', 'saturday', 'sunday', 'people', 'likely'])

    def fit(self, X, y=None):
        """
        Fit and return self.
        """
        return self

    def transform(self, X):
        """
        Pre-processing of each headline.
        """
        all_tokens = []
        for hl in X:
            hl = hl.decode('ascii',errors='ignore')
            hl = re.sub('[^a-zA-Z0-9\n\.]', '',hl)
            hl = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", hl)
            hl = TextBlob(hl)
            # Remove punctuation using TextBlob.words
            # hl = ' '.join(w for w in hl.words if len(w) > 2 \
            #                 and w.lower() not in self.stopwords)
            all_tokens.append(list(self.tokenize(hl)))

        return all_tokens


    def tokenize(self, headline):
        """
        Removes stopwords returns a lemmatized list of tokens from a headline by
        applying word/punctuation tokenization, and POS tagging.
        It uses the POS tags to look up the lemma in WordNet, and returns
        the lowercase version of all words.
        """
        # Break the headline into pos tagged tokens
        #headline = TextBlob(headline.decode('ascii',errors='replace'))
        for token, tag in headline.tags:
            token = token.strip()

            if token.isalpha() and token not in self.stopwords:
                #lemma = self.lemmatize(token, tag)
                w = Word(token.lower())
                lemma = w.lemmatize(tag)
                yield lemma
