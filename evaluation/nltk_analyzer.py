import os
import pandas as pd
import numpy as np
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as sw
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import extract_unigram_feats, mark_negation

'''
This module uses off-the-shelf nltk sentiment analyzer to run
sentiment analysis and get a baseline score
'''
def run_sentiment_analysis(X_train, X_test, y_train, y_test):
    ''' Takes train/test lists, applies features and transform train
    and fits classifier on transformed train. Evaluates on test
    using this classifier
    INPUT: 4 lists
    OUTPUT: returns dictionary of scores 
    '''
    stp_wrds = sw.words('english')
    #Initialize sentiment analyzer and build vocab
    analyzer = SentimentAnalyzer()
    #Prep vocab
    vocab = analyzer.all_words([mark_negation(word_tokenize(\
                        hl.decode('ascii', errors = 'ignore'))) \
                            for hl in X_train])

    #Get unigrams list
    unigram_features = analyzer.unigram_word_feats(vocab, min_freq=10)
    analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_features)

    # Apply features and transform X_train
    X_train_map = analyzer.apply_features([mark_negation(\
                                    word_tokenize(hl.decode('ascii', errors = 'ignore'))) \
                                    for hl in X_train], labeled=False)

    # Apply features and transform X_test
    X_test_map = analyzer.apply_features([mark_negation(\
                        word_tokenize(hl.decode('ascii', errors = 'ignore'))) \
                        for hl in X_test], labeled=False)

    # Initialize NB classifier
    trainer = NaiveBayesClassifier.train
    classifier = analyzer.train(trainer, zip(X_train_map, y_train))

    score = analyzer.evaluate(zip(X_test_map, y_test))
    return score



if __name__ == '__main__':
    tr_df = pd.read_csv('phrase_bank_train.csv',  \
                names = ['headline', 'label'])

    X = tr_df['headline'].tolist()
    y = tr_df['label'].tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    # Fit data using nltk SentimentAnalyzer
    acc_score = run_sentiment_analysis(X_train, X_test, y_train, y_test)

    print "Accuracy: ", acc_score['Accuracy']
