import os
import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
import string
from sklearn.metrics import accuracy_score, precision_score, recall_score
import nltk

'''
This module calculates sentiment scores using domain-specific lexicon
approach. Using the number of positive/negative/neutral words in a
headline based on Loughran-McDonald lexicon, the code assigns a
sentiment score for each headline
'''
def load_lexicon(filepath):
    '''Loads positive, negative and neutral word lists and stopwords
    from Loughran-McDonald financial lexicon
    INPUT: str; file path of lexicon
    OUTPUT: list, list, list, list; list of pos/neg/net/stop words
    '''
    with open(filepath+'/PositiveWords.csv', 'r') as p:
        pos_wrdlns = p.readlines()
        p_lst = [pos.strip().lower() for pos in pos_wrdlns]

    with open(filepath+'/NegativeWords.csv', 'r') as ng:
        neg_wrdlns = ng.readlines()
        n_lst = [neg.strip().lower() for neg in neg_wrdlns]

    with open(filepath+'/NeutralWords.csv', 'r') as nt:
        nut_wrdlns = nt.readlines()
        nt_lst = [nut.strip().lower() for nut in nut_wrdlns]

    with open(filepath+'/LM_stopwords.csv', 'r') as st:
        stp_wrdlns = st.readlines()
        stp_lst = [sw.strip().lower() for sw in stp_wrdlns]

    return p_lst, n_lst, nt_lst, stp_lst


def clean_headline(headline, stp_wrds):
    '''Removes punctuation, stopwords and returns lowercase text in a
    list of single words
    INPUT: str, list of stop words
    OUTPUT: list of tokens
    '''
    headline = headline.decode('ascii',errors='ignore')
    tokenizer = RegexpTokenizer(r'\w+')
    tk_hl = tokenizer.tokenize(headline)

    clean_wrds = [w.lower() for w in tk_hl if w not in stp_wrds \
                            and w.isalpha()]
    return clean_wrds

def assign_sentiment(hdln, pos, neg, neut, sw):
    ''' Takes headline, list of pos, neg, neutral and stop words
    and returns sentiment score as 1/-1/0
    INPUT: str, list, list, list, list
    OUTPUT: -1/1/0
    '''
    hl_wrds = clean_headline(hdln, sw)
    l_pos = [w for w in hl_wrds if w in pos]
    n_pos = [w for w in hl_wrds if w in neg]
    nu_pos = [w for w in hl_wrds if w in neut]

    if len(l_pos) > max(len(n_pos), len(nu_pos)):
        # Assign pos polarity if there are more +ve wrds
        return 1
    elif len(n_pos) > max(len(l_pos), len(nu_pos)):
        return -1
    else:
        return 0

if __name__ == '__main__':
    tr_df = pd.read_csv('phrase_bank_train.csv',  \
                names = ['headline', 'label'])

    # Get actual labels into an array
    y = tr_df.pop('label').tolist()

    # Load positive/negative/neutral words from Loughran-McDonald dict
    lexiconpath = os.path.dirname(os.path.realpath('data/PositiveWords.csv'))
    pos_lst, neg_lst, ntr_lst, sw_lst = load_lexicon(lexiconpath)

    # Assign sentiment scores
    tr_df['sent_score'] = tr_df['headline'].apply(lambda x: \
                    assign_sentiment(x, pos_lst, neg_lst, ntr_lst, sw_lst))

    # Get scores
    y_pred = tr_df.sent_score.tolist()
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y_true=y, y_pred=y_pred, average='weighted')
    recl = recall_score(y_true=y, y_pred=y_pred, average='weighted')

    print "Base model performance using domain-specific lexicon:"
    print "Accuracy\tPrecision\tRecall"
    print "%.4f\t\t%.4f\t\t%.4f" % (acc, prec, recl)
