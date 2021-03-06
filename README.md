## Financial Headlines Sentiment Scores

## Summary
Given the explosion of unstructured data, the influence of text analytics is increasing across various domains. When it comes to its application in finance, Sentiment Analysis lies somewhere in between technical and fundamental analysis. The goal of this capstone project is to identify the sentiment polarity in a company specific financial headline as positive (1) or negative (-1) or neutral (0) and predict end-of-day sentiment scores for over 100,000 headlines.

## Methodology

Main steps in the pipeline are as follows:
1. Obtain financial phrase bank training data
2. Get current headlines test data from an api
3. Clean and pre-process text using TextBlob and Regex to remove all years, months and numbers. Use custom stop word list to enhance the quality of text before it goes into tokenization
4. Perform POS tagging and lemmatizing for tokenization after analyzing different ways to stemming/lemming
5. Use bi-grams and tfidf for vectorization
6. Obtain performance metrics from baseline model using financial domain specific lexicon based approach
7. Obtain performance metrics using off-the-shelf sentiment analyzer
8. Experiment different machine learning models to analyze precision/recall
9. Implement Precision Bootstrapping to address the problem of high- precision and low-recall
10. Extract important features using the final model and predict sentiment scores (the average of sentiment scores (1, -1, 0) in a day)

## Precision Bootstrapping

Precision Bootstrapping is a semi-supervised self-training framework aimed to improve the performance of a supervised ML model/classifier by learning from both labeled and unlabeled data. At a high level, the method works by iteratively labelling the unlabeled instances using a trained classifier and then retraining the classifier on the expanded training dataset. Process overview is shown in the image, 'Precision_Bootstrapping.png'.

In each iteration the trained model is evaluated on held-out labels. At the end of all iterations, the classifier/model that has good balance between precision and recall as per the current use-case would be selected as the final model. Final results improved the recall by 16% with a decrease in precision by 2%.       

## Top features

Examples of top bearish/bullish features:

  <img src="Top_ftrs.png" width="300" height="150">

### References:

1. Malo, Pekka & Sinha, Ankur & Takala, Pyry & Korhonen, Pekka & Wallenius, Jyrki. (2013). FinancialPhraseBank-v1.0.
2. https://www.cs.sfu.ca/~anoop/papers/pdf/semisup_naacl.pdf
3. http://cs.stanford.edu/people/danqi/papers/tac2015.pdf
4. https://www3.nd.edu/~mcdonald/Word_Lists.html
