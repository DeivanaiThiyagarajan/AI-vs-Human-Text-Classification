import pandas as pd

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import pickle as pkl
import gzip

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

import spacy
import string

# from gensim.models import Word2Vec
# import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

from sklearn.feature_extraction.text import CountVectorizer


class TextDetector:
    def __init__(self):
        # Initialize your machine learning model here
        # Get the list of stopwords
        self.stop_words = set(stopwords.words('english'))
        self.model = self.load_model()

    def load_model(self):
        with open(r"model.pkl", "rb") as model_file:
            model = pkl.load(model_file)

        return model

    def average_sentence_length(self, text):
        sentence_endings = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s"
        sentences = re.split(sentence_endings, text)
        lengths = [len(words.split()) for words in sentences]
        avg = sum(lengths) / len(sentences)
        return avg

    def word_count(self, text):
        return len(text.split())

    def punctuation_count(self, text):
        punctuation_pattern = r"[.,!?;:'\"()\[\]{}]"
        matches = re.findall(punctuation_pattern, text)
        return len(matches)

    def lower_case(self, text):
        return text.lower()

    def tokenize(self, text):
        tokens = word_tokenize(text)
        return tokens

    # Function to remove stop words
    def remove_stopwords(self, tokens):
        return [word for word in tokens if word not in self.stop_words]

    def compose_input_df(self, text):
        input_df = pd.DataFrame({'text': [text]})
        # print(input_df)
        input_df['avg_len_sentences'] = input_df['text'].apply(self.average_sentence_length)
        input_df['words_count'] = input_df['text'].apply(self.word_count)
        input_df['punctuations_count'] = input_df['text'].apply(self.punctuation_count)
        input_df['text'] = input_df['text'].apply(self.lower_case)
        input_df['tokens'] = input_df['text'].apply(self.tokenize)
        input_df['tokens'] = input_df['tokens'].apply(self.remove_stopwords)

        # Initialize CountVectorizer
        vectorizer = CountVectorizer(max_features=1000)

        # Fit and transform the tokenized text (joined back into strings)
        vectorized_matrix = vectorizer.fit_transform([" ".join(tokens) for tokens in input_df['tokens']])

        vectorized_matrix = vectorized_matrix.astype(np.int32)

        # Convert to DataFrame
        vectorized_df = pd.DataFrame(
            vectorized_matrix.toarray(),  # Convert sparse matrix to dense
            columns=vectorizer.get_feature_names_out()  # Get feature (word) names
        )

        # Concatenate the vectorized features with the original DataFrame
        input_df = pd.concat([input_df, vectorized_df], axis=1)

        return input_df

    def predict(self, text):
        # Dummy logic for now - replace with actual ML model prediction
        # if "AI" in text:
        #     return "This text is likely written by AI."
        # else:
        #     return "This text is likely written by a human."

        input_df = self.compose_input_df(text)
        print(input_df)

        X = input_df.iloc[:, 2:]
        X = X.loc[:, X.columns != 'tokens']

        input_predictions = self.model.predict(X)

        if input_predictions[0] == 0:
            return "This text is likely written by a human."
        else:
            return "This text is likely written by AI."
