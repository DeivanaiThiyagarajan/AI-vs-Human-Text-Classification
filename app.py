from flask import Flask, render_template, request, jsonify
import pickle as pkl

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#import all the necessary string and text related libraries
import re
import string
import textstat


#import all the nlp related libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
import os


app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__),'LogisticRegressionClassifier.pkl')   

with open(model_path, "rb") as f:
    model = pkl.load(f)     

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form.get('inputText')
    if not input_text:
        return jsonify({'error': 'No input provided'}), 400
    
    def average_sentence_length(text):
        sentence_endings = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s"
        sentences = re.split(sentence_endings, text)
        lengths = [len(words.split()) for words in sentences]
        avg = sum(lengths)/len(sentences)
        return avg
    def word_count(text):
        return len(text.split())
    def punctuation_count(text):
        punctuation_pattern = r"[.,!?;:'\"()\[\]{}]"
        matches = re.findall(punctuation_pattern, text)
        return len(matches)
    def readability_score(text):
        return textstat.flesch_reading_ease(text)
    
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    # Function to find the stop words ratio to the original number of words
    def stop_words_ratio(text):
        stop_count = 0
        new = text.split()
        for i in new:
            if(i in stop_words):
                stop_count+=1
        return stop_count/len(new) if len(new)>0 else 0
    
    def lower_case(text):
        return text.lower()
    
    def remove_stopwords(text):
        for i in stop_words:
            text.replace(i,'')
        return text
    
    def remove_tags(text):
        tags = ['\n', '\'']
        for tag in tags:
            text = text.replace(tag, '')
        return text
    
    def remove_punc(text):
        new_text = [x for x in text if x not in string.punctuation]
        new_text = ''.join(new_text)
        return new_text
    

    #Apply the function to the 'text' column
    avg_len_sentences = average_sentence_length(input_text)
    word_cnt = word_count(input_text)
    punc_cnt = punctuation_count(input_text)
    score = readability_score(input_text)
    stop_word_ratio = stop_words_ratio(input_text)
    
    
    input_text = remove_punc(remove_tags(remove_stopwords(lower_case(input_text))))

    input_data = {'text': input_text, 'avg_len_sentences': avg_len_sentences, 'punctuations_count': punc_cnt,
                   'readability_score': score, 'words_count': word_cnt, 'stop_word_ratio': stop_word_ratio}
    
    df = pd.DataFrame(input_data, index=[0])
    print(df)

    # Call the detector's prediction method (you'll implement this later)
    prediction = model.predict(df)
    print(prediction)

    if(prediction == 0.0):
        result = 'This is likely to be a Human Generated Text'
    else:
        result = 'This is likely to be an AI- Genrated Text'

    return jsonify({"result": str(result)})


if __name__ == '__main__':
    app.run(debug=True)
