import time as time
import numpy as np
import pandas as pd
import gc

import re
from bs4 import BeautifulSoup
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
from nltk.tokenize import ToktokTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tag.util import untag
import contractions
# import pycontractions # Alternative better package for removing contractions
from autocorrect import Speller

start = time.time()


spell = Speller()
token = ToktokTokenizer()
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
charac = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789'
stop_words = set(stopwords.words("english"))
adjective_tag_list = set(['JJ','JJR', 'JJS', 'RBR', 'RBS']) # List of Adjective's tag from nltk package


def clean_text(text):
    text = re.sub(r"\'", "'", text) # match all literal apostrophe pattern then replace them by a single whitespace
    text = re.sub(r"\n", " ", text) # match all literal Line Feed (New line) pattern then replace them by a single whitespace
    text = re.sub(r"\xa0", " ", text) # match all literal non-breakable space pattern then replace them by a single whitespace
    text = re.sub('\s+', ' ', text) # match all one or more whitespace then replace them by a single whitespace
    text = text.strip(' ')
    return text

def expand_contractions(text):
    text = contractions.fix(text)
    return text

def autocorrect(text):
    words = token.tokenize(text)
    words_correct = [spell(w) for w in words]
    return ' '.join(map(str, words_correct))

def remove_punctuation_and_number(text):
    return text.translate(str.maketrans(" ", " ", charac))

def remove_non_alphabetical_character(text):
    text = re.sub("[^a-z]+", " ", text) # remove non-alphabetical character
    text = re.sub("\s+", " ", text) # remove whitespaces left after the last operation
    return text

def remove_single_letter(text):
    text = re.sub(r"\b\w{1}\b", "", text) # remove all single letter
    text = re.sub("\s+", " ", text) # remove whitespaces left after the last operation
    text = text.strip(" ")
    return text

def remove_stopwords(text):
    words = token.tokenize(text)
    filtered = [w for w in words if not w in stop_words]

    return ' '.join(map(str, filtered))

def stem_text(text):
    """Stem the text"""
    words = nltk.word_tokenize(text) # tokenize the text then return a list of tuple (token, nltk_tag)
    stem_text = []
    for word in words:
        stem_text.append(stemmer.stem(word)) # Stem each words
    return " ".join(stem_text) # Return the text untokenize

def lemmatize_text(text):
    """Lemmatize the text by using tag """

    tokens_tagged = nltk.pos_tag(nltk.word_tokenize(text))  # tokenize the text then return a list of tuple (token, nltk_tag)
    lemmatized_text = []
    for word, tag in tokens_tagged:
        if tag.startswith('J'):
            lemmatized_text.append(lemmatizer.lemmatize(word,'a')) # Lemmatisze adjectives. Not doing anything since we remove all adjective
        elif tag.startswith('V'):
            lemmatized_text.append(lemmatizer.lemmatize(word,'v')) # Lemmatisze verbs
        elif tag.startswith('N'):
            lemmatized_text.append(lemmatizer.lemmatize(word,'n')) # Lemmatisze nouns
        elif tag.startswith('R'):
            lemmatized_text.append(lemmatizer.lemmatize(word,'r')) # Lemmatisze adverbs
        else:
            lemmatized_text.append(lemmatizer.lemmatize(word)) # If no tags has been found, perform a non specific lemmatization
    return " ".join(lemmatized_text)

def remove_by_tag(text, undesired_tag):
    """remove all words by using ntk tag (adjectives, verbs, etc.)"""
    words = token.tokenize(text) # Tokenize each words
    words_tagged = nltk.pos_tag(tokens=words, tagset=None, lang='eng') # Tag each words and return a list of tuples (e.g. ("have", "VB"))
    filtered = [w[0] for w in words_tagged if w[1] not in undesired_tag] # Select all words that don't have the undesired tags

    return ' '.join(map(str, filtered)) # Return the text untokenize

def readinquestions(rows_to_read, date_cut):
    dtypes_questions = {'Id': 'int32', 'CreationDate': 'str', 'Score': 'int16', 'Title': 'str', 'Body': 'str'}
    df = pd.read_csv('pythonpack/questions.csv',
                               usecols=['Id', 'CreationDate', 'Score', 'Title', 'Body'],
                               encoding="ISO-8859-1",
                               dtype=dtypes_questions,
                               nrows=rows_to_read
                               )
    df[['Title', 'Body']] = df[['Title', 'Body']].applymap(
        lambda x: str(x).encode("utf-8", errors='surrogatepass').decode("ISO-8859-1", errors='surrogatepass'))
    df['CreationDate'] = pd.to_datetime(df['CreationDate'], format='%Y-%m-%d')
    df = df.loc[(df['CreationDate'] >= date_cut)]
    #df_questions = df_questions[df_questions["Score"] >= 0]
    #df = df[:2000]
    df.info()
    return df

def readinanswers(rows_to_read, date_cut):
    dtypes_answers = {'Id':'int32', 'CreationDate': 'str', 'ParentId':'int32','Score': 'int16', 'Body': 'str'}
    df = pd.read_csv('pythonpack/answers.csv',
                               usecols=['Id','CreationDate','ParentId','Score', 'Body'],
                               encoding = "ISO-8859-1",
                               dtype=dtypes_answers,
                               nrows=rows_to_read
                               )
    df[['Body']] = df[['Body']].applymap(
        lambda x: str(x).encode("utf-8", errors='surrogatepass').decode("ISO-8859-1", errors='surrogatepass'))
    df['CreationDate'] = pd.to_datetime(df['CreationDate'], format='%Y-%m-%d')
    df = df.loc[(df['CreationDate'] >= date_cut)]
    #df_questions = df_questions[df_questions["Score"] >= 0]
    #df = df[:2000]
    df.info()
    return df


def scrub_text_loop_minimal(filename, df):
    t = []
    l = len(df)
    t_percent = int(l/100)
    for index in range(len(df)):

        if index % t_percent == 0:
            print("Processing Row {0} / {1} Time {2:.4f} - {3}".format(index,l, time.time()-start, filename))

        x = df.iloc[index]
        x= BeautifulSoup(x, 'html.parser').get_text()
        x= clean_text(x)

        t.append(x)

    return t

def scrub_text_loop_all(filename, df):
    t = []
    l = len(df)
    t_percent = int(l/100)
    for index in range(len(df)):

        if index % t_percent == 0:
            print("Processing Row {0} / {1} Time {2:.4f} - {3}".format(index,l, time.time()-start, filename))

        x = df.iloc[index]
        x= BeautifulSoup(x, 'html.parser').get_text()
        x= clean_text(x)
        x= expand_contractions(x)
        x = x.lower()
        x= remove_non_alphabetical_character(x)
        #x= remove_single_letter(x)
        x= remove_stopwords(x)
        x= remove_by_tag(x, adjective_tag_list)
        x= lemmatize_text(x)
        t.append(x)

    return t

def get_scrubbed_questions(filename):
    df = readinquestions(1000000,'2016-09-01')
    df['Body_clean'] = scrub_text_loop_all(filename, df['Body'])
    df['Title_clean'] = scrub_text_loop_all(filename, df['Title'])
    df['Body'] = scrub_text_loop_minimal(filename, df['Body'])
    df['Title'] = scrub_text_loop_minimal(filename, df['Title'])
    df.to_csv(filename, encoding='utf-8', errors='surrogatepass')

def get_scrubbed_answers(filename):
    df = readinanswers(1000000,'2016-08-01')
    b = scrub_text_loop_minimal(filename, df['Body'])
    df['Body'] = b
    df.to_csv(filename, encoding='utf-8', errors='surrogatepass')

get_scrubbed_questions("clean_questions.csv")
get_scrubbed_answers("clean_answers.csv")

end = time.time()

print("Execution Time {0:.4f} seconds".format(end-start))
print("COMPLETE scrubbing")