# generic librairies
import time as time
import numpy as np
import pandas as pd
import gc

# Text librairies
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

def clean_text(text):
    text = re.sub(r"\'", "'", text) # match all literal apostrophe pattern then replace them by a single whitespace
    text = re.sub(r"\n", " ", text) # match all literal Line Feed (New line) pattern then replace them by a single whitespace
    text = re.sub(r"\xa0", " ", text) # match all literal non-breakable space pattern then replace them by a single whitespace
    text = re.sub('\s+', ' ', text) # match all one or more whitespace then replace them by a single whitespace
    text = text.strip(' ')
    return text

def expand_contractions(text):
    """expand shortened words, e.g. 'don't' to 'do not'"""
    text = contractions.fix(text)
    return text

def autocorrect(text):
    words = token.tokenize(text)
    words_correct = [spell(w) for w in words]
    return ' '.join(map(str, words_correct)) # Return the text untokenize

def remove_punctuation_and_number(text):
    """remove all punctuation and number"""
    return text.translate(str.maketrans(" ", " ", charac))

def remove_non_alphabetical_character(text):
    """remove all non-alphabetical character"""
    text = re.sub("[^a-z]+", " ", text) # remove all non-alphabetical character
    text = re.sub("\s+", " ", text) # remove whitespaces left after the last operation
    return text

def remove_single_letter(text):
    """remove single alphabetical character"""
    text = re.sub(r"\b\w{1}\b", "", text) # remove all single letter
    text = re.sub("\s+", " ", text) # remove whitespaces left after the last operation
    text = text.strip(" ")
    return text

def remove_stopwords(text):
    """remove common words in english by using nltk.corpus's list"""
    words = token.tokenize(text)
    filtered = [w for w in words if not w in stop_words]

    return ' '.join(map(str, filtered)) # Return the text untokenize

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

dtypes_questions = {'Id':'int32', 'CreationDate':'str', 'Score': 'int16', 'Title': 'str', 'Body': 'str'}

spell = Speller()
token = ToktokTokenizer()
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
charac = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789'
stop_words = set(stopwords.words("english"))
adjective_tag_list = set(['JJ','JJR', 'JJS', 'RBR', 'RBS']) # List of Adjective's tag from nltk package

start = time.time()
df_questions = pd.read_csv('pythonpack/questions.csv',
                           usecols=['Id', 'CreationDate', 'Score', 'Title', 'Body'],
                           encoding = "ISO-8859-1",
                           dtype=dtypes_questions,
                           nrows=100
                           )
df_questions[['Title', 'Body']] = df_questions[['Title', 'Body']].applymap(lambda x: str(x).encode("utf-8", errors='surrogatepass').decode("ISO-8859-1", errors='surrogatepass'))

df_questions['CreationDate'] = pd.to_datetime(df_questions['CreationDate'], format='%Y-%m-%d')
df_questions = df_questions.loc[(df_questions['CreationDate'] >= '2000-01-01')]
df_questions = df_questions[:2000]

df_questions.info()

# Remove all questions that have a negative score
#df_questions = df_questions[df_questions["Score"] >= 0]

def scrub_text_loop(df):
    t = []
    b = []
    l = len(df)
    for index in range(len(df)):

        if index % 10 == 0:
            print("Processing Row {0} / {1} Time {2:.4f}".format(index,l, time.time()-start))

        x = df['Title'].iloc[index]
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

        x = df['Body'].iloc[index]
        x= BeautifulSoup(x, 'html.parser').get_text()
        x= clean_text(x)
        x= expand_contractions(x)
        x = x.lower()
        x= remove_non_alphabetical_character(x)
        #x= remove_single_letter(x)
        x= remove_stopwords(x)
        x= remove_by_tag(x, adjective_tag_list)
        x= lemmatize_text(x)
        b.append(x)

    return b, t

b, t = scrub_text_loop(df_questions)
df_questions['Body'] = b
df_questions['Title'] = t

fileName = 'df_questions_scrubbed.csv'
df_questions['Text'] = df_questions['Title'] + ' ' + df_questions['Body']
df_questions.to_csv(fileName, encoding='utf-8', errors='surrogatepass')

end = time.time()

print("Execution Time {0:.4f} seconds".format(end-start))
print("COMPLETE scrubbing {0}".format(fileName))