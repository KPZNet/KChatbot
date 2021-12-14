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

dtypes_questions = {'Id':'int32', 'Score': 'int16', 'Title': 'str', 'Body': 'str'}

df_questions = pd.read_csv('pythonpack/questions.csv',
                           usecols=['Id', 'Score', 'Title', 'Body'],
                           encoding = "ISO-8859-1",
                           dtype=dtypes_questions,
                           nrows=10000
                           )

df_questions[['Title', 'Body']] = df_questions[['Title', 'Body']].applymap(lambda x: str(x).encode("utf-8", errors='surrogatepass').decode("ISO-8859-1", errors='surrogatepass'))

# Remove all questions that have a negative score
#df_questions = df_questions[df_questions["Score"] >= 0]

spell = Speller()
token = ToktokTokenizer()
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
charac = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789'
stop_words = set(stopwords.words("english"))
adjective_tag_list = set(['JJ','JJR', 'JJS', 'RBR', 'RBS']) # List of Adjective's tag from nltk package

df_questions.info()

df_questions['Body'][11]

# Parse question and title then return only the text
df_questions['Body'] = df_questions['Body'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
df_questions['Title'] = df_questions['Title'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())

df_questions['Body'][11]

def clean_text(text):
    text = re.sub(r"\'", "'", text) # match all literal apostrophe pattern then replace them by a single whitespace
    text = re.sub(r"\n", " ", text) # match all literal Line Feed (New line) pattern then replace them by a single whitespace
    text = re.sub(r"\xa0", " ", text) # match all literal non-breakable space pattern then replace them by a single whitespace
    text = re.sub('\s+', ' ', text) # match all one or more whitespace then replace them by a single whitespace
    text = text.strip(' ')
    return text


df_questions['Title'] = df_questions['Title'].apply(lambda x: clean_text(x))
df_questions['Body'] = df_questions['Body'].apply(lambda x: clean_text(x))

df_questions['Body'][11]

def expand_contractions(text):
    """expand shortened words, e.g. 'don't' to 'do not'"""
    text = contractions.fix(text)
    return text


df_questions['Title'] = df_questions['Title'].apply(lambda x: expand_contractions(x))
df_questions['Body'] = df_questions['Body'].apply(lambda x: expand_contractions(x))


df_questions['Body'][11]


def autocorrect(text):
    words = token.tokenize(text)
    words_correct = [spell(w) for w in words]
    return ' '.join(map(str, words_correct)) # Return the text untokenize

df_questions['Title'] = df_questions['Title'].str.lower()
df_questions['Body'] = df_questions['Body'].str.lower()

df_questions['Body'][11]

def remove_punctuation_and_number(text):
    """remove all punctuation and number"""
    return text.translate(str.maketrans(" ", " ", charac))



def remove_non_alphabetical_character(text):
    """remove all non-alphabetical character"""
    text = re.sub("[^a-z]+", " ", text) # remove all non-alphabetical character
    text = re.sub("\s+", " ", text) # remove whitespaces left after the last operation
    return text


df_questions['Title'] = df_questions['Title'].apply(lambda x: remove_non_alphabetical_character(x))
df_questions['Body'] = df_questions['Body'].apply(lambda x: remove_non_alphabetical_character(x))

df_questions['Body'][11]

def remove_single_letter(text):
    """remove single alphabetical character"""
    text = re.sub(r"\b\w{1}\b", "", text) # remove all single letter
    text = re.sub("\s+", " ", text) # remove whitespaces left after the last operation
    text = text.strip(" ")
    return text

df_questions['Title'] = df_questions['Title'].apply(lambda x: remove_single_letter(x))
df_questions['Body'] = df_questions['Body'].apply(lambda x: remove_single_letter(x))

df_questions['Body'][11]

def remove_stopwords(text):
    """remove common words in english by using nltk.corpus's list"""
    words = token.tokenize(text)
    filtered = [w for w in words if not w in stop_words]

    return ' '.join(map(str, filtered)) # Return the text untokenize


df_questions['Title'] = df_questions['Title'].apply(lambda x: remove_stopwords(x))
df_questions['Body'] = df_questions['Body'].apply(lambda x: remove_stopwords(x))

df_questions['Body'][11]

def remove_by_tag(text, undesired_tag):
    """remove all words by using ntk tag (adjectives, verbs, etc.)"""
    words = token.tokenize(text) # Tokenize each words
    words_tagged = nltk.pos_tag(tokens=words, tagset=None, lang='eng') # Tag each words and return a list of tuples (e.g. ("have", "VB"))
    filtered = [w[0] for w in words_tagged if w[1] not in undesired_tag] # Select all words that don't have the undesired tags

    return ' '.join(map(str, filtered)) # Return the text untokenize

df_questions['Title'] = df_questions['Title'].apply(lambda x: remove_by_tag(x, adjective_tag_list))
df_questions['Body'] = df_questions['Body'].apply(lambda x: remove_by_tag(x, adjective_tag_list))


df_questions['Body'][11]


words = ["program", "programs", "programer", "programing", "programers"]

for w in words:
    print(w, " : ", stemmer.stem(w))


def stem_text(text):
    """Stem the text"""
    words = nltk.word_tokenize(text) # tokenize the text then return a list of tuple (token, nltk_tag)
    stem_text = []
    for word in words:
        stem_text.append(stemmer.stem(word)) # Stem each words
    return " ".join(stem_text) # Return the text untokenize

print(lemmatizer.lemmatize("stripes", "v"))
print(lemmatizer.lemmatize("stripes", "n"))
print(lemmatizer.lemmatize("are"))
print(lemmatizer.lemmatize("are", "v"))


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

df_questions['Title'] = df_questions['Title'].apply(lambda x: lemmatize_text(x))
df_questions['Body'] = df_questions['Body'].apply(lambda x: lemmatize_text(x))

df_questions['Body'][11]

df_questions['Text'] = df_questions['Title'] + ' ' + df_questions['Body']


df_questions.to_csv('df_questions_fullclean.csv', encoding='utf-8', errors='surrogatepass')

print("COMPLETE")