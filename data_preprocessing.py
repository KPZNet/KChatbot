
import time as time
import numpy as np
import pandas as pd
import os

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

import csv
import json
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action
from tqdm import tqdm
from nlpaug.util import Action
from nlpaug.util.file.download import DownloadUtil
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

def scrub_sentence_all(x):
    x= BeautifulSoup(x, 'html.parser').get_text()
    x= clean_text(x)
    x= expand_contractions(x)
    x = x.lower()
    x= remove_non_alphabetical_character(x)
    x= remove_single_letter(x)
    x= remove_stopwords(x)
    x= remove_by_tag(x, adjective_tag_list)
    x= lemmatize_text(x)
    return x

def scrub_sentence_min(x):
    x= BeautifulSoup(x, 'html.parser').get_text()
    x= clean_text(x)
    x= expand_contractions(x)
    return x

def scrub_sentence_mid(x):
    x= BeautifulSoup(x, 'html.parser').get_text()
    x= clean_text(x)
    x= remove_punctuation_and_number(x)
    x= expand_contractions(x)
    #x= remove_non_alphabetical_character(x)
    x= remove_single_letter(x)
    x= autocorrect(x)
    return x

def get_randos(text, numrandos, keeporig = False):
    at = []
    if keeporig:
        at.append(text)
    
    #aug = naw.SynonymAug(aug_src='wordnet')
    #aug = naf.Sequential([aug_bert,aug_w2v])

    #TOPK=20 #default=100
    #ACT = 'insert' #"substitute"
 
    #aug = naw.ContextualWordEmbsAug(
    #    model_path='distilbert-base-uncased', 
    #    #device='cuda',
    #    action=ACT, top_k=TOPK)

    aug = naw.SynonymAug(aug_src='wordnet', model_path=None, name='Synonym_Aug', aug_min=1, aug_max=10, aug_p=0.3, lang='eng', 
                     stopwords=None, tokenizer=None, reverse_tokenizer=None, stopwords_regex=None, force_reload=False, 
                     verbose=0)

    for i in range(numrandos):
        t = aug.augment(text)
        at.append(t)

    return at

def build_answer_dictionary(dfa):
    rdict = {}
    for row in dfa:
        id = row["ParentId"]
        quests = []
        if id in rdict.keys():
            quests = rdict[id]
            quests.append(row)
            rdict[id] = quests
        else:
            quests.append(row)
            rdict[id] = quests
    return rdict

def find_parent(pid, dr):
    rs = []
    if pid in dr:
        rs = dr[pid]
    return rs

def csv_to_json(cQ, cA, total_sets, augs, jsonFilePath):
    jsonArray = []

    with open(cA, encoding='utf-8') as csvfA:
        csvReaderA = csv.DictReader(csvfA)
        answersDict = build_answer_dictionary(csvReaderA)
        with open(cQ, encoding='utf-8') as csvfQ:
            csvReaderQ = csv.DictReader(csvfQ)

            irow = 0
            for row in csvReaderQ:

                if irow % 10 == 0:
                    print("Processing ROW {0} / {1} with {2} augments".format(irow, total_sets, augs))

                id = row['Id']
                rs = find_parent( id , answersDict)
                jtag = row["Title"]
                jtagscrubbed = scrub_sentence_mid(jtag)
                patterns = get_randos(jtagscrubbed, augs)
                patterns.insert(0,jtagscrubbed)
                patterns.insert(0,jtag)
                jresponses = [b['Body'] for b in rs]
                if len(jresponses) > 0:
                    jresponses_cleaned = []
                    for j in jresponses:
                        jsc = scrub_sentence_min(j)
                        jresponses_cleaned.append(jsc)
                    jrec = {'tag':id, 'patterns':patterns ,'responses':jresponses_cleaned}
                    jsonArray.append(jrec)
                    irow += 1
                if irow >= total_sets:
                    break

    #convert python jsonArray to JSON String and write to file
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jdict = {"intents":jsonArray}
        jsonString = json.dumps(jdict, indent=4)
        jsonf.write(jsonString)

def convert_qa_to_json(jsonFilePath, cQ, cA, max_sets, num_augmented_answers):
    csv_to_json(cQ, cA, max_sets, num_augmented_answers, jsonFilePath)
    print("Completed intents_qa JSON file")


def scrub_jsonfile(tfile, randos=10):
    data = None
    with open(tfile, "r") as file:
        data = json.load(file)
        
        l = len(data['intents'])
        j = 0
        for intent in data['intents']:
            if j % (l/100) == 0:
                print("Processing Intent {0} / {1}".format(j,l))
            j += 1
            
            plistscrubbed = []
            plistorig = []
            for pattern in intent['patterns']:
                plistorig.append(pattern)
                p = scrub_sentence_min(pattern)
                a = get_randos(p, randos, True) 
                plistscrubbed += a
               
            intent['patterns'] = list(set(plistscrubbed))
            intent['patterns_orig'] = plistorig

    split_tup = os.path.splitext(tfile)
    filen = split_tup[0] + '_mready.json'
    with open(filen, 'w', encoding='utf-8') as jsonf:
        jsonString = json.dumps(data, indent=4)
        jsonf.write(jsonString)


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
    df = df[:rows_to_read]
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
    df = df[:rows_to_read]
    df.info()
    return df

def preprocess_statexchangebot_csvs():
    cQ = "stats_a.csv"
    cA = "stats_q.csv"
    jsonFilePath = "StatsQA.json"
    csv_to_json(cQ, cA, 10000, 40, jsonFilePath)
    
    print("Done building clean JSON")

def post_process_statbot_json():
    scrub_jsonfile('intents_statbot.json', 10)
    print("Done processing statbot JSON")

def post_process_cmovies_json():
    scrub_jsonfile('cmovies.json', 50)
    print("Done processing CMOVIES JSON")





