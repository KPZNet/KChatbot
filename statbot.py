import pickle

import colorama
import matplotlib.pyplot as plot
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

colorama.init()
from colorama import Fore, Style

from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

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
from nltk.corpus import stopwords
import contractions
# import pycontractions # Alternative better package for removing contractions
from autocorrect import Speller

import csv
import json
import nlpaug.augmenter.word as naw

spell = Speller()
token = ToktokTokenizer()
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
charac = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789'
stop_words = set(stopwords.words("english"))
adjective_tag_list = set(['JJ','JJR', 'JJS', 'RBR', 'RBS']) # List of Adjective's tag from nltk package


from nltk.tokenize import word_tokenize  

class creply:
    def __init__(self, resp, patt, tg):
        self.responses = resp
        self.patterns = patt
        self.tag = tg

def deploy_model(model_name):
    mdir = model_name+'ChatModel\\'
    
    fstr = 'rmdir /s /q '+mdir
    os.system(fstr)

    fstr = 'mkdir '+mdir
    os.system(fstr)

    filename = model_name+'_vectorized_sentences.pickle'
    fstr = 'copy {0} {1}'.format(filename, mdir)
    os.system(fstr)

    filename = model_name+'_rdict.pickle'
    fstr = 'copy {0} {1}'.format(filename, mdir)
    os.system(fstr)

    filename = model_name+'_label_encoder.pickle'
    fstr = 'copy {0} {1}'.format(filename, mdir)
    os.system(fstr)

    filename = model_name+'_labels.pickle'
    fstr = 'copy {0} {1}'.format(filename, mdir)
    os.system(fstr)

    filename = model_name+'_num_classes.pickle'
    fstr = 'copy {0} {1}'.format(filename, mdir)
    os.system(fstr)

    filename = model_name+'_responses.pickle'
    fstr = 'copy {0} {1}'.format(filename, mdir)
    os.system(fstr)

    filename = model_name+'_training_labels_encoded.pickle'
    fstr = 'copy {0} {1}'.format(filename, mdir)
    os.system(fstr)

    filename = model_name+'NNModel'
    mdirm = mdir+'\\'+filename
    fstr = 'mkdir '+mdirm
    os.system(fstr)
    fstr = 'xcopy /E /H /Y {0} {1}'.format(filename, mdirm)
    os.system(fstr)

    print('Model Deployed')


def pickle_data(model_name, rdict, labels, lbl_encoder, responses, training_labels_encoded, num_classes):

    with open(model_name+'_'+ 'rdict.pickle', 'wb') as ecn_file:
        pickle.dump(rdict, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(model_name+'_'+ 'labels.pickle', 'wb') as ecn_file:
        pickle.dump(labels, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(model_name+'_'+ 'label_encoder.pickle', 'wb') as ecn_file:
        pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(model_name+'_'+ 'responses.pickle', 'wb') as ecn_file:
        pickle.dump(responses, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
   
    with open(model_name+'_'+ 'training_labels_encoded.pickle', 'wb') as ecn_file:
        pickle.dump(training_labels_encoded, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
   
    with open(model_name+'_'+ 'num_classes.pickle', 'wb') as ecn_file:
        pickle.dump(num_classes, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickles(model_name):

    with open(model_name+'_'+ 'rdict.pickle', 'rb') as enc:
        rdict = pickle.load(enc)

    with open(model_name+'_'+ 'labels.pickle', 'rb') as enc:
        labels = pickle.load(enc)
 
    with open(model_name+'_'+ 'label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    with open(model_name+'_'+ 'responses.pickle', 'rb') as enc:
        responses = pickle.load(enc)

    with open(model_name+'_'+ 'training_labels_encoded.pickle', 'rb') as enc:
        training_labels_encoded = pickle.load(enc)

    with open(model_name+'_'+ 'num_classes.pickle', 'rb') as enc:
        num_classes = pickle.load(enc)

    return rdict, labels, lbl_encoder, responses, training_labels_encoded, num_classes

def convert_to_ndarr(ps, verbose = 0):
    max_len = len( ps[0] )
    l = len(ps)
    array_2d = np.ndarray((len(ps), max_len))
    
    for x in range(0, array_2d.shape[0]):
        if verbose > 0:
            if x % 100 == 0:
                print("Converted {0}/{1}".format(x, l))
    
        for y in range(0, array_2d.shape[1]):
            array_2d[x][y] = ps[x][y]
    
    ps = array_2d
    return ps

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def compare(p1, p2):
    u = sbert_model.encode(p1)[0]
    v = sbert_model.encode(p2)[0]
    d = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return d

def vectorize_input(inp):
    ps = []
    p = sbert_model.encode([inp])[0]
    ps.append(p)
    ps = convert_to_ndarr(ps)
    return ps

def vectorize_all_sentences(training_sentences, verbose = 0):
    l = len(training_sentences)
    i = 0
    ps = []

    for s in training_sentences:
        if verbose > 0:
            if i % 100 == 0:
                print("Vectorized {0} / {1} sentences".format(i, l))
            i += 1

        p = sbert_model.encode([s])[0]
        ps.append(p)

    ps = convert_to_ndarr(ps)

    return ps


def __readin_intensions(tfile):
    with open(tfile) as file:
        data = json.load(file)
        
    training_sentences = []
    training_labels = []
    labels = []
    responses = []

    rdict = {}
    
    for intent in data['intents']:
        for pattern in intent['patterns']:
            training_sentences.append(pattern)

            training_labels.append(intent['tag'])
        responses.append(intent['responses'])

        rpl = creply(intent['responses'], intent['patterns'], intent['tag'] )
        rdict[intent['tag']] = rpl
        
        if intent['tag'] not in labels:
            labels.append(intent['tag'])
                      
    num_classes = len(labels)
    return rdict, intent, labels, num_classes, responses, training_labels, training_sentences

def __label_encoder(training_labels):
    lbl_encoder = LabelEncoder()
    lbl_encoder.fit(training_labels)
    training_labels_encoded = lbl_encoder.transform(training_labels)
    return lbl_encoder, training_labels_encoded

def pickle_vectorized_sentences(model_name, sentences):
    with open(model_name+'_'+ 'vectorized_sentences.pickle', 'wb') as handle:
        pickle.dump(sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_vectorized_sentences(model_name):
    with open(model_name+'_'+ 'vectorized_sentences.pickle', 'rb') as handle:
        sentences = pickle.load(handle)
    return sentences

def build_vectorized_model(epochs,num_classes, training_labels_encoded, vectorized_sentences):
    epochs = epochs
    max_len = vectorized_sentences.shape[1]
    model = Sequential()
    model.add(Dense(16, input_dim=max_len))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.summary()

    history = model.fit(vectorized_sentences, np.array(training_labels_encoded), epochs=epochs, verbose=1)

    return epochs, history, model

def save_model_to_file(model, model_name):
    model.save(model_name)


def build_trainingdata(intents_file):
    rdict, intent, labels, num_classes, responses, training_labels, training_sentences = __readin_intensions(intents_file)
    lbl_encoder, training_labels_encoded = __label_encoder(training_labels)

    return rdict, intent, labels, num_classes, responses, training_labels, training_sentences,lbl_encoder, training_labels_encoded

def pickle_trainingdata(model_name, rdict, labels, lbl_encoder, responses, training_labels_encoded, num_classes):
    pickle_data(model_name = model_name,rdict = rdict, labels = labels, lbl_encoder = lbl_encoder,
                responses = responses, training_labels_encoded = training_labels_encoded, 
                num_classes = num_classes)
                

def build_training_data(intents_file, model_name, vectorize = False):
    rdict, intent, labels, num_classes, responses, training_labels,training_sentences,lbl_encoder, training_labels_encoded = build_trainingdata(intents_file)
    pickle_trainingdata(model_name,rdict, labels, lbl_encoder, responses, training_labels_encoded, num_classes)
    if vectorize == True:
        vectorized_sentences = vectorize_all_sentences(training_sentences, verbose = 1)
        pickle_vectorized_sentences(model_name, vectorized_sentences)
    print("Done encoding AND pickled")

def build_modeler(model_name, epochs):
    rdict, labels, lbl_encoder, responses, training_labels_encoded, num_classes = load_pickles(model_name)
    vectorized_sentences = load_vectorized_sentences(model_name)
    
    epochs, history, model = build_vectorized_model(epochs ,num_classes,training_labels_encoded,vectorized_sentences)
    save_model_to_file(model, model_name+'NNModel')
    return vectorized_sentences


def build_trainer(intents_file, model_name, vectorize = False):
    rdict, intent, labels, num_classes, responses, training_labels,training_sentences,lbl_encoder, training_labels_encoded = build_trainingdata(intents_file)
    pickle_trainingdata(model_name,rdict, labels, lbl_encoder, responses, training_labels_encoded, num_classes)
    if vectorize:
        vectorized_sentences = vectorize_all_sentences(training_sentences, verbose = 1)
        pickle_vectorized_sentences(model_name, vectorized_sentences)
    print("Done encoding AND pickled")

def build_modeler(model_name, epochs):
    rdict, labels, lbl_encoder, responses, training_labels_encoded, num_classes = load_pickles(model_name)
    vectorized_sentences = load_vectorized_sentences(model_name)
    
    epochs, history, model = build_vectorized_model(epochs ,num_classes,training_labels_encoded,vectorized_sentences)
    save_model_to_file(model, model_name+'NNModel')
    return vectorized_sentences

def build_statbot():
    model_name = 'statbotQA'
    build_trainer('intents_statbot_mready.json', model_name, vectorize=True)
    build_modeler(model_name, 50)
    deploy_model(model_name)
    print("Built!")

def build_cmovies():
    model_name = 'cmovies'
    build_trainer('cmovies_mready.json', model_name, vectorize=True)
    build_modeler(model_name, 50)
    deploy_model(model_name)
    print("Built!")

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
    scrub_jsonfile('intents_statbot.json', 1)
    print("Done processing statbot JSON")

def post_process_cmovies_json():
    scrub_jsonfile('cmovies.json', 50)
    print("Done processing CMOVIES JSON")


def findwordmatch(tsentence, baseword):
    p = -10.0
    bestguess = ''
    for s in tsentence:
        dprod = compare([baseword], [s])
        pnew = max(dprod, p)
        if( pnew > p):
            bestguess = s
            p = pnew
    return bestguess

def getdatafile(sentence):
    tsent = word_tokenize(sentence)
    subs = '.csv'
    res = list(filter(lambda x: subs in x, tsent))
    if len(res) == 0:
        subs = '.dat'
        res = list(filter(lambda x: subs in x, tsent))
        
    if res is not None:
        if len(res) >0:
            res = res[0]
    return res

def vectorize_input_pythonqa(inp):
    p = vectorize_all_sentences( [inp] )
    return p

lastChat = ''
dataFile = None
f1 = None
fdataset = None
fileisopen = False


def chat():

    sb_model = keras.models.load_model('statbotQAChatModel\\statbotQANNModel')
    sb_rdict, sb_labels, sb_lbl_encoder, sb_responses, sb_training_labels_encoded, sb_num_classes = load_pickles('statbotQAChatModel\\statbotQA')

    stat_model = keras.models.load_model('statsQAChatModel\\statsQANNModel')
    stat_rdict, stat_labels, stat_lbl_encoder, stat_responses, stat_training_labels_encoded, stat_num_classes = load_pickles('statsQAChatModel\\statsQA')

    #stat_model = keras.models.load_model('cmoviesChatModel\\cmoviesNNModel')
    #stat_rdict, stat_labels, stat_lbl_encoder, stat_responses, stat_training_labels_encoded, stat_num_classes = load_pickles('cmoviesChatModel\\cmovies')


    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break

        inp_scb = scrub_sentence_min(inp)
        inp_v = vectorize_input_pythonqa(inp_scb)
        result = sb_model.predict(inp_v)
        m = np.argmax(result)
        prob = result[0,m]

        if prob >= 0.75:
            tag = sb_lbl_encoder.inverse_transform([np.argmax(result)])
            t = tag[0]
            rd = sb_rdict[t]
            c = rd.responses
            patterns = rd.patterns
            if len(c) == 0:
                print("I'm sorry, I don't understand, say that again please?")
            else:
                print(Fore.LIGHTMAGENTA_EX + "\tScrubbed:" + Style.RESET_ALL , inp_scb)
                print(Fore.LIGHTMAGENTA_EX + "\tTAG:" + Style.RESET_ALL , t)
                print(Fore.LIGHTMAGENTA_EX + "\tNN Likelyhood:" + Style.RESET_ALL , prob)
                print(Fore.LIGHTMAGENTA_EX + "\tMatched:" + Style.RESET_ALL , patterns[0])
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(c))


            if tag == 'opendata':
                bm = getdatafile(inp)

                if (bm is not None) and (len(bm) > 0) :
                    print('DATAFILE open request: MATCH [[{0}]]'.format(bm))
                    try:
                        fdataset = pd.read_csv(bm)
                        print("I opened the file {0} for you".format(bm))
                        print("Here is quick summary and view of the data\n\n")
                        print('---------------')
                        print(fdataset.head())
                        print('---------------')
                        fileisopen = True
                    except FileNotFoundError:
                        print("I could not find that file, can you please try again")
                    except Exception as e:
                        print("Could NOT open file, not sure why, here was exception {0}".format(e))
                else:
                    print("...but please let me know the file name, say it again?")
            if tag == 'mean':
                if fileisopen:
                    fdataset.plot.box(title="Box and whisker plot", grid=True)
                    plot.show()
            if tag == 'histogram':
                if fileisopen:
                    fdataset.plot.hist(title="Histogram plot", grid=True)
                    plot.show()
            if tag == 'plot':
                if fileisopen:
                    fdataset.plot(title="Data plot", grid=True)
                    plot.show()
            if tag == 'standarddeviation':
                if fileisopen:
                    print( fdataset.std() )

        else:
            print("I searched Stats Exchange to find an answer for you, and here is what I found...")
            print('')
            result2 = stat_model.predict(inp_v)
            m2 = np.argmax(result2)
            prob2 = result2[0,m2]
            tag2 = stat_lbl_encoder.inverse_transform([np.argmax(result2)])

            t2 = tag2[0]
            rd2 = stat_rdict[t2]
            c2 = rd2.responses
            patterns2 = rd2.patterns
            print(Fore.LIGHTMAGENTA_EX + "\t\tScrubbed:" + Style.RESET_ALL , inp_scb)
            print(Fore.LIGHTMAGENTA_EX + "\t\tTAG:" + Style.RESET_ALL , t2)
            print(Fore.LIGHTMAGENTA_EX + "\t\tNN Likelyhood:" + Style.RESET_ALL , prob2)
            print(Fore.LIGHTMAGENTA_EX + "\t\tMatched:" + Style.RESET_ALL , patterns2[0])
            print(Fore.GREEN + "\t\tChatBot:" + Style.RESET_ALL , np.random.choice(c2))
                                    
def start_chat():
    print(Fore.YELLOW + "Welcome to KBot data Analyst!" + Style.RESET_ALL)
    chat()

if __name__ == "__main__":
    start_chat()











