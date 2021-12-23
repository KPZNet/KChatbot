import json 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

import colorama 
colorama.init()
from colorama import Fore, Style, Back

import random
import pickle

import build_model
from build_model import vectorize_all_sentences
from build_model import load_pickles

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

from nltk.tokenize import word_tokenize  

def compare(p1, p2):
    u = sbert_model.encode(p1)[0]
    v = sbert_model.encode(p2)[0]
    d = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return d

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
    return res

def vectorize_input_pythonqa(inp):
    p = build_model.vectorize_all_sentences( [inp] )
    return p

lastChat = ''
dataFile = None

def chat(model_name):
    mdir = model_name+'ChatModel\\'
    model = keras.models.load_model(mdir+model_name+'NNModel')
    rdict, labels, lbl_encoder, responses, training_labels_encoded, num_classes = load_pickles(mdir+model_name)

    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break

        inp_scb = scrub_sentence_min(inp)
        inp_v = vectorize_input_pythonqa(inp_scb)
        result = model.predict(inp_v)
        m = np.argmax(result)
        prob = result[0,m]
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        t = tag[0]
        rd = rdict[t]
        c = rd.responses
        patterns = rd.patterns
        if len(c) == 0:
            print("chatbot does not understand")
        else:

            if tag == 'opendata':
                bm = getdatafile(inp)

                if bm is not None:
                    print('DATAFILE open request: MATCH [[{0}]]'.format(bm))

            print(Fore.LIGHTMAGENTA_EX + "\tScrubbed:" + Style.RESET_ALL , inp_scb)
            print(Fore.LIGHTMAGENTA_EX + "\tTAG:" + Style.RESET_ALL , t)
            print(Fore.LIGHTMAGENTA_EX + "\tPropability:" + Style.RESET_ALL , prob)
            print(Fore.LIGHTMAGENTA_EX + "\tMatched:" + Style.RESET_ALL , patterns[0])
            print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(c))


def start_chat(model_name):
    print(Fore.YELLOW + "Welcome to KBot data Analyst!" + Style.RESET_ALL)
    chat(model_name)









