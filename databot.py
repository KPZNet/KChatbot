import json
from logging import exception 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

import colorama 
colorama.init()
from colorama import Fore, Style, Back

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

from nltk.tokenize import word_tokenize  
from data_preprocessing import scrub_sentence_min

from statbot_lib import *

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
fileisopen = False


def chat():

    sb_model = keras.models.load_model('statbotQAChatModel\\statbotQANNModel')
    sb_rdict, sb_labels, sb_lbl_encoder, sb_responses, sb_training_labels_encoded, sb_num_classes = load_pickles('statbotQAChatModel\\statbotQA')

    stat_model = keras.models.load_model('statsQAChatModel\\statsQANNModel')
    stat_rdict, stat_labels, stat_lbl_encoder, stat_responses, stat_training_labels_encoded, stat_num_classes = load_pickles('statsQAChatModel\\statsQA')


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
        tag = sb_lbl_encoder.inverse_transform([np.argmax(result)])

        t = tag[0]
        rd = sb_rdict[t]
        c = rd.responses
        patterns = rd.patterns
        if len(c) == 0:
            print("chatbot does not understand")
        else:
            print(Fore.LIGHTMAGENTA_EX + "\tScrubbed:" + Style.RESET_ALL , inp_scb)
            print(Fore.LIGHTMAGENTA_EX + "\tTAG:" + Style.RESET_ALL , t)
            print(Fore.LIGHTMAGENTA_EX + "\tNN Likelyhood:" + Style.RESET_ALL , prob)
            print(Fore.LIGHTMAGENTA_EX + "\tMatched:" + Style.RESET_ALL , patterns[0])
            print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(c))

            if tag == 'opendata':
                bm = getdatafile(inp)

                if bm is not None:
                    print('DATAFILE open request: MATCH [[{0}]]'.format(bm))
                    try:
                        f1 = open(bm)
                        print("I opened the file {0} for you".format(bm))
                        fileisopen = True
                    except FileNotFoundError:
                        print("I could not find that file, can you please try again")
                    except Exception as e:
                        print("Could NOT open file, not sure why, here was exception {0}".format(e))
                else:
                    print("What file would you like to open?")
            if tag == 'mean':
                if fileisopen:
                    print("mean is:XXX.XXX")
                        
                
                    
                    


def start_chat():
    print(Fore.YELLOW + "Welcome to KBot data Analyst!" + Style.RESET_ALL)
    chat()

if __name__ == "__main__":
    start_chat()










