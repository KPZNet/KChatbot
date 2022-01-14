import json
from logging import exception 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import matplotlib.pyplot as plot

import colorama 
colorama.init()
from colorama import Fore, Style, Back

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

from nltk.tokenize import word_tokenize  
from data_preprocessing import scrub_sentence_min

from statbot_lib import *
import click

def clrscr():
   # Clear screen using click.clear() function
    click.clear()

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

def vectorize_sent_inputs(inp):
    p = vectorize_all_sentences( [inp] )
    return p




def chat():

    sb_model = keras.models.load_model('statbotQANNModel')
    sb_rdict, sb_labels, sb_lbl_encoder, sb_responses, sb_training_labels_encoded, sb_num_classes = load_pickles('statbotQA')

    stat_model = keras.models.load_model('statsQANNModel')
    stat_rdict, stat_labels, stat_lbl_encoder, stat_responses, stat_training_labels_encoded, stat_num_classes = load_pickles('statsQA')

    lastChat = ''
    dataFile = None
    f1 = None
    fdataset = None
    fileisopen = False
    debugOn = False

    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break

        inp_scb = scrub_sentence_min(inp)
        inp_v = vectorize_sent_inputs(inp_scb)
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
                if debugOn:
                    print(Fore.LIGHTMAGENTA_EX + "\tScrubbed:" + Style.RESET_ALL , inp_scb)
                    print(Fore.LIGHTMAGENTA_EX + "\tTAG:" + Style.RESET_ALL , t)
                    print(Fore.LIGHTMAGENTA_EX + "\tNN Likelyhood:" + Style.RESET_ALL , prob)
                    print(Fore.LIGHTMAGENTA_EX + "\tMatched:" + Style.RESET_ALL , patterns[0])
                
            if tag == 'opendata':
                bm = getdatafile(inp)

                if (bm is not None) and (len(bm) > 0) :
                    #print('DATAFILE open request: MATCH [[{0}]]'.format(bm))
                    try:
                        fdataset = pd.read_csv(bm)
                        print("I opened file {0} for you".format(bm))
                        print("Here is quick summary of data\n\n")
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
            elif tag == 'mean':
                if fileisopen:
                    print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(c))
                    fdataset.plot.box(title="Box and whisker plot", grid=True)
                    plot.show()
                else:
                    print("Please open a data file first")
            elif tag == 'histogram' :
                if fileisopen:
                    print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(c))
                    fdataset.plot.hist(title="Histogram plot", grid=True)
                    plot.show()
                else:
                    print("Please open a data file first")
            elif tag == 'plot' :
                if fileisopen:
                    print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(c))
                    fdataset.plot(title="Data plot", grid=True)
                    plot.show()
                else:
                    print("Please open a data file first")
            elif tag == 'standarddeviation' :
                if fileisopen:
                    print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(c))
                    print( fdataset.std() )
                else:
                    print("Please open a data file first")
            else:           
                print(Fore.GREEN + "" + Style.RESET_ALL , np.random.choice(c))

        else:
            result2 = stat_model.predict(inp_v)
            m2 = np.argmax(result2)
            prob2 = result2[0,m2]
            if prob >= 0.75:
                tag2 = stat_lbl_encoder.inverse_transform([np.argmax(result2)])

                t2 = tag2[0]
                rd2 = stat_rdict[t2]
                c2 = rd2.responses
                patterns2 = rd2.patterns
                if debugOn:
                    print(Fore.LIGHTMAGENTA_EX + "\t\tScrubbed:" + Style.RESET_ALL , inp_scb)
                    print(Fore.LIGHTMAGENTA_EX + "\t\tTAG:" + Style.RESET_ALL , t2)
                    print(Fore.LIGHTMAGENTA_EX + "\t\tNN Likelyhood:" + Style.RESET_ALL , prob2)
                    print(Fore.LIGHTMAGENTA_EX + "\t\tMatched:" + Style.RESET_ALL , patterns2[0])
                
                print("I talked to Stat Exchange and here is what they say...\n")
                print(Fore.GREEN + "" + Style.RESET_ALL , np.random.choice(c2))
            else:
                print("I don't understand your question, can you try again please")
                                    
def start_chat():
    print(Fore.YELLOW + "Welcome to KBot data Analyst!" + Style.RESET_ALL)
    chat()

if __name__ == "__main__":
    start_chat()










