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


def vectorize_input_pythonqa(inp):
    p = build_model.vectorize_all_sentences( [inp] )
    return p

def chat(model_name):
    mdir = model_name+'ChatModel\\'
    model = keras.models.load_model(mdir+model_name+'NNModel')
    rdict, labels, lbl_encoder, responses, training_labels_encoded, num_classes = load_pickles(mdir+model_name)

    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break

        inp_v = vectorize_input_pythonqa(inp)
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
            print(Fore.LIGHTMAGENTA_EX + "\tTAG:" + Style.RESET_ALL , t)
            print(Fore.LIGHTMAGENTA_EX + "\tPropability:" + Style.RESET_ALL , prob)
            print(Fore.LIGHTMAGENTA_EX + "\tMatched:" + Style.RESET_ALL , patterns[0])
            print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(c))


def start_chat(model_name):
    print(Fore.YELLOW + "Welcome to KBot data Analyst!" + Style.RESET_ALL)
    chat(model_name)









