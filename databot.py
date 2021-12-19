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
from build_model import vectorize_input
from build_model import load_pickles

def build_data_dictionary(wdata):
    dit = {}
    for i in wdata['intents']:
        g = i

with open("intents.json") as file:
    data = json.load(file)
    build_data_dictionary(data)

def vectorize_input(inp):
    p = build_model.vectorize_input( inp )
    return p

def chat():
    model = keras.models.load_model('chat_model')
    rdict, labels, lbl_encoder, responses, training_labels_encoded, num_classes, max_len = load_pickles()

    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break

        inp_v = vectorize_input(inp)
        result = model.predict(inp_v)
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        t = tag[0]
        c = rdict[t]
        print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(c))


print(Fore.YELLOW + "Welcome to KBot, a data analyist!" + Style.RESET_ALL)
chat()









