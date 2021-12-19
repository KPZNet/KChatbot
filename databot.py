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

#build_model.build()

def build_data_dictionary(wdata):
    dit = {}
    for i in wdata['intents']:
        g = i

with open("intents.json") as file:
    data = json.load(file)
    build_data_dictionary(data)

def load_pickles():

    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)
    with open('intent.pickle', 'rb') as enc:
        intent = pickle.load(enc)
    with open('training_labels.pickle', 'rb') as enc:
        training_labels = pickle.load(enc)
    with open('labels.pickle', 'rb') as enc:
        labels = pickle.load(enc)
    return lbl_encoder, tokenizer, intent, training_labels, training_sentences, labels

def vectorize_input(inp):
    p = build_model.vectorize_input(inp)
    return p

def chat():
    model = keras.models.load_model('chat_model')
    lbl_encoder, intent, training_labels, training_sentences, labels = load_pickles()

    # parameters
    max_len = 20

    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break

        inp_v = vectorize_input(inp)
        result = model.predict(inp, truncating='post', maxlen=max_len)
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(i['responses']))

        # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,random.choice(responses))


print(Fore.YELLOW + "Welcome to KBot, a data analyist!" + Style.RESET_ALL)
chat()







