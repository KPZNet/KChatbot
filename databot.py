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

#build_model.build()

def build_data_dictionary(wdata):
    dit = {}
    for i in wdata['intents']:
        g = i

with open("intents.json") as file:
    data = json.load(file)
    build_data_dictionary(data)

def load_pickles():
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)
    with open('intent.pickle', 'rb') as enc:
        intent = pickle.load(enc)
    with open('training_labels.pickle', 'rb') as enc:
        training_labels = pickle.load(enc)
    with open('training_sentences.pickle', 'rb') as enc:
        training_sentences = pickle.load(enc)
    with open('labels.pickle', 'rb') as enc:
        labels = pickle.load(enc)
    return lbl_encoder, tokenizer, intent, training_labels, training_sentences, labels

def chat():
    model = keras.models.load_model('chat_model')
    lbl_encoder, tokenizer, intent, training_labels, training_sentences, labels = load_pickles()

    # parameters
    max_len = 20

    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(i['responses']))

        # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,random.choice(responses))


print(Fore.YELLOW + "Welcome to KBot, a data analyist!" + Style.RESET_ALL)
chat()







