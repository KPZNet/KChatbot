import json 
import os
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd

from statbot_lib import *

from databot import start_chat


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

    build_trainer('intents_statbot.json', model_name = 'statbotQA', vectorize=True)
    build_modeler('statbotQA', 50)
    deploy_model('statbotQA')
    print("Built!")

    start_chat()

if __name__ == "__main__":
    build_statbot()


