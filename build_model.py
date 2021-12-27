import json 
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle
import nltk
import pandas as pd

from gensim.models import FastText
from gensim.models import KeyedVectors

from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

class creply:
    def __init__(self, resp, patt, tg):
        self.responses = resp
        self.patterns = patt
        self.tag = tg

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


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

def build_trainingdata(intents_file):
    rdict, intent, labels, num_classes, responses, training_labels, training_sentences = __readin_intensions(intents_file)
    lbl_encoder, training_labels_encoded = __label_encoder(training_labels)

    return rdict, intent, labels, num_classes, responses, training_labels, training_sentences,lbl_encoder, training_labels_encoded

def pickle_trainingdata(model_name, rdict, labels, lbl_encoder, responses, training_labels_encoded, num_classes):
    pickle_data(model_name = model_name,rdict = rdict, labels = labels, lbl_encoder = lbl_encoder,
                responses = responses, training_labels_encoded = training_labels_encoded, 
                num_classes = num_classes)
                

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

def build_trainer(intents_file, model_name):
    rdict, intent, labels, num_classes, responses, training_labels,training_sentences,lbl_encoder, training_labels_encoded = build_trainingdata(intents_file)
    pickle_trainingdata(model_name,rdict, labels, lbl_encoder, responses, training_labels_encoded, num_classes)
    vectorized_sentences = vectorize_all_sentences(training_sentences, verbose = 1)
    pickle_vectorized_sentences(model_name, vectorized_sentences)
    print("Done encoding AND pickled")

def build_modeler(model_name, epochs):
    rdict, labels, lbl_encoder, responses, training_labels_encoded, num_classes = load_pickles(model_name)
    vectorized_sentences = load_vectorized_sentences(model_name)
    
    epochs, history, model = build_vectorized_model(epochs ,num_classes,training_labels_encoded,vectorized_sentences)
    save_model_to_file(model, model_name+'NNModel')
    return vectorized_sentences


def main():
    model_name = 'pythonQA'
    intents_file = 'intents_qa.json'

    #build_trainer(intents_file, model_name = model_name)
    #build_modeler(model_name, 50)
    #deploy_model(model_name)
    print("Built!")

    start_chat(model_name)