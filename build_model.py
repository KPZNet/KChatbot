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

intents_file = 'intents.json'

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
        rdict[intent['tag']] = intent['responses']
        
        if intent['tag'] not in labels:
            labels.append(intent['tag'])
            
            
    num_classes = len(labels)
    return rdict, intent, labels, num_classes, responses, training_labels, training_sentences

def __label_encoder(training_labels):
    lbl_encoder = LabelEncoder()
    lbl_encoder.fit(training_labels)
    training_labels_encoded = lbl_encoder.transform(training_labels)
    return lbl_encoder, training_labels_encoded

def __tokenize_vobabulary(training_sentences):
    vocab_size = 1000
    embedding_dim = 300
    max_len = 20
    oov_token = "<OOV>"
    
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(training_sentences)
    #padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)
    padded_sequences = pad_sequences(sequences, truncating='post')

    recs = padded_sequences.shape[0]
    max_len = padded_sequences.shape[1]
    vocab_size = len(word_index) + 1

    print("Number of Query Records = {0}".format(recs))
    print("Max sentence vector length = {0}".format(max_len))
    print("Number of Words = {0}".format(vocab_size))

    return padded_sequences, tokenizer


def pickle_vectorized_sentences(sentences):
    with open('vectorized_sentences.pickle', 'wb') as handle:
        pickle.dump(sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_vectorized_sentences():
    with open('vectorized_sentences.pickle', 'rb') as handle:
        sentences = pickle.load(handle)
    return sentences

def convert_to_ndarr(ps):
    max_len = len( ps[0] )
    l = len(ps)
    array_2d = np.ndarray((len(ps), max_len))
    
    for x in range(0, array_2d.shape[0]):
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

def __tokenize_vobabulary_2(training_sentences):
    max_len = 0
    oov_token = "<OOV>"

    l = len(training_sentences)
    i = 0
    ps = []

    for s in training_sentences:
        if i % 10 == 0:
            print("Vectorized {0} / {1} sentences".format(i, l))
        i += 1

        p = sbert_model.encode([s])[0]
        ps.append(p)

    ps = convert_to_ndarr(ps)

    return ps


def __build_vectorized_model(num_classes,padded_sequences,training_labels):
    epochs = 50
    max_len = padded_sequences.shape[1]
    model = Sequential()
    model.add(Dense(16, input_dim=max_len))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.summary()

    history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs, verbose=1)

    return epochs, history, model

def __build_model(vocab_size, num_classes,padded_sequences,training_labels):
    epochs = 500
    embedding_dim = 16
    max_len = padded_sequences.shape[1]
    model = Sequential()
    model.add(Embedding(input_dim = vocab_size, output_dim = embedding_dim, input_length=max_len))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.summary()
    history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs, verbose=1)
    return epochs, history, model

def __save_model_to_file(model):
    return model.save("chat_model")

def pickle_data(rdict, labels, lbl_encoder, responses, training_labels, num_classes, max_length):

    with open('rdict.pickle', 'wb') as ecn_file:
        pickle.dump(rdict, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

    with open('labels.pickle', 'wb') as ecn_file:
        pickle.dump(labels, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

    with open('label_encoder.pickle', 'wb') as ecn_file:
        pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

    with open('responses.pickle', 'wb') as ecn_file:
        pickle.dump(responses, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
   
    with open('training_labels.pickle', 'wb') as ecn_file:
        pickle.dump(training_labels, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
   
    with open('num_classes.pickle', 'wb') as ecn_file:
        pickle.dump(num_classes, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

    with open('max_length.pickle', 'wb') as ecn_file:
        pickle.dump(max_length, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickles():

    with open('rdict.pickle', 'rb') as enc:
        rdict = pickle.load(enc)

    with open('labels.pickle', 'rb') as enc:
        labels = pickle.load(enc)
 
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    with open('responses.pickle', 'rb') as enc:
        responses = pickle.load(enc)

    with open('training_labels.pickle', 'rb') as enc:
        training_labels = pickle.load(enc)

    with open('num_classes.pickle', 'rb') as enc:
        num_classes = pickle.load(enc)

    with open('max_length.pickle', 'rb') as enc:
        max_length = pickle.load(enc)

    return rdict, labels, lbl_encoder, responses, training_labels, num_classes, max_length



def build_B():

    rdict, intent, labels, num_classes, responses, training_labels, training_sentences = __readin_intensions(intents_file)
    lbl_encoder, training_labels_encoded = __label_encoder(training_labels)
    vectorized_sentences = load_vectorized_sentences()
    max_length = vectorized_sentences.shape[1]

    epochs, history, model = __build_vectorized_model(num_classes,vectorized_sentences,training_labels_encoded)
    
    pickle_data(rdict = rdict, labels = labels, lbl_encoder = lbl_encoder, 
                responses = responses, training_labels = training_labels_encoded, 
                num_classes = num_classes,max_length = max_length)
    __save_model_to_file(model)


def build_A():
    intent, labels, num_classes, responses, training_labels, training_sentences = __readin_intensions(intents_file)
    lbl_encoder, training_labels_encoded = __label_encoder(training_labels)
    padded_sequences, tokenizer = __tokenize_vobabulary(training_sentences)
    vocabulary_size = len(tokenizer.word_index)
    epochs, history, model = __build_model(vocabulary_size, num_classes,padded_sequences,training_labels_encoded)
    __save_model_to_file(model)
    #pickle_data(labels, lbl_encoder, responses, tokenizer, training_labels)


if __name__ == "__main__":


    print("Building Model")
    #build_A()
    build_B()
    print("Built")
