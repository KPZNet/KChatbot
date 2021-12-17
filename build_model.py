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

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def make_sentence_vector(sentence, words):
    sentence_matrix =[]
    for t in sentence:
        try:
            if t in words:
                sentence_matrix.append(words[t])
            else:
                print(t, " was not found")
        except Exception:
            pass
    if len(sentence_matrix) != 0:
        sentence_matrix =  np.array(sentence_matrix)
    return np.average(sentence_matrix,axis=0)


def __readin_intensions(tfile):
    with open(tfile) as file:
        data = json.load(file)
        
    training_sentences = []
    training_labels = []
    labels = []
    responses = []
    
    for intent in data['intents']:
        for pattern in intent['patterns']:
            training_sentences.append(pattern)
            training_labels.append(intent['tag'])
        responses.append(intent['responses'])
        
        if intent['tag'] not in labels:
            labels.append(intent['tag'])
            
    num_classes = len(labels)
    return intent, labels, num_classes, responses, training_labels, training_sentences

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

    return embedding_dim, max_len, oov_token, padded_sequences, sequences, tokenizer, vocab_size, word_index

def pickleSentences(sentences):
    with open('transformed_sents.pickle', 'wb') as handle:
        pickle.dump(sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)

def loadsentencespickle():
    # load tokenizer object
    with open('transformed_sents.pickle', 'rb') as handle:
        sentences = pickle.load(handle)
    return sentences

def __tokenize_vobabulary2(training_sentences, loadpickle = False, picklesentences = False):
    vocab_size = 3000
    embedding_dim = 16
    max_len = 0
    oov_token = "<OOV>"

    i = 0
    l = len(training_sentences)
    ps = []

    if loadpickle:
        ps = loadsentencespickle()
    else:
        for s in training_sentences:
            if i % 10 == 0:
                print("Vectorized {0} / {1} sentences".format(i, l))
            i += 1

            p = sbert_model.encode([s])[0]
            ps.append(p)
        if picklesentences:
            pickleSentences(ps)

    max_len = ps.shape[1]
    # Create a matrix of 3x4 dimensions - 3 rows and four columns
    array_2d = np.ndarray((len(ps), max_len))
    # Populate the 2 dimensional array created using nump.ndarray
    for x in range(0, array_2d.shape[0]):
        for y in range(0, array_2d.shape[1]):
            array_2d[x][y] = ps[x][y]

    ps = array_2d
    return embedding_dim, max_len, oov_token, ps, vocab_size




def __build_model2(vocab_size,embedding_dim,max_len,num_classes,padded_sequences,training_labels):

    model = Sequential()
    model.add(Dense(16, input_dim=max_len))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer='adam', metrics=['accuracy'])
    
    model.summary()
    epochs = 500
    history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs, verbose=1)
    return epochs, history, model

def __build_model(vocab_size,embedding_dim,max_len,num_classes,padded_sequences,training_labels):

    model = Sequential()
    model.add(Embedding(input_dim = vocab_size, output_dim = embedding_dim, input_length=max_len))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer='adam', metrics=['accuracy'])
    
    model.summary()
    epochs = 500
    history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs, verbose=1)
    return epochs, history, model

# to save the trained model
def __save_model_to_file(model):
    return model.save("chat_model")

def pickle_data(labels, lbl_encoder, responses, tokenizer, training_labels, training_sentences):
    # to save the fitted tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # to save the fitted label encoder
    with open('label_encoder.pickle', 'wb') as ecn_file:
        pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
    # to save the responses
    with open('responses.pickle', 'wb') as ecn_file:
        pickle.dump(responses, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
    # to save the intent JSON
    with open('intent.pickle', 'wb') as ecn_file:
        pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
    # to save the intent JSON
    with open('training_labels.pickle', 'wb') as ecn_file:
        pickle.dump(training_labels, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
    # to save the intent JSON
    with open('training_sentences.pickle', 'wb') as ecn_file:
        pickle.dump(training_sentences, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
    # to save the intent JSON
    with open('labels.pickle', 'wb') as ecn_file:
        pickle.dump(labels, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickles():
    # load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    # load label encoder object
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)
    # load label encoder object
    with open('intent.pickle', 'rb') as enc:
        intent = pickle.load(enc)
    with open('training_labels.pickle', 'rb') as enc:
        training_labels = pickle.load(enc)
    with open('training_sentences.pickle', 'rb') as enc:
        training_sentences = pickle.load(enc)
    with open('labels.pickle', 'rb') as enc:
        labels = pickle.load(enc)
    return lbl_encoder, tokenizer, intent, training_labels, training_sentences, labels

def build():
    intent, labels, num_classes, responses, training_labels, training_sentences = __readin_intensions('intents_qa.json')
    lbl_encoder, training_labels_encoded = __label_encoder(training_labels)
    
    #embedding_dim, max_len, oov_token, padded_sequences, sequences, tokenizer, vocab_size, word_index = __tokenize_vobabulary(training_sentences)
    embedding_dim, max_len, oov_token, padded_sequences, vocab_size = __tokenize_vobabulary2(training_sentences, True, False)
    epochs, history, model = __build_model2(vocab_size,embedding_dim,max_len,num_classes,padded_sequences,training_labels_encoded)
    __save_model_to_file(model)

    #pickle_data(labels, lbl_encoder, responses, tokenizer, training_labels, training_sentences)


if __name__ == "__main__":
    print("Building Model")
    build()
    print("Built")
