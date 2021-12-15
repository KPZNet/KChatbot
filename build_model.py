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


def vectorize_sentence(stnc):

    words = open("cc.en.300.vec","r").read().splitlines()

    _words = dict()
    for w in words[1]:
        w = w.split()
        word = w.pop(0)
        _words[word] = list(map(float, w))


def make_sentence_vector(tokens, words):
    sentence_matrix =[]
    for t in tokens:
        if t in words:
            sentence_matrix.append(words[t])
        else:
            print(t, " was not found")
    sentence_matrix = np.array(sentence_matrix)
    return np.average(sentence_matrix,axis=0)

sentence = "Hello my name"
tokens = sentence.split()
tokens

vector = make_sentence_vector(tokens, words)
vector.shape

def tell_me_how_similar(s1, s2, words):
    v1 = make_sentence_vector(s1, words)
    v2 = make_sentence_vector(s2, words)
    return cosine(v1,v2)

sentence = "Hello my name is bob"
tokens1 = sentence.split()
sentence = "Hello my name is joe"
tokens2 = sentence.split()

tell_me_how_similar(tokens1,tokens2,words)




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
    training_labels = lbl_encoder.transform(training_labels)
    return lbl_encoder, training_labels

def __tokenize_vobabulary(training_sentences):
    vocab_size = 1000
    embedding_dim = 16
    max_len = 20
    oov_token = "<OOV>"
    
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)
    return embedding_dim, max_len, oov_token, padded_sequences, sequences, tokenizer, vocab_size, word_index

def __build_model(vocab_size,embedding_dim,max_len,num_classes,padded_sequences,training_labels):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer='adam', metrics=['accuracy'])
    
    model.summary()
    epochs = 500
    history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)
    return epochs, history, model

# to save the trained model
def __save_model_to_file(model):
    return model.save("chat_model")

def build():
    intent, labels, num_classes, responses, training_labels, training_sentences = __readin_intensions('intents.json')
    lbl_encoder, training_labels_encoded = __label_encoder(training_labels)
    embedding_dim, max_len, oov_token, padded_sequences, sequences, tokenizer, vocab_size, word_index = __tokenize_vobabulary(training_sentences)
    epochs, history, model = __build_model(vocab_size,embedding_dim,max_len,num_classes,padded_sequences,training_labels_encoded)
    __save_model_to_file(model)

    pickle_data(labels, lbl_encoder, responses, tokenizer, training_labels, training_sentences)


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
