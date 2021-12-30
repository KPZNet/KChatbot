import json 
import os
import numpy as np 
import pickle
import pandas as pd

from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')


class creply:
    def __init__(self, resp, patt, tg):
        self.responses = resp
        self.patterns = patt
        self.tag = tg

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

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def compare(p1, p2):
    u = sbert_model.encode(p1)[0]
    v = sbert_model.encode(p2)[0]
    d = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return d

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
