
from build_model import *
from databot import start_chat

import os

print("Done Imports")

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

    build_trainer('intents_statbot.json', model_name = 'statbotQA')
    build_modeler('statbotQA', 50)
    #deploy_model('statbotQA')
    print("Built!")

    #start_chat(model_name)

if __name__ == "__main__":
    build_statbot()