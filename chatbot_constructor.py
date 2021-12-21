
from build_model import *
from databot import start_chat
print("Done Imports")

def build_trainer(intents_file, model_name):
    rdict, intent, labels, num_classes, responses, training_labels,training_sentences,lbl_encoder, training_labels_encoded = build_trainingdata(intents_file)
    pickle_trainingdata(model_name,rdict, labels, lbl_encoder, responses, training_labels_encoded, num_classes)
    vectorized_sentences = vectorize_all_sentences(training_sentences)
    pickle_vectorized_sentences(model_name, vectorized_sentences)
    print("Done encoding AND pickled")

def build_modeler(model_name, epochs):
    rdict, labels, lbl_encoder, responses, training_labels_encoded, num_classes = load_pickles(model_name)
    vectorized_sentences = load_vectorized_sentences(model_name)
    
    epochs, history, model = build_vectorized_model(epochs ,num_classes,training_labels_encoded,vectorized_sentences)
    save_model_to_file(model, model_name+'NNModel')
    return vectorized_sentences

model_name = 'conversationQA'
intents_file = 'intents.json'
#build_trainer(intents_file, model_name = model_name)
#build_modeler(model_name, 500)
print("Built!")

start_chat(model_name)