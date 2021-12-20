#%%
from build_model import pickle_trainingdata, build_trainingdata
print("Done Imports")

#%%
def build_trainer(model_name):
    rdict, intent, labels, num_classes, responses, training_labels, training_sentences,lbl_encoder, training_labels_encoded = build_trainingdata()
    pickle_trainingdata(model_name,rdict, labels, lbl_encoder, responses, training_labels_encoded, num_classes)
    #vectorized_sentences = vectorize_all_sentences(training_sentences)
    #pickle_vectorized_sentences(vectorized_sentences)
    print("Done encoding nad pickling")

#%%
def build_modeler(model_name):
    rdict, labels, lbl_encoder, responses, training_labels, num_classes = load_pickles()
    vectorized_sentences = load_vectorized_sentences()
    
    epochs, history, model = build_vectorized_model(5,num_classes,training_labels_encoded,vectorized_sentences)
    save_model_to_file(model_name,model)
    return vectorized_sentences

build_trainer('pythonQA')
build_modeler('pythonQA')

print("Built")