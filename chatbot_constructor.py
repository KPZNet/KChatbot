#%%
from build_model import *


#%%
rdict, intent, labels, num_classes, responses, training_labels, training_sentences,lbl_encoder, training_labels_encoded = build_trainingdata()
pickle_trainingdata(rdict, labels, lbl_encoder, responses, training_labels_encoded, num_classes)
#vectorized_sentences = vectorize_all_sentences(training_sentences)
#pickle_vectorized_sentences(vectorized_sentences)
#%%
rdict, labels, lbl_encoder, responses, training_labels, num_classes = load_pickles()
vectorized_sentences = load_vectorized_sentences()

epochs, history, model = __build_vectorized_model(5,num_classes,training_labels_encoded,vectorized_sentences)
__save_model_to_file(model)

print("Built")