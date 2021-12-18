import nlpaug.augmenter.word as naw
import numpy as np

from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

#Cosine similarity function
def cosine(p1, p2):
    u = sbert_model.encode(p1)[0]
    v = sbert_model.encode(p2)[0]
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

#Augment by adding contextual words
def get_word_adder(text, numrandos):
    at = []
    aug = naw.ContextualWordEmbsAug(
        model_path='bert-base-uncased', action="insert")

    for i in range(numrandos):
        t = aug.augment(text)
        at.append(t)

    return at

#Augment by swapping words with their synonyms
def get_synonyms(text, numrandos):
    at = []
    aug = naw.SynonymAug(aug_src='wordnet')
    for i in range(numrandos):
        t = aug.augment(text)
        at.append(t)
    return at

#Compare augmented sentences with original
#and print out cosine similarity
def s_dotp(s, syns):
    print("Dot Product Cosine Similarity:")
    print("{0}".format(s))
    for sent in syns:
        dotprod = cosine(s, sent)
        print("\t{0:.2} - {1}".format(dotprod,sent))


def run_sample_augmentation():
    s = ["My dog has a wet nose"]
    syns = get_synonyms(s, 5)
    s_dotp(s, syns)
    
    s = ["It is raining very hard outside"]
    syns = get_synonyms(s, 5)
    s_dotp(s, syns)
    
    
    s = ["My dog has a wet nose"]
    syns = get_word_adder(s, 5)
    s_dotp(s, syns)
    
    s = ["It is raining very hard outside"]
    syns = get_word_adder(s, 5)
    s_dotp(s, syns)
    return s, syns

#run_sample_augmentation()

if __name__ == "__main__":
    sentences = open("sentences.txt","r").read().splitlines()
    print("\nText Augmentation\n")
    for s in sentences:
        print("Synonyms Augmentation")
        syns = get_synonyms(s, 5)
        s_dotp(s, syns)
        print("Contextual Addition Augmentation")
        syns = get_word_adder(s, 5)
        s_dotp(s, syns)