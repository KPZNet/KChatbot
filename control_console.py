from data_preprocessing import post_process_statbot_json
from build_model import *
from databot import start_chat


def compare(p1, p2):
    u = sbert_model.encode(p1)
    v = sbert_model.encode(p2)
    d = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return d

def comp():
    s1 = "Please open the file"
    s2 = "Can you open this file"
    d = compare(s1, s2)
    print("\n\n")
    print('  Sentence 1 : \"{0}\"'.format(s1) )
    print('  Sentence 2 : \"{0}\"'.format(s2) )
    print("  Vectorized Comparison Dot Product: {0:.4f}".format(d))


#post_process_statbot_json()
#build_statbot()
#start_chat()

