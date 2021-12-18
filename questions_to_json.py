import csv
import json
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action
from tqdm import tqdm
from nlpaug.util import Action
from nlpaug.util.file.download import DownloadUtil


def get_randos(text, numrandos):
    at = []
    at.append(text)
    aug = naw.SynonymAug(aug_src='wordnet')
    for i in range(numrandos):
        t = aug.augment(text)
        at.append(t)

    return at

def build_answer_dictionary(dfa):
    rdict = {}
    for row in dfa:
        id = row["ParentId"]
        quests = []
        if id in rdict.keys():
            quests = rdict[id]
            quests.append(row)
            rdict[id] = quests
        else:
            quests.append(row)
            rdict[id] = quests
    return rdict

def find_parent(pid, dr):
    rs = []
    if pid in dr:
        rs = dr[pid]
    return rs

def csv_to_json(cQ, cA, total_sets, augs, jsonFilePath):
    jsonArray = []

    with open(cA, encoding='utf-8') as csvfA:
        csvReaderA = csv.DictReader(csvfA)
        answersDict = build_answer_dictionary(csvReaderA)
        with open(cQ, encoding='utf-8') as csvfQ:
            csvReaderQ = csv.DictReader(csvfQ)

            irow = 0
            for row in csvReaderQ:

                if irow % 100 == 0:
                    print("Processing row {0} / {1} with {2} augments".format(irow, total_sets, augs))

                id = row['Id']
                rs = find_parent( id , answersDict)
                jtag = row["Title"]
                patterns = get_randos(jtag, augs)
                jpatterns = patterns #row["Body"]
                #jtag_clean = row["Title_clean"]
                #jpatterns_clean = row["Body_clean"]
                jresponses = [b['Body'] for b in rs]
                jrec = {'tag':id, 'patterns':jpatterns ,'responses':jresponses}
                jsonArray.append(jrec)
                irow += 1
                if irow >= total_sets:
                    break

    #convert python jsonArray to JSON String and write to file
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jdict = {"intents":jsonArray}
        jsonString = json.dumps(jdict, indent=4)
        jsonf.write(jsonString)

def convert_qa_to_json():
    cQ = r'clean_questions.csv'
    cA = r'clean_answers.csv'
    jsonFilePath = r'intents_qa.json'
    csv_to_json(cQ, cA, 10000, 40, jsonFilePath)
    print("Completed intents_qa JSON file")