import csv
import json


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

def csv_to_json(cQ, cA, jsonFilePath):
    jsonArray = []

    with open(cA, encoding='utf-8') as csvfA:
        csvReaderA = csv.DictReader(csvfA)
        answersDict = build_answer_dictionary(csvReaderA)
        with open(cQ, encoding='utf-8') as csvfQ:
            csvReaderQ = csv.DictReader(csvfQ)

            for row in csvReaderQ:
                id = row['Id']
                rs = find_parent( id , answersDict)
                jtag = row["Title"]
                jpatterns = row["Body"]
                jtag_clean = row["Title_clean"]
                jpatterns_clean = row["Body_clean"]
                jresponses = rs #[b['Body'] for b in rs]
                jrec = {'id':id, 'tag':jtag, 'pattern':jpatterns, 'tag_clean':jtag_clean, 'pattern_clean':jpatterns_clean,'response':jresponses}
                jsonArray.append(jrec)

    #convert python jsonArray to JSON String and write to file
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jdict = {"intents":jsonArray}
        jsonString = json.dumps(jdict, indent=4)
        jsonf.write(jsonString)

cQ = r'clean_questions.csv'
cA = r'clean_answers.csv'
jsonFilePath = r'intents_qa.json'
csv_to_json(cQ, cA, jsonFilePath)
print("Completed intents_qa JSON file")