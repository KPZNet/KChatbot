import csv
import json

def find_parent(pid, dr):
    rs = []
    for row in dr:
        id = row["ParentId"]
        if id == pid:
            b = row["Body"]
            rs.append(b)
    return rs

def csv_to_json(cQ, cA, jsonFilePath):
    jsonArray = []

    with open(cA, encoding='utf-8') as csvfA:
        csvReaderA = csv.DictReader(csvfA)
        with open(cQ, encoding='utf-8') as csvfQ:
            csvReaderQ = csv.DictReader(csvfQ)

            #convert each csv row into python dict
            for row in csvReaderQ:
                #add this python dict to json array
                id = row["Id"]
                rs = find_parent(id,csvReaderA)
                jtag = row["Title"]
                jpatterns = row["Text"]
                jresponses = rs
                jrec = {'tag':jtag, 'pattern':jpatterns, 'response':jresponses}
                jsonArray.append(jrec)

    #convert python jsonArray to JSON String and write to file
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jdict = {"intents":jsonArray}
        jsonString = json.dumps(jdict, indent=4)
        jsonf.write(jsonString)

cQ = r'df_questions_fullclean.csv'
cA = r'df_answers_fullclean.csv'
jsonFilePath = r'intents2.json'
csv_to_json(cQ, cA, jsonFilePath)