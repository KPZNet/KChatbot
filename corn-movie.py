import json

lines = open('movieinput/movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open('movieinput/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

# Create a dictionary to map each line's id with its text
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]


# Create a list of all of the conversations' lines' ids.
convs = []
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    convs.append(_line.split(','))

#id and conversation sample
for k in convs[300]:
    print (k, id2line[k])
# Sort the sentences into questions (inputs) and answers (targets)
questions = []
answers = []
for conv in convs:
    for i in range(len(conv)-1):
        questions.append(id2line[conv[i]])
        answers.append(id2line[conv[i+1]])
        
# Compare lengths of questions and answers
print(len(questions))
print(len(answers))

jsonArray = []
for i in range(len(questions)):
    jrec = {'tag':i, 'patterns':[questions[i] ],'responses':[answers[i]]}
    jsonArray.append(jrec)
    if i > 100:
        break

#convert python jsonArray to JSON String and write to file
with open('cmovies.json', 'w', encoding='utf-8') as jsonf:
    jdict = {"intents":jsonArray}
    jsonString = json.dumps(jdict, indent=4)
    jsonf.write(jsonString)


