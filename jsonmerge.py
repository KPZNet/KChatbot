import json
import pandas as pd

with open('file1.json') as f1:               
    data1 = json.load(f1)

with open('file2.json') as f2:                       
    data2 = json.load(f2)
    
df1 = pd.DataFrame([data1])                      
df2 = pd.DataFrame([data2])                     

MergeJson = pd.concat([df1, df2], axis=1)        

MergeJson.to_json("file_merged.json")          
