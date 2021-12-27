
# Importing the required libraries
import xml.etree.ElementTree as Xet
import pandas as pd

#Id,OwnerUserId,CreationDate,Score,Title,Body
#Id,OwnerUserId,CreationDate,ParentId,Score,Body
def StatsExchangeXMLtoCSV():
    colsQ = ["Id", "PostTypeId", "AcceptedAnswerId", "CreationDate", "Title"]
    colsA = ["Id", "PostTypeId", "ParentId", "CreationDate", "Body"]
    rowsQ = []
    rowsA = []
    
    # Parsing the XML file
    xmlparse = Xet.parse('stats.stackexchange.com\Posts.xml')
    root = xmlparse.getroot()
    for i in root:
    	k = i.attrib
    	rid = k.get("Id")
    	postidtype = k.get("PostTypeId")
    	if postidtype is not None:
    		if postidtype == '1':
    			answerid = k.get("AcceptedAnswerId")
    			cdate = k.get("CreationDate")
    			title = k.get("Title")
    			rowsQ.append({"Id": rid,
    						"PostTypeId": postidtype,
    						"AcceptedAnswerId": answerid,
    						"CreationDate": cdate,
    						"Title": title})
    		if postidtype == '2':
    			cdate = k.get("CreationDate")
    			parentId = k.get("ParentId")
    			body = k.get("Body")
    			rowsA.append({"Id": rid,
    						"PostTypeId": postidtype,
    						"ParentId": parentId,
    						"CreationDate": cdate,
    						"Body": body})
    dfQ = pd.DataFrame(rowsQ, columns=colsQ)
    dfA = pd.DataFrame(rowsA, columns=colsA)
    
    # Writing dataframe to csv
    dfA.to_csv('stats_qus.csv')
    dfQ.to_csv('stats_ans.csv')

#StatsExchangeXMLtoCSV()
