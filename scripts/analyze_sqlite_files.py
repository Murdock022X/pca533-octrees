import os
import sqlite3
import pandas
from pathlib import Path
import csv

rootdir = os.path.join("..", "outputs_031126")


quantities = ['n10k', 'n100k', 'n1m', 'n10m', 'n100m']

for quantity in quantities:
    dictionary = dict()
    for root, subFolders, files in os.walk(rootdir):
        for file in files:
            if(file.endswith(".sqlite") and quantity in file):
                connection = sqlite3.connect(os.path.join(root,file))
                #print(os.path.join(root,file))
                df = pandas.read_sql_query("""
                SELECT text, avg(end-start) as time FROM NVTX_EVENTS 
                WHERE text = "Initial" and (end-start) != (select max(end-start) from NVTX_EVENTS where text = "Initial") or 
                text = "Perturb" and (end-start) != (select max(end-start) from NVTX_EVENTS where text = "Perturb")
                group by text
                order by text
                """, connection)
                p = Path(os.path.join(root,file))
                #print(p.parts[2])
                if(p.parts[2] not in dictionary):
                    dictionary[p.parts[2]] = dict();
                dictionary[p.parts[2]][p.parts[3]] = df.values.tolist();
                #print(df)
                connection.close()
#print(dictionary)

    with open('..\\outputs_031126\\analyzed_' + quantity + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for dist, perturbations in dictionary.items():
            header = [dist]
            #print(perturbations)
            for perturbation in perturbations:
                header.append(perturbation[2:])
            writer.writerow(header)
            row = []
            #print(perturbations)
            addLabel = True
            for perturbation, values in perturbations.items():
                header.append(values)
                if(addLabel):
                    row.append(values[0][0])
                    addLabel = False
                row.append(values[0][1])
            writer.writerow(row)
            row = []
            addLabel = True
            for perturbation, values in perturbations.items():
                header.append(values)
                if(addLabel):
                    row.append(values[1][0])
                    addLabel = False
                row.append(values[1][1])
            writer.writerow(row)
       
