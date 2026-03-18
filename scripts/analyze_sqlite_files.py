import os
import sqlite3
import pandas
from pathlib import Path
import csv
import matplotlib.pyplot as plt

rootdir = os.path.join("..", "outputs_031126")


quantities = ['n10k', 'n100k', 'n1m', 'n10m', 'n100m']
graph_info = {
'filament_xyz': ('solid','^', 'Line XYZ'), 
'filament_yz': ('dotted','s', 'Line YZ'), 
'filament_z': ('dashed','v', 'Line Z'), 
'pancake': ('solid','D', 'Disc'), 
'pancake_initial_tilted': ('dashdot','x', 'Disc Tilted'),
'uniform': ('dotted','.', 'Cube')
 }



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

    x = [.001,.01,.1,.5]
    x_pos = [0,1,2,3]
    
    fig, ax = plt.subplots(constrained_layout=True)
    
    plots = []
    names = []
    for dist, perturbations in dictionary.items():
        ys = []
        for perturbation, values in perturbations.items():
            ys.append(values[1][1] / values[0][1])
        dist_plot, = ax.plot(x_pos, ys, ls = graph_info[dist][0], marker = graph_info[dist][1], linewidth=2, color = 'k')
        plots.append(dist_plot)
        names.append(graph_info[dist][2])
    
    ax.legend(plots, names)
    
    plt.xticks(x_pos, labels=[str(x) for x in x])
    #plt.ylim(.36, 1.44)
    plt.savefig('..\\outputs_031126\\plotted_' + quantity + '.png', bbox_inches="tight")

    with open('..\\outputs_031126\\analyzed_' + quantity + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for dist, perturbations in dictionary.items():
            if(True):
                header = [dist]
                row = ["Difference"]                   
                for perturbation, values in perturbations.items():
                    header.append("0."+perturbation[3:])
                    row.append(values[1][1] / values[0][1])
                writer.writerow(header)
                writer.writerow(row)
            else:
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
       
