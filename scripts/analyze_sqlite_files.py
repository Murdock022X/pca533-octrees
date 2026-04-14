import os
import sqlite3
import pandas
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

rootdir = os.path.join("..", "outputs_031126")


quantities = ['n10k', 'n100k', 'n1m', 'n10m', 'n100m']
graph_info = {
'filament_xyz': ('solid','^', 'filament_xyz'), 
'filament_yz': ('dotted','s', 'filament_yz'), 
'filament_z': ('dashed','v', 'filament_z'), 
'pancake': ('solid','D', 'pancake'), 
'pancake_initial_tilted': ('dashdot','x', 'pancake_tilted'),
'uniform': ('dotted','.', 'uniform')
 }

hatch_info = {
'ComputeKeys': ('*'), 
'ReorderXYZK': ('|'), 
'SortKeys': ('o'), 
'UpdateInternal': ('X'), 
'UpdateLeaves': ('\\'),
}

scientific = {
'n10k': ('10^4'), 
'n100k': ('10^5'), 
'n1m': ('10^6'), 
'n10m': ('10^7'), 
'n100m': ('10^8'),
}

for quantity in quantities:
    dictionary = dict()
    composition = dict()
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
                
                initialBreakdown = pandas.read_sql_query("""
                SELECT Events.text, (Events.end - Events.start) as time FROM NVTX_EVENTS as Events
                WHERE exists (select * from NVTX_EVENTS as SubEvents where SubEvents.end > Events.end and SubEvents.start < Events.end and SubEvents.text = "Initial"
                and (SubEvents.start) != (select min(start) from NVTX_EVENTS where text = "Initial"))
                group by text
                order by Events.start
                """, connection)
                
                perturbedBreakdown = pandas.read_sql_query("""
                SELECT Events.text, (Events.end - Events.start) as time FROM NVTX_EVENTS as Events
                WHERE exists (select * from NVTX_EVENTS as SubEvents where SubEvents.end > Events.end and SubEvents.start < Events.end and SubEvents.text = "Perturb"
                and (SubEvents.start) != (select min(start) from NVTX_EVENTS where text = "Perturb"))
                group by text
                order by Events.start
                """, connection)
                
               
                if(p.parts[2] not in composition):
                    composition[p.parts[2]] = dict()
                composition[p.parts[2]][p.parts[3]] = []
                composition[p.parts[2]][p.parts[3]].append(initialBreakdown.values.tolist())
                composition[p.parts[2]][p.parts[3]].append(perturbedBreakdown.values.tolist())
                
                
                
                connection.close()
#print(dictionary)
    #print(composition)
    if True:
        for dist, graphData in composition.items():
            labels = []
            for key, value in graphData.items():
                labels.append("Init")
                labels.append("."+key[3:])

            partitioned = dict()

            for key, value in graphData.items():
                i = 0
                for metric in value[0]:
                    if(metric[0] not in partitioned):
                        partitioned[metric[0]] = []
                    partitioned[metric[0]].append(metric[1]/1000000)
                    if(value[1][i][0] not in partitioned):
                        partitioned[value[1][i][0]] = []
                    partitioned[value[1][i][0]].append(value[1][i][1]/1000000)
                    i = i + 1
                    
            width = .75

            fig, ax = plt.subplots()
            bottom = np.zeros(8)
            across = 1
            spacing = np.arange(0,8, 1)
            for i in range(0,spacing.size):
                spacing[i] += i//2 * across
            

            for label, partition in partitioned.items():
                p = ax.bar(x = spacing, height = partition, width = width, label=label, bottom=bottom, align = 'center', hatch = hatch_info[label])
                bottom += partition
            plt.grid(axis='y')
            ax.set_axisbelow(True)
            ax.set_ylabel('Execution Time (ms)')
            ax.set_xlabel('Perturbation Strength')
            ax.legend()
            ax.set_title("Breakdown of " + dist + " with n = $"+scientific[quantity]+"$")
            plt.xticks(spacing, labels=labels)
            plt.savefig('..\\outputs_031126\\breakdown_' + quantity + '_' + dist + '.png', bbox_inches="tight")
            
    
    
    if True:
        labels = []
        
        partitioned = dict()

        for key, value in composition.items():
            labels.append(graph_info[key][2])
            for metric in value['s0p001'][0]:
                if(metric[0] not in partitioned):
                    partitioned[metric[0]] = []
                partitioned[metric[0]].append(metric[1]/1000000)

        width = .75

        fig, ax = plt.subplots()
        bottom = np.zeros(6)


        for boolean, partition in partitioned.items():
            #print(boolean)
            p = ax.bar(x = labels, height = partition, width = width, label=boolean, bottom=bottom, align = 'center', hatch = hatch_info[boolean])
            bottom += partition
            
        ax.tick_params(axis='x', which='major', labelsize=8)
        plt.grid(axis='y')
        ax.set_axisbelow(True)
        ax.set_ylabel('Execution Time (ms)')
        ax.set_xlabel('Distribution')
        ax.legend()
        ax.set_title("Breakdown of inital distributions with n = $"+scientific[quantity]+"$")
        plt.savefig('..\\outputs_031126\\breakdown_' + quantity + '.png', bbox_inches="tight")
    
    
    if True:
        x = [.001,.01,.1,.5]
        x_pos = [0,1,2,3]
        
        fig, ax = plt.subplots(constrained_layout=True)
        
        plots = []
        names = []
        for dist, perturbations in dictionary.items():
            ys = []
            for perturbation, values in perturbations.items():
                ys.append(100 * values[1][1] / values[0][1])
            dist_plot, = ax.plot(x_pos, ys, ls = graph_info[dist][0], marker = graph_info[dist][1], linewidth=2, color = 'k')
            plots.append(dist_plot)
            names.append(graph_info[dist][2])
        
        ax.legend(plots, names)
        
        ax.set_ylabel('Percent Change')
        ax.set_xlabel('Perturbation Strength')
        plt.xticks(x_pos, labels=[str(x) for x in x])
        plt.grid(axis='y')
        ax.set_axisbelow(True)
        ax.set_title("Relative Change in Execution Time After Perturbation for n = $"+scientific[quantity]+"$")
        plt.savefig('..\\outputs_031126\\plotted_' + quantity + '.png', bbox_inches="tight")

    with open('..\\outputs_031126\\analyzed_relative_' + quantity + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for dist, perturbations in dictionary.items():
            header = [dist]
            row = ["Difference"]                   
            for perturbation, values in perturbations.items():
                header.append("0."+perturbation[3:])
                row.append(values[1][1] / values[0][1])
            writer.writerow(header)
            writer.writerow(row)
            
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
       
