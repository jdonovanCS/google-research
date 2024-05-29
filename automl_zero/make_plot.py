import numpy as np
from matplotlib import pyplot as plt
import os
import sqlite3
import seaborn as sns

baseline_db = "baseline.db3"
baseline_full_db = "baseline_full.db3"
baseline_qd_db = "baseline_qd.db3"
baseline_full_qd_db = "baseline_full_qd.db3"

dbs = [baseline_db, baseline_qd_db, baseline_full_db, baseline_full_qd_db]
colors = ['blue', 'orange', 'red', 'purple']

table = "progress"
fields = "evol_id, mean, num_indivs"
where_clause = "TRUE"
results = {d: {} for d in dbs}

def plot_mean_and_bootstrapped_ci_multiple(input_data = None, title = 'overall', name = "change this", x_label = "x", y_label = "y", x_mult=1, y_mult=1, save_name="", compute_CI=True, maximum_possible=None, show=None, sample_interval=None, legend_loc=None, alpha=1, y=None):
    """ 
     
    parameters:  
    input_data: (numpy array of numpy arrays of shape (max_k, num_repitions)) solution met
    name: numpy array of string names for legend 
    x_label: (string) x axis label 
    y_label: (string) y axis label 
     
    returns: 
    None 
    """ 
 
    generations = len(input_data[0])
 
    fig, ax = plt.subplots() 
    ax.set_xlabel(x_label) 
    ax.set_ylabel(y_label) 
    ax.set_title(title) 
    for i in range(len(input_data)): 
        CIs = [] 
        mean_values = [] 
        for j in range(generations): 
            mean_values.append(np.mean(input_data[i][j])) 
            if compute_CI:
                CIs.append(bootstrap.ci(input_data[i][j], statfunction=np.mean)) 
        mean_values=np.array(mean_values) 
 
        high = [] 
        low = [] 
        if compute_CI:
            for j in range(len(CIs)): 
                low.append(CIs[j][0]) 
                high.append(CIs[j][1]) 
 
        low = np.array(low) 
        high = np.array(high) 

        if type(y) == type(None):
            y = range(0, generations)
        if (sample_interval != None):
            y = np.array(y)*sample_interval 
        ax.plot(y, mean_values, label=name[i], alpha=alpha)
        if compute_CI:
            ax.fill_between(y, high, low, alpha=.2) 
        if legend_loc is not None:
            ax.legend(bbox_to_anchor=legend_loc['bbox'], loc=legend_loc['loc'], ncol=1)
        else:
            ax.legend()
    
    if maximum_possible:
        ax.hlines(y=maximum_possible, xmin=0, xmax=generations, linewidth=2, color='r', linestyle='--', label='best poss. acc.')
        ax.legend()

    if save_name != "":
        plt.savefig('plots/' + save_name)
    if show != None:
        plt.show()


for db in dbs:

    conn = sqlite3.connect(db)
    c = conn.cursor()
    query = "SELECT " + fields + " FROM " + table + " WHERE " + where_clause + ";"
    # print(query)
    res = c.execute(query)
    res = res.fetchall()
    for r in res:
        # print(db, r[0])
        if r[0] not in results[db]:
            results[db][r[0]] = []
        results[db][r[0]].append((r[1], r[2]))

print(results)

for i, (db, v) in enumerate(results.items()):
    db_x, db_y = [], []
    color = colors[i]
    for evol_id in results[db]:
        x, y = zip(*results[db][evol_id])
        db_x.extend(x)
        db_y.extend(y)
    # print(len(db_x), len(db_y))
    # plt.plot(np.unique(db_y), np.poly1d(np.polyfit(db_y, db_x, 1)), (np.unique(db_y)), color=color)
    ax=sns.regplot(x=db_y, y=db_x, scatter_kws={'s':1}, logx=True)
    # ax.set_xscale('log')
    plt.show()

    



