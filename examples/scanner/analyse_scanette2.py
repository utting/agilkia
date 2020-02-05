# -*- coding: utf-8 -*-
"""
Example analysis of Scanette logs - new CSV format.

Reads Scanette CSV files with these columns:
# Columns Docs from Frederick, 2019-10-18.
  0: id is an identifier of the line (some numbers may be missing)
  1: timestamp is in Linux format with three extra digits for milliseconds.
  2: sessionID provides the identifier of the session - each client is different.
  3: objectInstance is the object instance on which the operation is invoked.
  4: operation is the name of the operation (action).
  5: parameters is a list of the parameter values, or [] if there are no parameters.
  6: result is the status code returned (? means that the operation does
    not return anything - void)

Created on Thu Oct 17 16:45:39 2019

@author: utting@usc.edu.au
"""

import csv
from pathlib import Path
from datetime import datetime, date, time
from sklearn.cluster import MeanShift
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as pltcm
import matplotlib.markers as pltmarkers
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from collections import Counter
import sys

import agilkia

# %%

DISPLAY_TIME = 3
TEST_MODE = "--test" in sys.argv
if TEST_MODE:
    print(f"Running in test mode - all figures will close after {DISPLAY_TIME} seconds.")

# %%

def read_traces_csv(path: Path) -> agilkia.TraceSet:
    # print("now=", datetime.now().timestamp())
    with path.open("r") as input:
        trace1 = agilkia.Trace([])
        for line in csv.reader(input):
            # we ignore the line id.
            timestr = line[1].strip()
            timestamp = date.fromtimestamp(int(timestr) / 1000.0)
            # print(timestr, timestamp.isoformat())
            sessionID = line[2].strip()
            objInstance = line[3].strip()
            action = line[4].strip()
            paramstr = line[5].strip()
            result = line[6].strip()
            # now we identify the main action, inputs, outputs, etc.
            if paramstr == "[]":
                inputs = {}
            else:
                if  paramstr.startswith("[") and paramstr.endswith("]"):
                    paramstr = paramstr[1:-1]
                inputs = {"param" : paramstr}
            if result == "?":
                outputs = {}
            else:
                outputs = {'Status': float(result)}
            others = {
                    'timestamp': timestamp,
                    'sessionID': sessionID,
                    'object': objInstance
                    }
            event = agilkia.Event(action, inputs, outputs, others)
            trace1.append(event)
    traceset = agilkia.TraceSet([])
    traceset.append(trace1)
    return traceset


# %% Read traces and save in the Agilkia JSON format.

french_chars = {'scanner': '.', 'abandon': 'a', 'supprimer': 's',
                'ajouter': '+', 'debloquer': 'd', 'fermerSession': 'f',
                'ouvrirSession': 'o', 'payer': 'p', 'transmission': 't'}
english_chars = {'scanner': '.',        # 'scan'
                 'abandon': 'a',        # 'abandon'
                 'supprimer': 'd',      # 'delete'
                 'ajouter': '+',        # 'add'
                 'debloquer': 'u',      # 'unlock'
                 'fermerSession': 'c',  # 'closeSession'
                 'ouvrirSession': 'o',  # 'openSession'
                 'payer': 'p',          # 'pay'
                 'transmission': 't'    # 'transmit'
                 }
traceset = read_traces_csv(Path("127.0.0.1-1571403244552.csv"))
traceset.set_event_chars(english_chars)
    # was: {"scanner": ".", "abandon": "a", "supprimer": "s", "ajouter": "+"}

print("Event chars:\n  ", traceset.get_event_chars())  # default summary char for each kind of event.

print(f"One long trace, saved to log_one.json.  Length={len(traceset[0])} events.")
traceset.save_to_json(Path("log_one.json"))
print(str(traceset[0])[:200], "etc")  # everything is in one big trace initially.


# %% Split into separate traces, first based on Scanette number.

print("\n\n==== grouped by sessionID number ====")
traceset2 = traceset.with_traces_grouped_by("sessionID", property=True)
for tr in traceset2[0:10]:
    print("   ", tr)
print(f"    etc")
print(f"{len(traceset2)} traces now, saved into log_split.json")
traceset2.save_to_json(Path("log_split.json"))

# %% Sometimes one needs to split into smaller traces each time an event happens.

#print("\n\n==== split before each S.debloquer ====")
# traceset3 = traceset2.with_traces_split(start_action="S.debloquer")
traceset3 = traceset2

# %% Get some data about each trace, to use for clustering traces.

data = traceset3.get_trace_data(method="action_counts")  # or add: method="action_status_counts"
print(data.sum().sort_values())

# %% Now cluster the traces (default MeanShift)

clusterer = MeanShift()
normalizer = MinMaxScaler()
num_clusters = traceset3.create_clusters(data, algorithm=clusterer, normalizer=normalizer)
print(num_clusters, "clusters found - see clusters_out.txt")
with Path("clusters_out.txt").open("w") as out:
    for i in range(num_clusters):
        out.write(f"Cluster {i}:\n")
        for tr in traceset3.get_cluster(i):
            out.write(f"    {tr}\n")

# %% Visualize the coverage of each cluster (the better is to have non-overlapping coverage)

print("Drawing coverage of scanette model by each cluster - see coverage_cluster_<>.svg")
scanetteModel = agilkia.ScanetteModel("scanette.dot")
for i in range(num_clusters):
    plot = scanetteModel.plotableModel()
    plot.plotSequencesWithColors(traceset3.get_cluster(i), agilkia.Color(50, 100, 255))
    plot.export("coverage_cluster_{}.svg".format(i))

# %% Count number of traces in each cluster

counts = Counter(traceset3.get_clusters())
count_pairs = sorted(list(counts.items()))
print("cluster sizes:")
for c,n in count_pairs:
    print(f"    {c:3d}  {n:4d}")

# or graph them:
# plt.bar([x for (x,y) in count_pairs], [y for (x,y) in count_pairs], log=True)

# %% Visualise clusters (using default TSNE) - not great for this example

traceset3.visualize_clusters(block=(not TEST_MODE))

# %% Visualise cluster using PCA to map them into 2D.

# Here is a reasonably good choice of markers for the 13 Scanner clusters.
# (good choice depends upon how many clusters, how many points in each cluster, overlap, etc.)
markers = "1234+x*.<^>vo" # "sphPXd"
xlim = (-1.0,+3.0)
ylim = (-0.65, +0.9)
vis = PCA(n_components=2)
traceset3.visualize_clusters(algorithm=vis, xlim=xlim, ylim=ylim,
                             markers=markers,
                             markersize=9,
                             filename="scanette_clusters.pdf",
                             block=(not TEST_MODE))

# %% To make cluster based on an HMM method, use the folowing lines (the API need to be fixed)

print("Computing clusters with a method based on HMM models")
# we set number of cluster to 3 and models size to 5. Better results can
# be obtained with higher values but it will take more time, so, for this
# sample script we keep it quick
hmm_clusterer = agilkia.HMM_ClusterAlgo(K=3, states=5)


def eventToSymbol(event: agilkia.Event) -> str:
    """
    Transform an event from a scanette trace to a symbol for HMM.
    Several version of this method can be writtent depending on what is wanted in the model.
    """
    symbol = event.action
    status = event.outputs.get('Status')
    if status is not None and event.action != 'payer':
        symbol += " " + str(int(status))
    return symbol


hmm_clusterer.eventToSymbol = eventToSymbol
hmm_clusterer.fit(traceset3)
hmm_clusters = hmm_clusterer.predict(traceset3)  # This return a list of labels for the traces
counts = Counter(hmm_clusters)
count_pairs = sorted(list(counts.items()))
print("cluster sizes:")
for c, n in count_pairs:
    print(f"    {c:3d}  {n:4d}")
hmm_clusterer.visualize(traceset3).exportSingle("hmm_clusters.dot")

# %% Print PCA dimensions

# print(traceset3.get_event_chars())
print(data.columns)
print("PCA components:")
print(vis.components_)
print("PCA explained variance ratios:")
print(vis.explained_variance_ratio_)
print("Total explained variance ratio:", sum(vis.explained_variance_ratio_))


# %% Read traces from functional tests

tests = read_traces_csv(Path("tests_1571927354131.csv"))
event_chars3 = traceset3.get_event_chars()
tests.set_event_chars(event_chars3)
print(f"Reading system tests - one trace, {len(tests[0])} events.")

# %% Split tests into separate traces

tests2 = tests.with_traces_grouped_by("sessionID", property=True)
assert tests2.get_event_chars() == event_chars3

print(f"Split into {len(tests2)} test traces, saved into tests_split.json")
tests2.save_to_json(Path("tests_split.json"))

# %% Do some statistical analysis about action frequencies

test_df = tests2.to_pandas()
print(test_df)
print("\nAction frequencies:\n", test_df["Action"].value_counts())
print("\nStatus frequencies:\n", test_df["Status"].value_counts())

# %% Get the action-count data for the test traces.

test_data = tests2.get_trace_data(method="action_counts")
# NOTE: we must make sure this has same columns in same order!
# print("data.columns     ", list(data.columns))
# print("test_data.columns", list(test_data.columns))

# add the missing 'ajouter' column (all zeroes) before 'fermerSession' (5)
if len(data.columns) < 10:
    test_data.insert(1, 'ajouter', 0)
else:
    test_data.insert(0, 'none', 0)
    keep = ['debloquer_0', 'scanner_0', 'scanner_-2', 'transmission_0', 'abandon_0',
       'ouvrirSession_0', 'none', 'fermerSession_0',
       'payer_-102', 'payer_-42', 'payer_0', 'payer_-4',
       'none',  # transmission_1
       'payer_-1', 'payer_2',
       'none', 'none', 'none', 'none',  # extras
       'supprimer_-2', 'scanner_-1']
    test_data = test_data[keep]
# check that names of columns are the same (but status values can be different)
if not all([c1[:5]==c2[:5] for (c1,c2) in zip(data.columns,test_data.columns) if c2 != "none"]):
    print("ERROR: test columns are not the same as data.columns.")
    print("data.columns=", data.columns)
    print("test.columns=", test_data.columns)

# %% Put the test traces into the original clusters from above.

# NOTE: with action_status_counts, we cannot use same normalizer here, because of error:
#
#  File "/Users/utting/anaconda3/envs/bus-testing/lib/python3.7/site-packages/sklearn/preprocessing/data.py", line 389, in transform
#    X *= self.scale_
#
#ValueError: operands could not be broadcast together with shapes (30,20) (21,) (30,20)

test_clusters = clusterer.predict(normalizer.transform(test_data))
print("tests clustered - see test_clusters_out.txt")
print(test_clusters)
with Path("test_clusters_out.txt").open("w") as out:
    for i in range(num_clusters):
        out.write(f"Cluster {i}:\n")
        for pos, cluster in enumerate(test_clusters):
            if cluster == i:
                out.write(f"    {tests2[pos]}\n")

# %% Check that the tests are using the same event_chars for display.

print(tests2.get_event_chars())

# %% Count the tests in each cluster and graph them vs customer clusters

test_counts = Counter(test_clusters)
test_pairs = sorted(list(test_counts.items()))
plt.bar([x for (x,y) in count_pairs], [y for (x,y) in count_pairs], width=0.4, log=True)
plt.bar([x+0.4 for (x,y) in test_pairs], [y for (x,y) in test_pairs], width=0.4, log=True)
plt.savefig("cust_vs_tests_log.pdf")
plt.show(block=(not TEST_MODE))
print("Number of tests =", sum([v for (_,v) in test_pairs]))

# %% check test cluster results (when regression testing)

assert test_pairs == [(0, 7), (1, 9), (2, 2), (6, 2), (7, 8), (10, 1), (11, 1)]

# %% Visualise them into the SAME clusters by passing in the pre-fitted models

#print(vis.transform(test_data))
n = tests2.create_clusters(test_data, algorithm=clusterer, normalizer=normalizer, fit=False)
print("clusters=", n)
print(tests2.get_clusters())
print(len(tests2.get_clusters()))

# %%

tests2.visualize_clusters(algorithm=vis, fit=False,
                          xlim=xlim, ylim=ylim,
                          markers=markers, markersize=9,
                          filename="test_clusters.pdf",
                          block=(not TEST_MODE))


# %%

print("Trying 3D plot...")
vis3 = PCA(n_components=3)
traceset3.visualize_clusters(algorithm=vis3, block=(not TEST_MODE))

plot_data = vis3.transform(normalizer.transform(data))
print(plot_data[:10])

plot_test = vis3.transform(normalizer.transform(test_data))
print(plot_test[:10])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = pltcm.get_cmap('hsv')

ax.scatter(plot_data[:, 0], plot_data[:, 1], plot_data[:, 2],
           c=traceset3.get_clusters(), cmap=colors)
ax.scatter(plot_test[:, 0], plot_test[:, 1], plot_test[:, 2],
           c=tests.get_clusters(), cmap=colors, marker="+")
plt.show(block=(not TEST_MODE))
