import os
import pandas as pd


def preproc(datasetName, datasetFolder=None, full=False):
    full = "full-" if full else ""
    if datasetFolder:
        nverts_path = os.path.join(datasetFolder, "{}-{}nverts.txt".format(datasetName, full))
        simplices_path = os.path.join(datasetFolder, "{}-{}simplices.txt".format(datasetName, full))
        times_path = os.path.join(datasetFolder, "{}-{}times.txt".format(datasetName, full))
    else:
        nverts_path = 'data\\{}\\{}-{}nverts.txt'.format(datasetName, datasetName, full)
        simplices_path = 'data\\{}\\{}-{}simplices.txt'.format(datasetName, datasetName, full)
        times_path = 'data\\{}\\{}-{}times.txt'.format(datasetName, datasetName, full)

    with open(nverts_path, 'r') as f:
        nverts = [int(line) for line in f.readlines()]

    with open(simplices_path, 'r') as f:
        simplices = [int(line) for line in f.readlines()]

    with open(times_path, 'r') as f:
        times = [int(line) for line in f.readlines()]

    N = len(set(simplices))
    E = len(nverts)

    print("Dataset name: {}\nNum. nodes: {}\tNum hyperedges: {}".format(datasetName, N, E))

    ctr = 0
    id, nodes, times_list = [], [], []
    for i in range(E):
        he_id = i
        n_nodes = nverts[he_id]
        first = ctr
        last = ctr + n_nodes
        he_nodes = ",".join([str(x) for x in simplices[first:last]])
        edge_time = times[he_id]

        id.append(he_id + 1)
        nodes.append(he_nodes)
        times_list.append(edge_time)

        ctr = last

    data = {
        "id": id,
        "nodes": nodes,
        "time": times_list
    }

    return pd.DataFrame(data=data)


dataName = "congress-bills"
data_folder = os.path.join("data", dataName)

df = preproc(dataName, full=False)
print(df.head())
csv_path = os.path.join(data_folder, "{}-hypergraph.txt".format(dataName))
df.to_csv(csv_path, sep=";", index=False)
