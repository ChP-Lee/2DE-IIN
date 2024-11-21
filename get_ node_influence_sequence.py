from Node_influence import _2DSE
from datetime import datetime
import pandas as pd
from ogb.lsc import MAG240MDataset



def start(g_path,t_path,name):
    if name == "MAG240M":
        dataset = MAG240MDataset(root=g_path)
        edges = dataset.edge_index('paper', 'paper').T
        t_path = dataset.paper_label

    else:
        df = pd.read_csv(g_path, delimiter='\t', skiprows=4, header=None, names=['FromNodeId', 'ToNodeId'])
        edges = df.values
    print(datetime.now().strftime("%m/%d %H:%M:%S"))
    start = datetime.now().strftime("%m/%d %H:%M:%S")

    seg = _2DSE(edges)
    node_inf = seg.fit(max_item=10, verbose=True, patience=2)
    print(node_inf[:20])
    end = datetime.now().strftime("%m/%d %H:%M:%S")
    print(datetime.now().strftime("%m/%d %H:%M:%S"))
    start_time = datetime.strptime(start, "%m/%d %H:%M:%S")
    end_time = datetime.strptime(end, "%m/%d %H:%M:%S")
    time_diff = end_time - start_time
    seconds = time_diff.total_seconds()

    print(f"time consuming:{seconds}")



DATASETS={
    "amazon":["",""],
    "youtube":["",""],
    "LiveJournal":["",""],
    "dblp":["",""],
    "orkut":["",""],
    "MAG240M":["",""]
}

if __name__ == "__main__":
    for name in [ "amazon" ,"youtube", "dblp","LiveJournal","orkut"]:

        print("-" * 40)
        print(f"dataset:{name}")
        start(DATASETS[name][0],DATASETS[name][1],name)
