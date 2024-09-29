import networkx as nx 
import numpy as np 
import pandas as pd 

from src.build_network import ResIN
from typing import List, Dict, Any

# several operationalizations of polarization measure

# def set_edge_distance(g:nx.Graph):
#     ds = {}
#     for e in g.edges(data=True):
#         w = e[2]["weight"]
#         ds[(e[0],e[1])] = 1/w
#     nx.set_edge_attributes(g, ds, "distance")


## ------------------------- ##
## ----- network-level ----- ##
## ------------------------- ##

VAL_RANGES = {
    "spend_serv":(1.0,7.0),
    "gov_health":(1.0,7.0),
    "guar_jobs":(1.0,7.0),
    "abort":(1.0,4.0),
    "aid_black":(1.0,7.0),
}

def get_binary_comm(resin:ResIN, comm_attr:str) -> List:
    comms = [set(), set()]
    for n,v in resin.node_attrs[comm_attr].items():
        if v > 0:
            comms[0].add(n)
        else:
            comms[1].add(n)
    return comms 

def get_modularity(resin:ResIN, comms:List) -> float:
    mod = nx.community.modularity(resin.g, comms)
    return mod

def get_diameter(resin:ResIN, weight_col="weight") -> float:
    return nx.diameter(resin.g, weight=weight_col)

def get_linearization(resin:ResIN) -> float:
    pos = resin.pos_new
    xmax = max([x[0] for x in pos.values()])
    xmin = min([x[0] for x in pos.values()])
    ymax = max([x[1] for x in pos.values()])
    ymin = min([x[1] for x in pos.values()])
    lin = (xmax - xmin)/(ymax - ymin)
    return lin

def get_pol_distance(resin:ResIN, 
                     val_ranges:Dict = VAL_RANGES,
                     weight_col:str = "distance") -> List:
    ds = []
    for c,(cmin,cmax) in val_ranges.items():
        d = nx.shortest_path_length(
            resin.g, 
            f"{c}:{cmin}",
            f"{c}:{cmax}",
            weight=weight_col)
        ds.append(d)
    return ds

def m(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)

def cov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - m(x, w)) * (y - m(y, w))) / np.sum(w)

def corr(x, y, w):
    """Weighted Correlation"""
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))

def get_assortativity(resin:ResIN, 
                          node_attr:str="ft_rep_net"):
    g = resin.g
    node_attrs = resin.node_attrs[node_attr]
    edge_wghts = nx.get_edge_attributes(g, "weight")
    x = []
    y = []
    w = []
    for n1,n2 in g.edges():
        x.append(node_attrs[n1])
        y.append(node_attrs[n2])
        w.append(edge_wghts[(n1, n2)])
    c = corr(np.array(x), np.array(y), np.array(w))
    return c



## ---------------------- ##
## ----- node-level ----- ##
## ---------------------- ##

# def get_bridging(resin:ResIN, weight_col = "weight"):

def get_centrality_df(resin:ResIN,
                      strength:bool = True,
                      betweenness:bool = True,
                      closeness:bool = True) -> pd.DataFrame:
    g = resin.g
    df = pd.DataFrame()
    df["node"] = list(g.nodes())
    if strength:
        df["strength"] = [x[1] for x in nx.degree(g, weight="weight")]
    if betweenness:
        df["betweenness"] = dict(nx.betweenness_centrality(g, weight="distance")).values()
    if closeness:
        df["closeness"] = dict(nx.closeness_centrality(g, distance="distance")).values()
    return df

## ---------------------------- ##
## ----- individual-level ----- ##
## ---------------------------- ##

PARTISAN_LABEL_CATG = {
    1: -1, 2: -1, 3: -1, 
    4: 0,
    5: 1, 6: 1, 7: 1,
}

def calculate_survey_polar(anes_df, node_cols):
    max_val = np.sum([anes_df[col].max() for col in node_cols])
    anes_df["lnr_ideo_csst"] = anes_df.apply(lambda x: 
                                             abs(1-2*np.nansum([x[col] for col in node_cols]/max_val)), axis=1)
    return anes_df

def count_all_possible_links(g:nx.Graph):
    possible_links = 0
    for i,n1 in enumerate(g.nodes()):
        for j,n2 in enumerate(g.nodes()):
            if j>i:
                iss1 = n1.split(":")[0]
                iss2 = n2.split(":")[0]
                if iss1 != iss2:
                    possible_links += 1 
    return possible_links


def get_density_weighted(g:nx.Graph, weight_col:str="weight"):
    s = np.sum([e[2]["weight"] for e in g.edges(data=True)])
    n = len(g.nodes())
    # denm = n*(n-1)/2
    denm = count_all_possible_links(g)
    return s/denm

def get_respondent_polar(resin, response_dict:dict, issues:List) -> pd.DataFrame:
    respondent_df = pd.DataFrame()
    sort_ls = []
    affp_ls = []
    part_ls = []
    ln_sort_ls = []
    g = resin.g
    for _, response in response_dict.items():
        sub_nodes = [f"{x}:{response[x]}" for x in issues if not np.isnan(response[x])]
        subg = g.subgraph(sub_nodes) # get subgraph 
        sort_ls.append(get_density_weighted(subg))
        affp_ls.append(response["ft_rep_net_abs"])
        part_ls.append(response["dem_rep_3"])
        ln_sort_ls.append(response["lnr_ideo_csst"])
    respondent_df["sort"] = sort_ls 
    respondent_df["affp"] = affp_ls
    respondent_df["part"] = part_ls
    respondent_df["ln_sort"] = ln_sort_ls
    respondent_df.dropna(inplace=True)

    return respondent_df