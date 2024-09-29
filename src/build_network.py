import numpy as np 
import pandas as pd
import networkx as nx 
from typing import List, Dict, Any

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.functions import make_dummy_df, phi, corr_nan
from src.downstream import rotate_point 
from src.utils.network_tools import partition_girvan_newman


class ResIN():
    def __init__(self, 
                 df:pd.DataFrame, 
                 node_cols:List = [],
                 weight_col:str = "") -> None:
        
        self.input_df = df
        self.dummy_df = make_dummy_df(df, node_cols=node_cols, weight_col=weight_col)
        self.dummy_df.replace(np.NaN, "Ref", inplace=True)
        self.node_list = list(set(self.dummy_df.columns) - set(["weights"]))
        self.node_attrs = {}
        
    def make_graph(self,
                   alpha:float = .05,
                   get_p:bool = True,
                   remove_nan:bool = False,
                   remove_non_significant:bool = False,
                   exclude_same_question:bool = True,
                   print_:bool = False,
                   square_corr:bool = False,
                   ) -> None:
        
        if get_p==False and remove_non_significant==True:
            print("Warning: Setting remove_non_significant to False as get_p is False!")
            remove_non_significant=False
        
        # list_of_nodes = self.dummy_df.columns
        self.g = nx.Graph()
        self.edge_weights = {} # record both positive and negative weights
        
        count = 0
        for i, node_i in enumerate(self.node_list):
            for j, node_j in enumerate(self.node_list):
                
                if j <= i: # do not run the same couple twice
                    continue
                
                if print_:
                    count += 1
                    l = len(self.node_list)
                    n_tot = l*(l-1)/2
                    print(count,"/",n_tot, " = ", np.round(count/n_tot,decimals=2)*100, '%')
                    
                basename1 = node_i.split(sep=':')[0]
                basename2 = node_j.split(sep=':')[0]
                
                if exclude_same_question:
                    if basename1 == basename2: # if they belong to the same item
                        continue # skip this pair 

                # Get the two columns
                c1 = self.dummy_df[node_i]
                c2 = self.dummy_df[node_j]
                
                if remove_nan: # if skipping nan values
                    if ("Ref" in node_i) or ("Ref" in node_j): # then, if any node is a nan response item
                        continue # skip this pair of node
                    
                    # for other non-na response items, we will remove rows of responses that returned nan to this question
                    if basename1+":Ref" in self.dummy_df.columns: # first check if there's a nan reponse to this question
                        c1_n = self.dummy_df[basename1+":Ref"] # get the refused values of each item
                    else: # if there's no nan response to this question
                        c1_n = self.dummy_df[node_i].replace(True, False) # we create a null mask 
                    if basename2+":Ref" in self.dummy_df.columns:
                        c2_n = self.dummy_df[basename2+":Ref"]
                    else:
                        c2_n = self.dummy_df[node_j].replace(True, False)

                    mask = np.logical_not(np.logical_or(c1_n, c2_n)) # get a mask of the refused values
                    
                    c1 = c1[mask] # select only the non-nan element
                    c2 = c2[mask]
                
                if get_p:
                    (r,p) = phi(c1,c2, get_p=True)
                else:
                    r = phi(c1,c2, get_p=False)
                
                # Check if there are the conditions for drawing a node
                if remove_non_significant: 
                    condition = r>0 and p<alpha
                else:
                    condition = r>0

                if condition:
                    if square_corr:
                        self.g.add_weighted_edges_from([(node_i,node_j,r**2)],weight="weight")
                    else:
                        self.g.add_weighted_edges_from([(node_i,node_j,r)],weight="weight")
                    if get_p:
                        self.g.add_weighted_edges_from([(node_i,node_j,p)],weight="p")
                        sig = float(p<alpha) # Boolean are not accepted as edge weight
                        self.g.add_weighted_edges_from([(node_i,node_j,sig)],weight="sig")
                
                self.edge_weights[(node_i, node_j)] = r


    # def make_heatmap(self, df_heat, type="standard"):
    #     dic_ = dict()

    #     type_="standard" #"sign"

    #     col_heat = df_heat["ThermoRep"]

    #     for node in self.g.nodes:
    #         col_node = self.dummy_df[node]

    #         (r,p) = corr_nan(col_node, col_heat)

    #         if type_=="sign":
    #             dic_[node] = np.sign(r)
    #         elif type_=="standard":
    #             dic_[node] = r

    #     nx.set_node_attributes(self.g, dic_, "ThermoRep")


    def compute_covariates(self, 
                           covariate_cols:List=[], 
                           remove_nan:bool=True):
        for this_covariate in covariate_cols:
            self.node_attrs[this_covariate] = {}
            for col in self.node_list:
                if remove_nan:
                    if "Ref" in col:
                        continue
                item,val = col.split(":")
                # print(col)
                subdf = self.input_df[self.input_df[item].astype(str)==str(val)][[this_covariate]].dropna()
                # print(f"values for {this_covariate} == {val}", len(subdf[this_covariate]))
                # print(np.nanmean(subdf[this_covariate]))
                if len(subdf) == 0:
                    print(col)
                self.node_attrs[this_covariate][col] = np.mean(subdf[this_covariate].astype(int))


    def get_item_size(self) -> None:
        self.node_attrs["size"] = {}
        for col in self.node_list:
            item,val = col.split(":")
            subdf = self.input_df[self.input_df[item].astype(str)==val]
            self.node_attrs["size"][col] = len(subdf)


    def adjust_coordinates(self) -> None:
        pos = nx.spring_layout(self.g, iterations=5000, seed=42) # Get the positions with the spring layout
        # Restructure the data type
        pos2 = [[],[]]
        key_list = [] # ordered list of the nodes
        for key in pos:
            pos2[0].append(pos[key][0])
            pos2[1].append(pos[key][1])
            key_list.append(key)

        # Use PCA to rotate the network in such a way that the x-axis is the main one
        pos3 = []
        for key in pos:
            pos3.append([pos[key][0],pos[key][1]])

        pca = PCA(n_components=2, power_iteration_normalizer="none", random_state=42)
        pca.fit(pos3)
        x_pca = pca.transform(pos3)

        # Get the x and y position of each node
        xx = x_pca[:,0]
        yy = x_pca[:,1]

        pos_new = {}
        for n,x,y in zip(list(pos.keys()), xx, yy):
            pos_new[n] = np.array([x,y])

        self.pos_new = pos_new

    
    def left_right_flip(self) -> None:
        pos_new2 = {}
        for n,(x,y) in self.pos_new.items():
            new_x = -x 
            pos_new2[n] = np.array([new_x,y])

        self.pos_new = pos_new2


    def adjust_coordinates_manual_rotate(self, unit:int=1) -> None:
        pos = nx.spring_layout(self.g, iterations=5000)

        degrees = np.arange(unit, 360+unit, unit)
        X_orig = [x[0] for x in pos.values()]
        Y_orig = [x[1] for x in pos.values()]

        opt_degree = unit 
        opt_x_sum = 0
        for d in degrees:
            X_new, Y_new = rotate_point(X_orig, Y_orig, d)
            x_sum = np.sum([np.abs(x) for x in X_new])
            if x_sum > opt_x_sum:
                opt_degree = d 
                opt_x_sum = x_sum
        
        xx, yy = rotate_point(X_orig, Y_orig, opt_degree)
        pos_new_manual = {}
        for n,x,y in zip(list(pos.keys()), xx, yy):
            pos_new_manual[n] = np.array([x,y])

        self.pos_new_manual = pos_new_manual

    def partition_network(
            self, 
            partition_func:Any=partition_girvan_newman, 
            compute_modeularity:bool=True) -> None:
        communities = partition_func(self.g)
        self.node_attrs["community_label"] = {}
        for i, this_comm in enumerate(communities):
            for node in this_comm:
                self.node_attrs["community_label"][node] = i 

        if compute_modeularity:
            mod = nx.community.modularity(self.g, communities, weight="weight")
            self.modularity = mod
        # print("# of communities:", len(communities))

    def measure_linearization(self) -> None:
        xs = [coord[0] for _,coord in self.pos_new.items()]
        ys = [coord[1] for _,coord in self.pos_new.items()]
        spreadx = max(xs) - min(xs)
        spready = max(ys) - min(ys)
        lin = 1 - spready / spreadx 
        self.linearization = lin 

    
    def set_edge_distance(self) -> None:
        ds = {}
        for e in self.g.edges(data=True):
            w = e[2]["weight"]
            ds[(e[0],e[1])] = 1/w
        nx.set_edge_attributes(self.g, ds, "distance")
    



class BNA():
    def __init__(self,
                 df:pd.DataFrame,
                 node_cols:list = []) -> None:
        
        self.input_df = df
        self.node_list = node_cols 
        self.node_attrs = {}

    def make_graph(self,
                   alpha:float = .05,
                   get_p:bool = True,
                   remove_non_significant:bool = False,
                   exclude_same_question:bool = True,
                   print_:bool = False,
                   square_corr:bool = False) -> None:
        
        if get_p==False and remove_non_significant==True:
            print("Warning: Setting remove_non_significant to False as get_p is False!")
            remove_non_significant=False

        
        self.g = nx.Graph()
        self.edge_weights = {}

        count = 0 
        for i, node_i in enumerate(self.node_list):
            for j, node_j in enumerate(self.node_list):

                if j <= i: # do not run the same couple twice
                    continue 
                
                if print_:
                    count += 1 
                    l = len(self.ndoe_list)
                    n_tot = l*(l-1)/2
                    print(count,"/",n_tot, " = ", np.round(count/n_tot,decimals=2)*100, '%')
                
                if exclude_same_question:
                    if node_i == node_j: # skip if they are the same question
                        continue  
                
                # get the two columns 
                c1 = self.input_df[node_i]
                c2 = self.input_df[node_j]

                (r,p) = corr_nan(c1, c2)
                r = abs(r)

                if remove_non_significant:
                    condition = r > 0 and p < alpha
                else:
                    condition = r > 0  

                if condition:
                    if square_corr: 
                        self.g.add_weighted_edges_from([(node_i,node_j,r**2)], weight="weight")
                    else:
                        self.g.add_weighted_edges_from([(node_i,node_j,r)],weight="weight")
                    if get_p:
                        self.g.add_weighted_edges_from([(node_i,node_j,p)],weight="p")
                        sig = float(p<alpha) # Boolean are not accepted as edge weight
                        self.g.add_weighted_edges_from([(node_i,node_j,sig)],weight="sig")
                
                self.edge_weights[(node_i, node_j)] = r 

    def compute_covariates(self, 
                           covariate_col:str) -> None:
        
        p_dict = {}
        r_dict = {}
        for node_col in self.node_list:
            (p,r) = corr_nan(self.input_df[node_col], self.input_df[covariate_col])
            p_dict[node_col] = p 
            r_dict[node_col] = r
        
        nx.set_node_attributes(self.g, p_dict, f"corr_{covariate_col}")
        nx.set_node_attributes(self.g, r_dict, f"sig_{covariate_col}")
                






