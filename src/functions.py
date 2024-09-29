import numpy as np 
import pandas as pd 
import scipy.stats as stt
from typing import List, Dict, Any

def make_dummy_df(
        df_nodes: pd.DataFrame, 
        node_cols:List=[],
        weight_col:str="") -> pd.DataFrame: # tested ok!
    """Construct a dummy dataframe with all dummy (True/False) attitude items

    Args:
        df_nodes (pd.DataFrame): the input dataframe where each column is an issue and each row is a response item to that issue
        weight_col (str, optional): the name for the weight column, will not be converted into a dummy variable. Defaults to "".

    Returns:
        pd.DataFrame: the output dummy table
    """

    df_dummy = pd.DataFrame()
    df = df_nodes.copy() # Select the dataframe of the nodes

    # here you can select a subset of columns
    # the weight column is always excluded from the node columns
    if len(node_cols) > 0:
        list_of_columns = set(node_cols) - set([weight_col]) 
    else:
        list_of_columns = set(df_nodes.columns) - set([weight_col])
    
    # directly add weight column to the output df 
    if len(weight_col) > 0:
        df_dummy["weights"] = df_nodes[weight_col]

    for col in list_of_columns: # For each column...
        values = (df[col].unique()) # ... get the list of the possible responses (i.e. nodes)
        
        for value in values: # For each response
            if type(value) == str: # check if the answer is type string
                name = str(col)+":"+str(value) # get the names as col:response
                df_dummy[name] = df[col] == value # get dummy-coded column

            else:
                # if np.isnan(value): # if it's a refused answer
                if pd.isnull(value):
                    name = str(col)+":"+"Ref" 
                    # df_dummy[name] = np.isnan(df[col])
                    df_dummy[name] = pd.isnull(df[col])
                else: # Otherwise
                    name = str(col)+":"+str(value) 
                    df_dummy[name] = df[col] == value
    
    return df_dummy


def phi_(n11:int, n00:int, n10:int, n01:int) -> float: # tested ok!
    """ Calculate the phi coefficient for two binary arrays
    # https://en.wikipedia.org/wiki/Phi_coefficient

    Args:
        n11 (int): the count of instances where x = y = 1
        n00 (int): the count of instances where x = y = 0
        n10 (int): the count of instances where x = 1 and y = 0
        n01 (int): the count of instances where x = 0 and y = 1

    Returns:
        float: the value of phi correlation 
    """
    n1p = n11+n10
    n0p = n01+n00
    np1 = n01+n11
    np0 = n10+n00
    
    num = n11*n00-n10*n01
    den_ = n1p*n0p*np0*np1
    
    if den_==0:
        phi_=np.nan
    else:
        phi_ = num/np.sqrt(den_)
    return phi_


def p_val(r:float, L:int) -> float:
    """Get the p-value for the phi coefficient
    """
    den = np.sqrt(1-r**2)
    deg_free = L-2
    if den==0:
        p = 0
    else:
        num = r*np.sqrt(deg_free)
        t = num/den
        p = stt.t.sf(abs(t), df=deg_free)*2
    return p


def phi(x:Any, y:Any, w:Any=None, get_p:bool=False, weighted:bool=False) -> Any: # tested ok!
    """Get phi correlation for a given set of (weighted) binary arrays 

    Args:
        x,y (Any): the binary arrays to correlate
        w (Any, optional): the response weights specified in the survey. Defaults to None.
        get_p (bool, optional): whether to get p-value for the phi correlation. Defaults to False.
        weighted (bool, optional): whether to consider survey weights when computing the phi correlation. Defaults to False.

    Returns:
        Any: the phi correlation (and the p-value, if needed)
    """
    
    m_eq = x==y
    m_diff = np.logical_not(m_eq)

    if weighted: 
        n11 = float(np.matmul(x[m_eq]==True, w[m_eq]))
        n00 = float(np.matmul(x[m_eq]==False, w[m_eq]))
        n10 = float(np.matmul(x[m_diff]==True, w[m_diff]))
        n01 = float(np.matmul(y[m_diff]==True, w[m_diff]))
    else:
        n11 = float(np.sum(x[m_eq]==True))
        n00 = float(np.sum(x[m_eq]==False))
        n10 = float(np.sum(x[m_diff]==True))
        n01 = float(np.sum(y[m_diff]==True))
    
    phi_val = phi_(n11,n00,n10,n01)
    
    if get_p:
        p = p_val(phi_val, len(x))
        return phi_val, p
    else:
        return phi_val
    

def corr_nan(x,y):
    # get the pearson's correlation r for two arrays that potentially have nan values
    x = np.array(x)
    y = np.array(y)
    
    m = np.isnan(x) + np.isnan(y)
    mm = np.logical_not(m)
    
    [r,p] = stt.pearsonr(x[mm],y[mm])
    
    return (r,p)