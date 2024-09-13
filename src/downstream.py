import pandas as pd 
import numpy as np 
import math
from typing import List, Dict, Any

def get_between_year_snapshot(
        year1:int, 
        year2:int, 
        nodes_all:pd.DataFrame, 
        flip_year1:List=[], 
        flip_year2:List=[],) -> pd.DataFrame:
    cols = ["node_label", "x", "y"]
    snapshot1 = nodes_all[nodes_all["year"]==year1][cols].rename(columns={"x":"x1", "y":"y1"})
    snapshot2 = nodes_all[nodes_all["year"]==year2][cols].rename(columns={"x":"x2", "y":"y2"})
    snapshots = snapshot1.merge(snapshot2, on="node_label")
    if year1 in flip_year1:
        snapshots["y1"] = -snapshots["y1"]
    if year2 in flip_year2:
        snapshots["y2"] = -snapshots["y2"]
    snapshots["delta_x"] = snapshots["x2"] - snapshots["x1"]
    snapshots["delta_y"] = snapshots["y2"] - snapshots["y1"]
    return snapshots


def sum_node_movement(X1, X2, Y1, Y2) -> float:
    delta_X = np.array(X2) - np.array(X1)
    delta_Y = np.array(Y2) - np.array(Y1)
    movement = 0
    for dx, dy in zip(delta_X, delta_Y):
        movement += np.sqrt(dx**2 + dy**2)
    return movement


def find_optimal_rotation(snapshot, unit=1):
    degrees = np.arange(unit, 360+unit, unit)
    # snapshot = get_between_year_snapshot(year_to_rotate, year_to_base, nodes_all)
    opt_degree = unit  # initialize with the unit degree
    opt_movement = 1e5  # initialize with an arbitrary large number
    X_orig = snapshot["x1"].tolist()
    Y_orig = snapshot["y1"].tolist()
    X_base = snapshot["x2"].tolist()
    Y_base = snapshot["y2"].tolist()
    for d in degrees:
        X_new, Y_new = rotate_point(X_orig, Y_orig, d)
        move = sum_node_movement(X_new, X_base, Y_new, Y_base)
        if move < opt_movement:
            opt_movement = move 
            opt_degree = d 
    return opt_degree, opt_movement


def rotate_point(X:List, Y:List, degree:int) -> [List, List]:
    # code reference: https://gis.stackexchange.com/questions/414600/rotating-a-set-of-points-with-an-angle-with-rotation-matrix-why-is-result-dist

    alpha = -degree * math.pi/180
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)

    X_new = np.array(X) - X_mean
    Y_new = np.array(Y) - Y_mean

    X_apu = math.cos(alpha)*X_new - math.sin(alpha)*Y_new 
    Y_apu = math.sin(alpha)*X_new + math.cos(alpha)*Y_new

    X_new = X_apu + X_mean
    Y_new = Y_apu + Y_mean

    return X_new, Y_new