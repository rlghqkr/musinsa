import numpy as np
from scipy.optimize import linear_sum_assignment

def layout_distance(bg_sig, cand_layout):
    # bg_sig: {boxes, labels}
    # cand_layout: {boxes, labels}
    B = len(bg_sig['boxes'])
    C = len(cand_layout['boxes'])
    if B == 0 or C == 0:
        return 1.0
    cost = np.zeros((B,C))
    for i in range(B):
        xi1, yi1, xi2, yi2 = bg_sig['boxes'][i]
        xc_i, yc_i = (xi1+xi2)/2, (yi1+yi2)/2
        li = bg_sig['labels'][i]
        for j in range(C):
            xj1, yj1, xj2, yj2 = cand_layout['boxes'][j]
            xc_j, yc_j = (xj1+xj2)/2, (yj1+yj2)/2
            lj = cand_layout['labels'][j]
            cat = 0.0 if li==lj else 1.0
            dist = np.hypot(xc_i-xc_j, yc_i-yc_j)
            cost[i,j] = 0.6*cat + 0.4*dist
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].mean())
