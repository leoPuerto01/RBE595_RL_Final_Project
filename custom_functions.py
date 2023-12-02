import airsim
from Bezier import Bezier
import numpy as np
N = 1.0
NODES_11 = np.array([[0.0,0.0,0.0],[N,-N,0.0],[0.0,0.0,-N],[N,-N,-N]])

N_POINTS = np.arange(0.0,1.0,0.01)



def getPath():
    path = Bezier.Curve(N_POINTS,NODES_11)
    p = list()
    for point in path:
        p.append(airsim.Vector3r(point[0],point[1],point[2]))
        
    return p