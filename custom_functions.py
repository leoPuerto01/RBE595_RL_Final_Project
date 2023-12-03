import airsim
from Bezier import Bezier
import numpy as np

# Constants
N = 1.0
NODES_0 = np.array([[0.0,0.0,0.0]])
NODES_1 = np.array([[0.0,0.0,-i] for i in np.arange(0.0,N+.01,0.01)])
NODES_2 = np.array([[0.0,-i,-i] for i in np.arange(0.0,N+.01,0.01)])
NODES_3 = np.array([[0.0,-i,0.0] for i in np.arange(0.0,N+.01,0.01)])
NODES_4 = np.array([[0.0,-i,i] for i in np.arange(0.0,N+.01,0.01)])
NODES_5 = np.array([[0.0,0.0,i] for i in np.arange(0.0,N+.01,0.01)])
NODES_6 = np.array([[0.0,i,i] for i in np.arange(0.0,N+.01,0.01)])
NODES_7 = np.array([[0.0,i,0.0] for i in np.arange(0.0,N+.01,0.01)])
NODES_8 = np.array([[0.0,i,-i] for i in np.arange(0.0,N+.01,0.01)])
NODES_9 = np.array([[i,0.0,0.0] for i in np.arange(0.0,N+.01,0.01)])
NODES_10 = np.array([[0.0,0.0,0.0],[N,0.0,0.0],[0.0,0.0,-N],[N,0.0,-N]])
NODES_11 = np.array([[0.0,0.0,0.0],[N,-N,0.0],[0.0,0.0,-N],[N,-N,-N]])
NODES_12 = np.array([[0.0,0.0,0.0],[N,0.0,0.0],[0.0,-N,0.0],[N,-N,0.0]])
NODES_13 = np.array([[0.0,0.0,0.0],[N,-N,0.0],[0.0,0.0,N],[N,-N,N]])
NODES_14 = np.array([[0.0,0.0,0.0],[N,0.0,0.0],[0.0,0.0,N],[N,0.0,N]])
NODES_15 = np.array([[0.0,0.0,0.0],[N,N,0.0],[0.0,0.0,N],[N,N,N]])
NODES_16 = np.array([[0.0,0.0,0.0],[N,0.0,0.0],[0.0,N,0.0],[N,N,0.0]])
NODES_17 = np.array([[0.0,0.0,0.0],[N,N,0.0],[0.0,0.0,-N],[N,N,-N]])



N_POINTS = np.arange(0.0,1.0,0.01)



def getPath(client):
    path = Bezier.Curve(N_POINTS,NODES_11)
    p = list()
    state = client.getMultirotorState()
    orientation = state.kinematics_estimated.orientation
    pos = state.kinematics_estimated.position
    for point in path:
        body_frame_vec = airsim.Vector3r(point[0],point[1],point[2])
        world_frame_vec = pos + orientation*body_frame_vec.to_Quaternionr()*orientation.star()
        p.append(world_frame_vec)
        
        
    return p

# input: MultirotorClient
# output: list of motion primitive paths
def generateMotionPrimitives(client):
    nodes_list = [NODES_0,NODES_1,NODES_2,NODES_3,NODES_4,NODES_5,
                  NODES_6,NODES_7,NODES_8,NODES_9,NODES_10,NODES_11,
                  NODES_12,NODES_13,NODES_14,NODES_15,NODES_16,NODES_17]
    # number of linear paths (Nodes 0 - 9) kept as variable in case it changes in future
    n_linear_paths = 10
    # list of motion primitives to output
    primitives_list = list()
    # state data of drone
    state = client.getMultirotorState()
    orientation = state.kinematics_estimated.orientation
    pos = state.kinematics_estimated.position
    for i in range(len(nodes_list)):
        p = list()
        # nodes that define linear path
        path = nodes_list[i]
        # nodes that define Bezier curve
        if i>=n_linear_paths:
            path = Bezier.Curve(N_POINTS,path)
        # convert each point in path from multirotor body frame to world frame
        for point in path:
            # print(path)
            body_frame_vec = airsim.Vector3r(point[0],point[1],point[2])
            world_frame_vec = pos + orientation*body_frame_vec.to_Quaternionr()*orientation.star()
            p.append(world_frame_vec)
        primitives_list.append(p)
        
    return primitives_list
            
        