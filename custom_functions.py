import airsim
from Bezier import Bezier
import numpy as np
from math import tanh

# Constants
D_MAX = 5.0
R_L = 0.0
R_U = 1.0
DEL_D_L = 0.0
DEL_D_U = 1.0
R_DP = -0.5
R_CP = -1.0
N = 1.0
NODES_0 = np.array([[0.0,0.0,0.0]])
NODES_1 = np.array([[0.0,0.0,-i] for i in np.arange(0.0,N+.01,0.01)])
NODES_2 = np.array([[0.0,-i,-i] for i in np.arange(0.0,N+.01,0.01)])
NODES_3 = np.array([[0.0,-i,0.0] for i in np.arange(0.0,N+.01,0.01)])
NODES_4 = np.array([[0.0,-i,i] for i in np.arange(0.0,N+.01,0.01)])
NODES_5 = np.array([[0.0,0.0,i] for i in np.arange(0.0,N+.01,0.01)])
NODES_6 = np.array([[0.0,i,i] for i in np.arange(0.0,N+.01,0.01)])
NODES_7 = np.array([[0.0,i,0.0] for i in np.arange(0.0,N+.01,0.01)])
NODES_8 = np.array([[0.0,i,-i] for i  in np.arange(0.0,N+.01,0.01)])
NODES_9 = np.array([[i,0.0,0.0] for i in np.arange(0.0,N+.01,0.01)])
NODES_10 = np.array([[0.0,0.0,0.0],[N,0.0,0.0],[0.0,0.0,-N],[N,0.0,-N]])
NODES_11 = np.array([[0.0,0.0,0.0],[N,-N,0.0],[0.0,0.0,-N],[N,-N,-N]])
NODES_12 = np.array([[0.0,0.0,0.0],[N,0.0,0.0],[0.0,-N,0.0],[N,-N,0.0]])
NODES_13 = np.array([[0.0,0.0,0.0],[N,-N,0.0],[0.0,0.0,N],[N,-N,N]])
NODES_14 = np.array([[0.0,0.0,0.0],[N,0.0,0.0],[0.0,0.0,N],[N,0.0,N]])
NODES_15 = np.array([[0.0,0.0,0.0],[N,N,0.0],[0.0,0.0,N],[N,N,N]])
NODES_16 = np.array([[0.0,0.0,0.0],[N,0.0,0.0],[0.0,N,0.0],[N,N,0.0]])
NODES_17 = np.array([[0.0,0.0,0.0],[N,N,0.0],[0.0,0.0,-N],[N,N,-N]])
NODES_LIST = [NODES_0,NODES_1,NODES_2,NODES_3,NODES_4,NODES_5,
                  NODES_6,NODES_7,NODES_8,NODES_9,NODES_10,NODES_11,
                  NODES_12,NODES_13,NODES_14,NODES_15,NODES_16,NODES_17]
N_LINEAR_PATHS = 10
N_POINTS = np.arange(0.0,1.0,0.01)


# input: MultirotorClient to get path for, index of motion primitive
# output: Path in world frame for client to execute
def getPath(client: airsim.MultirotorClient,i: int) -> list:
    # state data of drone
    state = client.getMultirotorState()
    orientation = state.kinematics_estimated.orientation
    pos = state.kinematics_estimated.position
    # list to hold the path
    p = list()
    # nodes that define linear path
    path = NODES_LIST[i]
    # nodes that define Bezier curve
    if i>=N_LINEAR_PATHS:
        path = Bezier.Curve(N_POINTS,path)
    # convert each point in path from multirotor body frame to world frame
    for point in path:
        # print(path)
        body_frame_vec = airsim.Vector3r(point[0],point[1],point[2])
        world_frame_vec = pos + orientation*body_frame_vec.to_Quaternionr()*orientation.star()
        p.append(world_frame_vec)   
    return p

# input: MultirotorClient
# output: list of motion primitive paths
def generateMotionPrimitives(client: airsim.MultirotorClient) -> list:
    nodes_list = [NODES_0,NODES_1,NODES_2,NODES_3,NODES_4,NODES_5,
                  NODES_6,NODES_7,NODES_8,NODES_9,NODES_10,NODES_11,
                  NODES_12,NODES_13,NODES_14,NODES_15,NODES_16,NODES_17]
    # number of linear paths (Nodes 0 - 9) kept as variable in case it changes in future
    
    # list of motion primitives to output
    primitives_list = list()
    # state data of drone
    state = client.getMultirotorState()
    orientation = state.kinematics_estimated.orientation
    pos = state.kinematics_estimated.position
    for i in range(len(nodes_list)):
        # nodes that define linear path
        path = NODES_LIST[i]
        # nodes that define Bezier curve
        if i>=N_LINEAR_PATHS:
            path = Bezier.Curve(N_POINTS,path)
        # convert each point in path from multirotor body frame to world frame
        for point in path:
            # print(path)
            body_frame_vec = airsim.Vector3r(point[0],point[1],point[2])
            world_frame_vec = pos + orientation*body_frame_vec.to_Quaternionr()*orientation.star()
            p.append(world_frame_vec)   
        primitives_list.append(p)
        
    return primitives_list

# execute the motion primitive from the list based on index i with velocity vel
# input: MultirotorClient to move, index of motion primitive, desired velocity
# output: None
def execute_motion_primitive(client, i: int, vel: float) -> None:
    p = getPath(client, i)
    client.moveOnPathAsync(path=p,velocity=vel).join()
    client.hoverAsync().join()

# reward function for the RL algorithm based on Camci et al.
# input: euclidean distance at beginning of timestep, at end of timestep, boolean asserted if drone has collided
# output: reward
def reward_function(d_t_min: float,d_t: float,collision: bool) -> float:
    if collision:
        return R_CP
    if abs(d_t)>D_MAX:
        return R_DP
    f = 0.5*(tanh((2*D_MAX-d_t)/D_MAX)+1)
    del_d = d_t-d_t_min
    if DEL_D_U < del_d:
        return R_L*f
    if DEL_D_L <= del_d and del_d <= DEL_D_U:
        return (R_L+(R_U-R_L)*(DEL_D_U-del_d)/(DEL_D_U-DEL_D_L))*f
    if del_d < DEL_D_L:
        return R_U*f
    