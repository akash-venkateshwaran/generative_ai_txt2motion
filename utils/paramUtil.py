import numpy as np

# Define a kinematic tree for the skeletal struture
# Chain: right lower - left lower - spine - right upper - left upper
kit_kinematic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]]

kit_raw_offsets = np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [-1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, 1]
    ]
)


kit_tgt_skel_id = "03950"


# neighbor list for the kinematic tree, except self-connection
kit_adj_list = [
    [1, 11, 16],
    [0, 2],
    [1, 3],
    [2, 4, 5, 8],
    [3],
    [3, 6],
    [5, 7],
    [6],
    [3, 9],
    [8, 10],
    [9],
    [0, 12],
    [11, 13],
    [12, 14],
    [13, 15],
    [14],
    [0, 17],
    [16, 18],
    [17, 19],
    [18, 20],
    [19],
]