import numpy as np 
import json

## Adjacency Matrix
def mAdjContour(N):
    sub = np.zeros([N, N])
    for i in range(0, N):
        sub[i, i-1] = 1
        sub[i, (i+1)%N] = 1
    return sub

## Downsampling Matrix
def mDownsampling(N):
    N2 = N//2
    sub = np.zeros([N2, N])
    
    for i in range(0, N2):
        if (2*i+1) == N:
            sub[i, 2*i] = 1
        else:
            sub[i, 2*i] = 1/2
            sub[i, 2*i+1] = 1/2
            
    return sub

## Upsampling Matrix
def mUpsampling(N):
    N2 = N//2
    sub = np.zeros([N, N2])
    
    for i in range(0, N):
        if i % 2 == 0:
            sub[i, i//2] = 1
        else:
            sub[i, i//2] = 1/2
            sub[i, (i//2 + 1) % N2] = 1/2
            
    return sub

### To create the dataset

def generate_adjacency_matrices(shapes, config):
    adjacency_matrices = {res: {} for res in config["resolutions"]}
    downsampling_matrices = {f"to_{config['resolutions'][i+1]}": {} for i in range(len(config['resolutions'])-1)}
    upsampling_matrices = {f"to_{config['resolutions'][i]}": {} for i in range(len(config['resolutions'])-1)}
    
    max_overall = 0
    for organ in config["organs"]:
        mean = np.mean(shapes[organ])
        if mean > max_overall:
            max_overall = mean
    
    for organ in config["organs"]:
        mean = np.mean(shapes[organ])
        max_factor = 2 ** (len(config["resolutions"]) - 1)
        min_resolution = 16 * max_factor

        scaled_mean = int(np.round(mean * config["scale_factor"]))
        scaled_mean = max(min_resolution, round(scaled_mean / max_factor) * max_factor)

        for i, res in enumerate(config["resolutions"]):
            adjacency_matrices[res][organ] = mAdjContour(scaled_mean // 2 ** i)
        
        for i in range(len(config["resolutions"]) - 1):
            downsampling_matrices[f"to_{config['resolutions'][i+1]}"][organ] = mDownsampling(scaled_mean // (2**i))
            upsampling_matrices[f"to_{config['resolutions'][i]}"][organ] = mUpsampling(scaled_mean // (2**i))

        print(f"Organ {organ} - Contour len mean: {mean} - Atlas contour size: {scaled_mean}")
        print(f"Adjacency matrices sizes: {[(res, len(adjacency_matrices[res][organ])) for res in config['resolutions']]}")
        print(f"Downsampling matrices sizes: {[(res, len(downsampling_matrices[res][organ])) for res in downsampling_matrices.keys()]}")
    
    return adjacency_matrices, downsampling_matrices, upsampling_matrices

def create_block_diagonal_matrix(matrices):
    N = np.sum([adj.shape[0] for adj in matrices.values()])
    block_diagonal = np.zeros([N, N])
    organ_id = np.zeros([N, 1])
    i = 0
    ordered_keys = []
    for k, adj in matrices.items():
        organ_id[i:i+adj.shape[0]] = k
        n = adj.shape[0]
        block_diagonal[i:i+n, i:i+n] = adj
        i += n
        ordered_keys.append(k)
    return block_diagonal, organ_id, np.array(ordered_keys)

def save_matrices(matrices, name, config):
    block_diagonal, organ_id, ordered_keys = create_block_diagonal_matrix(matrices)
    np.save(f"{config['output_path']}/Independent/adj_{name}_block_diagonal.npy", block_diagonal)
    np.save(f"{config['output_path']}/Independent/adj_{name}_organ_id.npy", organ_id)
    np.savetxt(f"{config['output_path']}/Independent/adj_{name}_organ_id.txt", ordered_keys, fmt="%s")
    json.dump({k: adj.astype('int').tolist() for k, adj in matrices.items()},
              open(f"{config['output_path']}/Independent/adjacency_matrices_{name}.json", "w"))

def create_sampling_matrix(matrices):
    N1 = np.sum([m.shape[0] for m in matrices.values()])
    N2 = np.sum([m.shape[1] for m in matrices.values()])
    block_matrix = np.zeros([N1, N2])
    i1, i2 = 0, 0
    for m in matrices.values():
        n1, n2 = m.shape
        block_matrix[i1:i1+n1, i2:i2+n2] = m
        i1 += n1
        i2 += n2
    return block_matrix


def generate_edge_info(config):
    for block in config["resolutions"]:
        block_diagonal = np.load(f"{config['output_path']}/Independent/adj_{block}_block_diagonal.npy")
        edges = np.argwhere(block_diagonal == 1)
        edges = edges[edges[:, 0] < edges[:, 1]]
        np.save(f"{config['output_path']}/Independent/all_edges_{block}.npy", edges)
        
        adjacency_matrices = json.load(open(f"{config['output_path']}/Independent/adjacency_matrices_{block}.json", "r"))
        N_organs = len(adjacency_matrices.keys())
        M_edges = [np.sum(np.array(adj)) for adj in adjacency_matrices.values()]
        max_edges = np.round(np.max(M_edges) / 2).astype(int)  # Each edge is counted twice in the adjacency matrix
        edge_matrix = np.zeros([N_organs, max_edges, 2])
        
        i = 0
        for c, (k, adj) in enumerate(adjacency_matrices.items()):
            adj = np.array(adj)
            edges = np.argwhere(adj == 1)
            edges = edges[edges[:, 0] < edges[:, 1]] + i
            edge_matrix[c, :edges.shape[0], :] = edges
            edge_matrix[c, edges.shape[0]:, :] = 0
            i += adj.shape[0]
        
        np.save(f"{config['output_path']}/Independent/edge_matrix_{block}.npy", edge_matrix)

"""    
def generate_edge_info(config):
    '''
    Generate edge information for a single organ using only the NPY adjacency matrix
    '''
    for block in config["resolutions"]:
        # Load the unified adjacency matrix
        block_diagonal = np.load(f"{config['output_path']}/Independent/adj_{block}_block_diagonal.npy")
        
        # Find all edges in the graph (only keep upper triangular part to avoid double-counting)
        edges = np.argwhere(block_diagonal == 1)
        edges = edges[edges[:, 0] < edges[:, 1]]  # Keep upper triangular only
        
        # Save all edges for reference
        np.save(f"{config['output_path']}/Independent/edge_matrix_{block}.npy", edges)
        
        print(f"Processed block {block}: Found {len(edges)} edges")
"""