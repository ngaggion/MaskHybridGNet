import cv2
import numpy as np
import networkx as nx
import json
from pathlib import Path

UNIFIED_DATA_DIR = "Unified"


def unified_data_path(config):
    """Return the on-disk directory for unified-representation artifacts."""
    return str(Path(config["output_path"]) / UNIFIED_DATA_DIR)

def build_unified_contour_graph(mask, distance_threshold=1.8):
    """Build a unified contour graph from a multi-organ segmentation mask"""
    
    # Get unique organ IDs
    unique_organs = np.unique(mask)
    unique_organs = unique_organs[unique_organs > 0]
    
    # Create mask for each organ
    organ_masks = {}
    for organ_id in unique_organs:
        organ_masks[organ_id] = (mask == organ_id).astype(np.uint8) * 255

    # Extract contours for each organ
    organ_contours = {}
    for organ_id, organ_mask in organ_masks.items():
        print(f"Processing organ ID: {organ_id}")
        contours, _ = cv2.findContours(organ_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        organ_contours[organ_id] = max(contours, key=cv2.contourArea)
    
    # Build the unified graph with shared points
    G = nx.Graph()
    organ_node_mappings = {}
    point_to_node = {}
    next_node_id = 0

    # Process each organ's contour
    for organ_id, contour in organ_contours.items():
        organ_node_mappings[organ_id] = []
        
        for i in range(len(contour)):
            x, y = contour[i][0]
            
            # Check if this point is close to any existing point
            found_match = False
            for key_point, node_id in point_to_node.items():
                dist = np.sqrt((x - key_point[0])**2 + (y - key_point[1])**2)
                if dist < distance_threshold and dist > 0:
                    organ_node_mappings[organ_id].append(node_id)
                    found_match = True
                    if 'organs' not in G.nodes[node_id]:
                        G.nodes[node_id]['organs'] = []
                    if organ_id not in G.nodes[node_id]['organs']:
                        G.nodes[node_id]['organs'].append(organ_id)
                    break
                    
            if not found_match:
                point_to_node[(x, y)] = next_node_id
                G.add_node(next_node_id, pos=(x, y), organs=[organ_id])
                organ_node_mappings[organ_id].append(next_node_id)
                next_node_id += 1  
    
    # Add edges following contour connectivity
    for organ_id, node_ids in organ_node_mappings.items():
        for i in range(len(node_ids)):
            G.add_edge(node_ids[i], node_ids[(i+1) % len(node_ids)], organ=organ_id)

    return G

def create_unified_adjacency_matrix(G):
    """Create a single unified adjacency matrix for the entire multi-organ graph"""
    
    # Get the total number of nodes
    num_nodes = G.number_of_nodes()
    
    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))
    
    # Create the adjacency matrix
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    
    # Fill the adjacency matrix based on graph edges
    for u, v in G.edges():
        adj_matrix[u, v] = 1
        adj_matrix[v, u] = 1 
    
    # Create the organ membership list
    organ_membership = []
    for i in range(num_nodes):
        if i in G.nodes():
            organs = G.nodes[i]['organs']
            # Join organ IDs with hyphen for shared nodes
            organ_membership.append('-'.join(map(str, sorted(organs))))
        else:
            raise ValueError(f"Node {i} not found in graph.")
    
    return adj_matrix, organ_membership


def create_downsampling_matrix(G, G_down, down_to_orig):
    """
    Create a downsampling matrix mapping from original nodes to downsampled nodes
    """
    # Initialize the downsampling matrix
    D = np.zeros((G_down.number_of_nodes(), G.number_of_nodes()), dtype=np.float32)
    
    # Map nodes to indices
    orig_node_to_idx = {node: i for i, node in enumerate(G.nodes())}
    
    # Fill the matrix
    for down_idx, orig_nodes in down_to_orig.items():
        if len(orig_nodes) == 1:
            # Single node maps directly with weight 1
            orig_idx = orig_node_to_idx[orig_nodes[0]]
            D[down_idx, orig_idx] = 1.0
        else:
            # Pair of nodes each contribute 0.5
            for orig_node in orig_nodes:
                orig_idx = orig_node_to_idx[orig_node]
                D[down_idx, orig_idx] = 0.5
    
    return D

def create_upsampling_matrix(G, G_down, down_to_orig):
    """
    Create an upsampling matrix mapping from downsampled nodes back to original nodes
    """
    # Initialize the upsampling matrix
    U = np.zeros((G.number_of_nodes(), G_down.number_of_nodes()), dtype=np.float32)
    
    # Map nodes to indices
    orig_node_to_idx = {node: i for i, node in enumerate(G.nodes())}
    
    # Fill the matrix
    for down_idx, orig_nodes in down_to_orig.items():
        for orig_node in orig_nodes:
            orig_idx = orig_node_to_idx[orig_node]
            U[orig_idx, down_idx] = 1.0
    
    return U

def identify_downsampling_pairs(G):
    """
    Identify pairs of nodes that can be downsampled.
    Criterion: Only pair nodes that each have exactly 2 neighbors.
    """
    pairs = []
    visited = set()
    
    for node in G.nodes():
        if node in visited:
            continue
            
        # Only consider nodes with exactly 2 neighbors (contour nodes)
        neighbors = list(G.neighbors(node))
        if len(neighbors) != 2:
            continue
            
        # Check each neighbor
        for neighbor in neighbors:
            if neighbor in visited:
                continue
                
            # Skip if neighbor doesn't have exactly 2 neighbors
            neighbor_neighbors = list(G.neighbors(neighbor))
            if len(neighbor_neighbors) != 2:
                continue
                
            # This pair can be downsampled, regardless of organ membership
            pairs.append((node, neighbor))
            visited.add(node)
            visited.add(neighbor)
            break
    
    return pairs

def downsample_one_step(G):
    """
    Downsample a graph to match a target size by selectively merging nodes
    """    
    # Find all possible pairs for downsampling
    all_pairs = identify_downsampling_pairs(G)
    
    # Build the downsampled graph
    G_down, down_to_orig = build_downsampled_graph(G, all_pairs)
    
    # Create matrices
    D = create_downsampling_matrix(G, G_down, down_to_orig)
    U = create_upsampling_matrix(G, G_down, down_to_orig)
    
    return G_down, down_to_orig, D, U

def build_downsampled_graph(G, pairs):
    """
    Build a downsampled graph by merging node pairs, preserving organ membership
    """
    # Create new graph
    G_down = nx.Graph()
    
    # Track mapping from downsampled nodes to original nodes
    down_to_orig = {}
    
    # Get nodes involved in pairs
    paired_nodes = set()
    for n1, n2 in pairs:
        paired_nodes.add(n1)
        paired_nodes.add(n2)
    
    # Add nodes to downsampled graph
    next_idx = 0
    orig_to_down = {}
    
    # First, add nodes representing merged pairs
    for n1, n2 in pairs:
        # Create node with averaged position
        avg_x = (G.nodes[n1]['pos'][0] + G.nodes[n2]['pos'][0]) / 2
        avg_y = (G.nodes[n1]['pos'][1] + G.nodes[n2]['pos'][1]) / 2
        
        # Merge organ memberships (ensuring no duplicates)
        n1_organs = set(G.nodes[n1]['organ_membership'].split('-'))
        n2_organs = set(G.nodes[n2]['organ_membership'].split('-'))
        merged_organs = n1_organs.union(n2_organs)
        merged_membership = '-'.join(sorted(o for o in merged_organs if o))
        
        merged_organs_int = [int(o) for o in merged_organs]

        G_down.add_node(next_idx, 
                        pos=(avg_x, avg_y),
                        organs=merged_organs_int,
                        organ_membership=merged_membership)
        
        # Update mappings
        down_to_orig[next_idx] = (n1, n2)
        orig_to_down[n1] = next_idx
        orig_to_down[n2] = next_idx
        
        next_idx += 1
    
    # Then, add all remaining nodes (not in pairs)
    for node in G.nodes():
        if node not in paired_nodes:
            G_down.add_node(next_idx, 
                           pos=G.nodes[node]['pos'],
                           organs=G.nodes[node]['organs'],
                           organ_membership=G.nodes[node]['organ_membership'])
            
            down_to_orig[next_idx] = (node,)
            orig_to_down[node] = next_idx
            
            next_idx += 1
    
    # Add edges to the downsampled graph
    for n1 in G.nodes():
        for n2 in G.neighbors(n1):
            if n1 < n2:  # Avoid duplicates
                # Map to downsampled nodes
                dn1 = orig_to_down[n1]
                dn2 = orig_to_down[n2]
                
                # Only add edge if downsampled nodes are different
                if dn1 != dn2 and not G_down.has_edge(dn1, dn2):
                    G_down.add_edge(dn1, dn2)
    
    return G_down, down_to_orig


def generate_edge_info(config):
    '''
    Generate edge information for the unified graph
    '''
    for block in config["resolutions"]:
        # Load the unified adjacency matrix
        data_path = unified_data_path(config)
        block_diagonal = np.load(f"{data_path}/adj_{block}_block_diagonal.npy")
        
        # Find all edges in the graph
        edges = np.argwhere(block_diagonal == 1)
        edges = edges[edges[:, 0] < edges[:, 1]]  # Keep upper triangular only
        np.save(f"{data_path}/all_edges_{block}.npy", edges)
        
        # Load per-organ contour
        circular_organ_order = json.load(open(f"{data_path}/organ_order_{block}.json", "r"))
        
        # Create edge matrix
        N_organs = len(circular_organ_order.keys())
        M_edges = [len(order) for order in circular_organ_order.values()]
        max_edges = np.round(np.max(M_edges)).astype(int)  
        edge_matrix = np.zeros([N_organs, max_edges, 2])
        
        # Each organ has its own consecutive node indices in its circular order
        for c, (k, order) in enumerate(circular_organ_order.items()):
            for i in range(len(order)):
                # Get the edge between current and next node in circular order
                edge_matrix[c, i, 0] = order[i]
                edge_matrix[c, i, 1] = order[(i + 1) % len(order)]
        
        np.save(f"{data_path}/edge_matrix_{block}.npy", edge_matrix)
        
def save_matrices(G, name, config, organ_ids):
    """
    Save matrices for the unified graph representation
    """
    data_path = unified_data_path(config)
    # Save the adjacency matrix and organ membership
    adj_matrix = nx.to_numpy_array(G)
    np.save(f"{data_path}/adj_{name}_block_diagonal.npy", adj_matrix)
    
    # Save organ membership as object array
    organ_membership = np.array([G.nodes[node].get('organ_membership', '') for node in G.nodes()])
    organ_id = organ_membership.reshape(-1, 1)
    np.save(f"{data_path}/adj_{name}_organ_id.npy", organ_id)
    np.savetxt(f"{data_path}/adj_{name}_organ_id.txt", sorted(organ_ids), fmt="%s")
    
    organ_id = organ_id.flatten().tolist()
    
    # Get unique organs
    unique_organs = []
    for organ in organ_id:
        for o in organ.split('-'):
            if o and o not in unique_organs:
                unique_organs.append(o)
    unique_organs = sorted(unique_organs)
    
    organ_order = {}

    for organ in unique_organs:
        # Get nodes for this organ
        organ_nodes = [i for i, organs in enumerate(organ_id) if organ in organs.split('-')]
            
        # Get adjacency info for these nodes
        ordered_nodes = []
        
        # Start with first node
        current = organ_nodes[0]
        ordered_nodes = [int(current)]
        
        # Get initial neighbor
        neighbors = np.where(adj_matrix[current] > 0)[0]
        neighbors = [n for n in neighbors if n in organ_nodes]
            
        # Pick any neighbor to start
        prev = current
        current = neighbors[0]
        ordered_nodes.append(int(current))
        
        # Follow the contour using adjacency
        while len(ordered_nodes) < len(organ_nodes):
            # Get current node's neighbors
            neighbors = np.where(adj_matrix[current] > 0)[0]
            neighbors = [n for n in neighbors if n in organ_nodes]
            
            # Find a neighbor that's not the previous node
            next_node = None
            for n in neighbors:
                if n != prev and n not in ordered_nodes:
                    next_node = n
                    break
            
            # If no new neighbor found, try to close the loop
            if next_node is None:
                if len(ordered_nodes) > 2 and ordered_nodes[0] in neighbors:
                    # We can close the loop
                    break
                else:
                    # Can't proceed further
                    break
            
            # Move to next node
            prev = current
            current = next_node
            ordered_nodes.append(int(current))
        
        organ_order[organ] = ordered_nodes        
    with open(f"{data_path}/organ_order_{name}.json", "w") as f:
        json.dump(organ_order, f, indent=4)
        
    return

