import numpy as np
import torch
from losses.contour import edge_length_loss
from pytorch3d.loss import chamfer_distance
import json

def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    return False

def initialize_edge_matrices_and_organ_counts(dataset, device, config):
    def load_edge_matrix(file_path):
        edge_matrix = np.load(file_path).astype('int')
        edge_matrix_tensor = torch.tensor(edge_matrix).to(device)
        edge_matrix_tensor.requires_grad = False
        return edge_matrix_tensor
    
    dataset_config = json.load(open(f"{dataset}/config.json"))
    resolutions = dataset_config['resolutions']
    # append the first resolution to the list of resolutions for deep supervision
    resolutions = [resolutions[0]] + resolutions
    
    if config['naive']:
        adj_path = "Naive"
    else:
        adj_path = "NonNaive"
    
    edge_matrices = [load_edge_matrix(f"{dataset}/{adj_path}/edge_matrix_{res}.npy") for res in resolutions]
    
    # Try to load precomputed edge adjacencies if available
    try:
        edge_adjacencies = [np.load(f"{dataset}/{adj_path}/edge_adjacency_{res}.npy", allow_pickle=True) for res in resolutions]
        edge_adjacencies = [torch.tensor(edge_adj, dtype=torch.long).to(device) for edge_adj in edge_adjacencies]
    except FileNotFoundError:
        print("Precomputed edge adjacencies not found, computing them now...")
        edge_adjacencies = [precompute_edge_adjacency(edge_matrix) for edge_matrix in edge_matrices]
        # Save the computed edge adjacencies for future use
        for res, edge_adj in zip(resolutions, edge_adjacencies):
            np.save(f"{dataset}/{adj_path}/edge_adjacency_{res}.npy", edge_adj, allow_pickle=True)
    
    edge_matrices = [preprocess_edge_matrix(edge_matrix) for edge_matrix in edge_matrices]
    
    # Load organ IDs 
    organ_id_arrays = [np.load(f"{dataset}/{adj_path}/adj_{res}_organ_id.npy", allow_pickle=True)[:, 0] for res in resolutions]
    
    # Check if we're dealing with the naive approach (integer IDs) or strings with hyphens
    is_naive = config.get('naive', False)
    if is_naive or not isinstance(organ_id_arrays[0][0], str) or '-' not in ''.join(organ_id_arrays[0].astype(str)):
        print(f"Using {'naive' if is_naive else 'non-naive'} adjacency matrix")
        # Original naive approach with integer organ IDs
        organ_order, organ_counts = zip(*[np.unique(ids, return_counts=True) for ids in organ_id_arrays])
        return edge_matrices, edge_adjacencies, organ_id_arrays, organ_order[0], list(organ_counts)
    else:
        print(f"Using {'naive' if is_naive else 'non-naive'} adjacency matrix")
        # New approach with string organ IDs (shared boundaries)
        # Get unique organ IDs across all nodes
        all_unique_organs = set()
        for organ_ids in organ_id_arrays:
            for org_str in organ_ids:
                for org in str(org_str).split('-'):
                    if org:  # Skip empty strings
                        all_unique_organs.add(org)
        
        # Sort the unique organs to ensure consistent ordering
        organ_order = sorted(list(all_unique_organs))
        
        # Count nodes per organ, including shared nodes in counts for all organs they belong to
        organ_counts = []
        for organ_ids in organ_id_arrays:
            # Initialize counts dictionary
            counts_dict = {org: 0 for org in organ_order}
            
            # Count nodes for each organ
            for org_str in organ_ids:
                for org in str(org_str).split('-'):
                    if org in counts_dict:
                        counts_dict[org] += 1
            
            # Convert to ordered array matching organ_order
            counts = np.array([counts_dict[org] for org in organ_order])
            organ_counts.append(counts)
        
        return edge_matrices, edge_adjacencies, organ_id_arrays, organ_order, organ_counts


def preprocess_edge_matrix(em):
    """Remove padded edges from edge matrix, return valid edges and mapping."""
    N_o, max_edges, _ = em.shape
    
    valid_edges_list = []
    organ_mapping = []
    edge_to_valid_idx = {}  # Maps (organ, edge) to index in valid list
    
    idx = 0
    for o in range(N_o):
        for e in range(max_edges):
            if em[o, e, 0] != em[o, e, 1]:  # Valid edge
                valid_edges_list.append(em[o, e])
                organ_mapping.append(o)
                edge_to_valid_idx[(o, e)] = idx
                idx += 1
    
    return (torch.stack(valid_edges_list) if valid_edges_list else torch.empty(0, 2, dtype=torch.long), 
            torch.tensor(organ_mapping), 
            edge_to_valid_idx)


def precompute_edge_adjacency(em):
    """Works for any edge matrix by finding edges that share nodes."""
    N_o, max_edges, _ = em.shape
    edge_pairs_list = []
    
    for o in range(N_o):
        # Find valid edges
        valid = (em[o, :, 0] != em[o, :, 1])
        valid_indices = torch.where(valid)[0]
        
        # Find which edges share a node
        for i in range(len(valid_indices)):
            for j in range(i, len(valid_indices)):
                if i != j:
                    edge_i = em[o, valid_indices[i]]
                    edge_j = em[o, valid_indices[j]]
                    
                    # Check if edges share a node
                    if (edge_i[0] in edge_j) or (edge_i[1] in edge_j):
                        edge_pairs_list.append([o, valid_indices[i], valid_indices[j]])                    
    
    if edge_pairs_list:
        edge_pairs = torch.tensor(edge_pairs_list)
        return edge_pairs[:, 0], edge_pairs[:, 1], edge_pairs[:, 2]
    else:
        return None, None, None

def compute_losses(out_tensors, edge_matrices, organ_order, organ_counts, organ_ids, target, organs_target, counts_target, weights, edge_adjacencies):    
    total_losses = {name: 0.0 for name in weights}
    
    # Check if we're dealing with the naive approach or shared boundaries
    is_naive = not isinstance(organ_ids[0][0], str) or '-' not in ''.join(organ_ids[0].astype(str))
    
    for out_tensor, edge_matrix, organ_count, organ_id, edge_adj in zip(out_tensors, edge_matrices, organ_counts, organ_ids, edge_adjacencies):
        average_edge, elasticity, curvature = edge_length_loss(out_tensor, edge_matrix, edge_adj)
        
        B = out_tensor.shape[0]
        
        if is_naive:
            # Original implementation for naive approach - UNCHANGED
            tensor = torch.zeros((B * organ_count.shape[0], organ_count.max(), 2), device=out_tensor.device)
            for b in range(B):
                for i in range(organ_count.shape[0]):
                    tensor[b * organ_count.shape[0] + i, :organ_count[i], :] = out_tensor[b, organ_id == organ_order[i], :]
        else:
            # New implementation for shared boundaries
            tensor = torch.zeros((B * organ_count.shape[0], organ_count.max(), 2), device=out_tensor.device)
            
            # Restructure the output tensor by organ
            for b in range(B):
                for i in range(organ_count.shape[0]):
                    organ_mask = np.zeros(organ_id.shape[0], dtype= bool)
                    org = organ_order[i]
                    for j in range(len(organ_id)):
                        node_organs = str(organ_id[j])
                        organ_mask[j] = org in node_organs.split('-')                    
                    
                    subset = out_tensor[b, organ_mask, :]                    
                    tensor[b * organ_count.shape[0] + i, :organ_count[i], :] = subset[:organ_count[i], :]
        
        # Rest of the function remains the same
        # Track available organs per batch and their sizes
        batch_available = []
        sizes_pred = []
        sizes_target = []
        available_flat = []
        
        B = organs_target.shape[0]
        for b in range(B):
            batch_organs = []
            idx = 0
            for organ in organs_target[b].numpy().tolist():
                sizes_target.append(counts_target[b][idx])
                
                if is_naive:
                    try:
                        order = np.where(organ_order == organ)[0][0]
                        sizes_pred.append(organ_count[order])
                        available_flat.append(b * organ_count.shape[0] + order)
                        batch_organs.append(order)
                    except:
                        print(f"Warning: Organ {organ} not found in organ_order {organ_order}")
                        idx += 1
                        continue
                else:
                    try:
                        org_str = str(organ)
                        order = organ_order.index(org_str)
                        sizes_pred.append(organ_count[order])
                        available_flat.append(b * len(organ_order) + order)
                        batch_organs.append(order)
                    except ValueError:
                        print(f"Warning: Organ {organ} not found in organ_order {organ_order}")
                        idx += 1
                        continue
                
                idx += 1
            
            batch_available.append(batch_organs)
        
        # Convert to tensors
        sizes_pred = torch.tensor(sizes_pred, device=out_tensor.device, dtype=torch.long, requires_grad=False)
        sizes_target = torch.tensor(sizes_target, device=out_tensor.device, dtype=torch.long, requires_grad=False)
        available_flat = torch.tensor(available_flat, device=out_tensor.device, dtype=torch.long, requires_grad=False)
        
        # Select tensors for available organs
        tensor = tensor[available_flat]
        
        # Concatenate all target batches
        target_tensor = torch.cat([target[b] for b in range(B)], dim=0)
        
        # Compute chamfer distance
        chamfer = chamfer_distance(tensor, target_tensor, x_lengths=sizes_pred, y_lengths=sizes_target, batch_reduction='sum')[0] / B
        
        total_losses['Chamfer'] += chamfer
        total_losses['Reg/Edge'] += average_edge
        total_losses['Reg/Elasticity'] += elasticity
        total_losses['Reg/Curvature'] += curvature
    
    total_loss = 0
    for name in total_losses:
        total_loss += weights[name] * total_losses[name]
        total_losses[name] = total_losses[name].item()
    
    return total_loss, total_losses, available_flat, batch_available
