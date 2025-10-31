import json
import numpy as np
import torch
from losses.contour import edge_length_loss
from pytorch3d.loss import chamfer_distance

def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    return False

def initialize_edge_matrices_and_organ_counts(dataset, device):
    def load_edge_matrix(file_path):
        edge_matrix = np.load(file_path).astype('int')
        edge_matrix_tensor = torch.tensor(edge_matrix).to(device)
        edge_matrix_tensor.requires_grad = False
        return edge_matrix_tensor

    dataset_config = json.load(open(f"{dataset}/config.json"))
    resolutions = dataset_config['resolutions']
    # append the first resolution to the list of resolutions for deep supervision
    resolutions = [resolutions[0]] + resolutions

    edge_matrices = [load_edge_matrix(f"{dataset}/Naive/edge_matrix_{res}.npy") for res in resolutions]
    organ_ids = [np.load(f"{dataset}/Naive/adj_{res}_organ_id.npy")[:, 0] for res in resolutions]
    organ_order, organ_counts = zip(*[np.unique(ids, return_counts=True) for ids in organ_ids])

    return edge_matrices, organ_ids, organ_order[0], list(organ_counts)

def compute_losses(out_tensors, edge_matrices, organ_order, organ_counts, organ_ids, target, organs_target, counts_target, weights):
    total_losses = {name: 0.0 for name in weights}
    
    for out_tensor, edge_matrix, organ_count, organ_id in zip(out_tensors, edge_matrices, organ_counts, organ_ids):
        average_edge, elasticity, curvature = edge_length_loss(out_tensor, edge_matrix)

        B = out_tensor.shape[0]
        tensor = torch.zeros((B * organ_count.shape[0], organ_count.max(), 2), device=out_tensor.device)
        
        for b in range(B):
            for i in range(organ_count.shape[0]):
                tensor[b * organ_count.shape[0] + i, :organ_count[i], :] = out_tensor[b, organ_id == organ_order[i], :]
        
        # Restructure how we track available organs
        batch_available = []  # List of lists to track available organs per batch
        sizes_pred = []
        sizes_target = []
        available_flat = []
        
        # Make this batch wise
        B = organs_target.shape[0]  # Number of batches
        for b in range(B):
            batch_organs = []  # Track available organs for this batch
            idx = 0
            for organ in organs_target[b].numpy().tolist():
                sizes_target.append(counts_target[b][idx])
                order = np.where(organ_order == organ)[0][0]
                sizes_pred.append(organ_count[order])
                available_flat.append(b * organ_count.shape[0] + order)  # For flat tensor indexing
                batch_organs.append(order)  # Just the organ index for this batch
                idx += 1
            batch_available.append(batch_organs)  # Store available organs for this batch
        
        # Convert to tensors
        sizes_pred = torch.tensor(sizes_pred, device=out_tensor.device, dtype=torch.long, requires_grad=False)
        sizes_target = torch.tensor(sizes_target, device=out_tensor.device, dtype=torch.long, requires_grad=False)
        available_flat = torch.tensor(available_flat, device=out_tensor.device, dtype=torch.long, requires_grad=False)
        
        # Select tensors for all batches
        tensor = tensor[available_flat]
        
        # Concatenate all target batches
        target_tensor = torch.cat([target[b] for b in range(B)], dim=0)

        # Compute chamfer distance for all batches
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