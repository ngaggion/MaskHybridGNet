import torch

def edge_length_loss(nodes, edge_matrices, edge_adjacency=None):
    #unpack edge_matrices and send to device
    valid_edges, organ_ids, edge_mapping = edge_matrices
    organ_ids = organ_ids.to(nodes.device)
    valid_edges = valid_edges.to(nodes.device)

    B = nodes.shape[0]
    N_o = torch.max(organ_ids) + 1  # Number of organs

    # Compute first order differences - all edges are valid
    first_order = nodes[:, valid_edges[:, 0], :] - nodes[:, valid_edges[:, 1], :]  # (B, n_valid_edges, 2)
    lengths = torch.norm(first_order, dim=2)  # (B, n_valid_edges)
    first_order_squared = torch.sum(first_order ** 2, dim=2)  # (B, n_valid_edges)
    
    # Count edges per organ
    edges_per_organ = torch.bincount(organ_ids, minlength=N_o).float() + 1e-8
    
    # Accumulate lengths and squared values by organ using scatter
    contour_lengths = torch.zeros(B, N_o, device=nodes.device)
    elasticity_sum = torch.zeros(B, N_o, device=nodes.device)
    
    for b in range(B):
        contour_lengths[b].index_add_(0, organ_ids, lengths[b])
        elasticity_sum[b].index_add_(0, organ_ids, first_order_squared[b])
    
    # Compute weights based on contour size
    weights = contour_lengths / (torch.max(contour_lengths) + 1e-8)
    
    # Mean lengths per organ
    mean_lengths = contour_lengths / edges_per_organ  # (B, N_o)
    
    # Same length loss - expand means to match edges
    mean_expanded = mean_lengths[:, organ_ids]  # (B, n_valid_edges)
    normalized_diffs = (lengths - mean_expanded) / (mean_expanded + 1e-8)
    
    # Accumulate squared differences by organ
    same_length_sum = torch.zeros(B, N_o, device=nodes.device)
    for b in range(B):
        same_length_sum[b].index_add_(0, organ_ids, normalized_diffs[b] ** 2)
    
    # Average by number of edges per organ
    same_length_loss = same_length_sum / edges_per_organ  # (B, N_o)
    elasticity_loss = elasticity_sum / edges_per_organ  # (B, N_o)
    
    # Curvature loss
    if edge_adjacency is not None:
        org_ids, edge1_ids, edge2_ids = edge_adjacency
        org_ids = org_ids.to(nodes.device)
        edge1_ids = edge1_ids.to(nodes.device)
        edge2_ids = edge2_ids.to(nodes.device)
        
        # Get edge vectors for consecutive pairs
        edges1 = first_order[:, edge1_ids, :]  # (B, n_pairs, 2)
        edges2 = first_order[:, edge2_ids, :]  # (B, n_pairs, 2)
        second_order = edges2 - edges1
        curvature_per_pair = torch.sum(second_order ** 2, dim=2)  # (B, n_pairs)
        
        # Accumulate curvature by organ
        curvature_sum = torch.zeros(B, N_o, device=nodes.device)
        pairs_per_organ = torch.zeros(N_o, device=nodes.device)
        
        for b in range(B):
            curvature_sum[b].index_add_(0, org_ids, curvature_per_pair[b])
        
        pairs_per_organ.index_add_(0, org_ids, torch.ones_like(org_ids, dtype=torch.float))
        curvature_loss = curvature_sum / (pairs_per_organ + 1e-8)  # (B, N_o)
    else:
        curvature_loss = torch.zeros(B, N_o, device=nodes.device)
    
    # Apply weights and average across organs and batch
    same_length_loss = torch.mean(same_length_loss * weights)
    elasticity_loss = torch.mean(elasticity_loss * weights)
    curvature_loss = torch.mean(curvature_loss * weights)
    
    return same_length_loss, elasticity_loss, curvature_loss
