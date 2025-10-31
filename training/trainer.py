import os
import time

import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from training.utils import initialize_edge_matrices_and_organ_counts, compute_losses, check_nan
from losses.dice import OneClassDiceLoss
from torch.nn import BCELoss 
import json
import numpy as np

def log_scale_function(t, start=1e-6, end=1e-3):
    return start * (end / start) ** t

def trainer(train_dataset, val_dataset, model, config, start_epoch=0, start_iterations=0, checkpoint=None, validate_every=1):
    torch.manual_seed(420)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    if start_epoch == 0:
        print("Starting training with batch size 1")
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True, drop_last=True
        )
    else:
        print("Resuming training with batch size", config['batch_size'])
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8, pin_memory=True, drop_last=True
        )
        
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config['val_batch_size'], num_workers=4
    )

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']
    )

    if config.get('resume', False) and checkpoint is not None and checkpoint['optimizer_state_dict'] is not None:
        print("Resuming optimizer from checkpoint")
        print("Optimizer state dict keys:", checkpoint['optimizer_state_dict'].keys())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    DATASET = config['DATASET']
    scheduler = StepLR(optimizer, step_size=config['stepsize'], gamma=config['gamma'])

    dice_loss_fun = OneClassDiceLoss()
    bce_loss_fun = BCELoss()
    
    # Initialize edge matrices and organ data
    edge_matrices, edge_adjacencies, organ_ids, organ_order, organ_counts = initialize_edge_matrices_and_organ_counts(DATASET, device, config)

    # For Tensorboard
    tensorboard = "../Training"
    folder = os.path.join(tensorboard, config['name'])

    os.makedirs(folder, exist_ok=True)
    writer = SummaryWriter(log_dir=folder)
    
    # Save the configuration in the tensorboard folder
    hyperparameters = {
        "name": config['name'],
        "latents": config['latents'],
        "initial_filters": config['initial_filters'],
        "raster_as_input": config['raster_as_input'],
        "naive": config['naive'],
        "use_dual": config['use_dual']
    }
    with open(f"{folder}/hyperparameters.json", "w") as f:
        json.dump(hyperparameters, f, indent=4)

    # Save the dataset configuration
    config_copy = config.copy()
    # remove all non serializable keys
    for key in list(config_copy.keys()):
        if not isinstance(config_copy[key], (str, int, float, bool)):
            del config_copy[key]
    with open(f"{folder}/dataset_config.json", "w") as f:
        json.dump(config_copy, f, indent=4)
        
    if not config['naive']:
        with open(f"{DATASET}/NonNaive/organ_order_full.json", "r") as f:
            circ_organ_order = json.load(f)
    else:
        circ_organ_order = None
        
    iterations = start_iterations
    best_val_loss = float('inf')
    
    validation_count = 0
    
    kld_annealing = np.linspace(0, 1, config['iterations'])
    kld_annealing = log_scale_function(kld_annealing, start=1e-6, end=config['kld_w'])
    curvature_annealing = np.linspace(0, 1, config['iterations'])
    curvature_annealing = log_scale_function(curvature_annealing, start=config['curvature_w']/100, end=config['curvature_w'])[::-1]
    edge_annealing = np.linspace(0, 1, config['iterations'] // 5)
    edge_annealing = log_scale_function(edge_annealing, start=config['edge_w']/100, end=config['edge_w'])
    edge_annealing_complete = np.ones(config['iterations']) * edge_annealing[-1]
    edge_annealing = np.concatenate((edge_annealing, edge_annealing_complete))
    elasticity_annealing = np.linspace(0, 1, config['iterations'])
    elasticity_annealing = log_scale_function(elasticity_annealing, start=config['elasticity_w']/100, end=config['elasticity_w'])[::-1]

    # chamfer annealing is linear from chamfer_w to 10 * chamfer_w
    chamfer_annealing = np.linspace(config['chamfer_w'], 10 * config['chamfer_w'], config['iterations'])
    
    for epoch in range(start_epoch, config['epochs']):
        model.train()

        t1 = time.time()
        loss_fns = ['Chamfer', 'Reg/Edge', 'Reg/Elasticity', 'Reg/Curvature']
        total_losses_accum = {name: 0.0 for name in loss_fns}
        prev_it = iterations
        total_losses_accum['Loss'] = 0.0
        total_losses_accum['Dice'] = 0.0
        total_losses_accum['BCE'] = 0.0
        total_losses_accum['Conv/Dice'] = 0.0
        total_losses_accum['Conv/BCE'] = 0.0
        total_losses_accum['Reg/KLD'] = 0.0

        if epoch == 2:
            # Discard train_loader and create a new one with the configured batch size
            # Force it to stop and create a new one
            del train_loader
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4
            )
        
        # make iterations the closest multiple of 4 due to batch size issues
        iterations = (iterations // config['batch_size']) * config['batch_size']
            
        for sample_batched in train_loader:
            image = sample_batched['image'].to(device)
            target = sample_batched['landmarks'].to(device)
            rasters_target = sample_batched['raster'].to(device).float()
            organs_target, counts_target = sample_batched['organs'], sample_batched['counts']
            
            # Determine what to use as input based on config
            model_input = rasters_target if config['raster_as_input'] else image
            
            # Forward pass
            model_output = model(model_input)

            if config['use_dual']:
                graph_output ,positions, segmentation = model_output
            else:
                graph_output, positions = model_output
                segmentation = None

            mu = model.mu
            log_var = model.log_var
                        
            optimizer.zero_grad()
            total_loss = 0.0
            
            # Process graph decoder output
            graph_tensors = [graph_output] + positions
            
            # Graph-specific losses
            weights = {
                'Chamfer': chamfer_annealing[iterations],
                'Reg/Edge': edge_annealing[iterations],
                'Reg/Elasticity': elasticity_annealing[iterations],
                'Reg/Curvature': curvature_annealing[iterations]
            }
            
            graph_loss, total_losses, available_flat, batch_available = compute_losses(
                graph_tensors, edge_matrices, organ_order, organ_counts,
                organ_ids, target, organs_target, counts_target, weights,
                edge_adjacencies
            )        
                
            if check_nan(graph_loss, "graph_loss"):
                return False

            for name in total_losses:
                total_losses_accum[name] = total_losses_accum[name] + total_losses[name]
            
            B = len(batch_available)  # Number of batches
            
            # Add rasterization loss after warmup
            if iterations >= config['warm_up_it'] and config['use_raster']:                
                # Build the rasterizations for all batches
                
                # Rasterize for this batch 
                if config['naive']:
                    rasters_pred = model.raster_naive(graph_tensors[0], organ_ids[0], config['inputsize'])
                else:
                    rasters_pred = model.raster_non_naive(graph_tensors[0], organ_ids[0], circ_organ_order, config['inputsize'])
                                
                all_raster_preds = []
                all_raster_targets = []
                for b in range(B):
                    available_organs = batch_available[b]
                    all_raster_preds.append(rasters_pred[b][available_organs])
                    all_raster_targets.append(rasters_target[b])
                
                # Concatenate all collected tensors
                all_raster_preds = torch.cat(all_raster_preds, dim=0)
                all_raster_targets = torch.cat(all_raster_targets, dim=0)
                
                # Compute losses once on the combined tensors
                graph_dice = dice_loss_fun(all_raster_preds, all_raster_targets) 
                graph_bce = bce_loss_fun(all_raster_preds, all_raster_targets) 
                
                # Add to graph loss
                graph_loss += config['dice_w'] * graph_dice + config['bce_w'] * graph_bce
                
                # Update accumulators
                total_losses_accum['Dice'] = total_losses_accum['Dice'] + graph_dice.item()
                total_losses_accum['BCE'] = total_losses_accum['BCE'] + graph_bce.item()
            
            total_loss += graph_loss
            
            # Convolutional segmentation losses
            if config['use_dual']:
                all_seg_preds = []
                all_seg_targets = []

                for b in range(B):
                    available_organs = batch_available[b]
                    all_seg_preds.append(segmentation[b][available_organs])
                    all_seg_targets.append(rasters_target[b])

                # Concatenate all collected tensors
                all_seg_preds = torch.cat(all_seg_preds, dim=0)
                all_seg_targets = torch.cat(all_seg_targets, dim=0)

                # Compute losses once
                conv_bce = bce_loss_fun(all_seg_preds, all_seg_targets)
                
                conv_dice = dice_loss_fun(all_seg_preds, all_seg_targets)

                conv_loss = config['conv_dice_w'] * conv_dice + config['conv_bce_w'] * conv_bce
                
                total_losses_accum['Conv/Dice'] = total_losses_accum['Conv/Dice'] + conv_dice.item()
                total_losses_accum['Conv/BCE'] = total_losses_accum['Conv/BCE'] + conv_bce.item()
                
                total_loss += conv_loss
                    
            # VAE KL divergence loss
            kld_loss = -0.5 * torch.mean(
                torch.mean(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
            )

            total_loss += kld_loss * kld_annealing[iterations]
            total_losses_accum['Reg/KLD'] = total_losses_accum['Reg/KLD'] + kld_loss.item()
            
            total_losses_accum['Loss'] = total_losses_accum['Loss'] + total_loss.item()
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
        
            iterations += image.shape[0]
            
            # Log progress
            if iterations % 240 == 0:
                t2 = time.time()
                total_losses_accum = {name: total_losses_accum[name] / (iterations - prev_it) * image.shape[0]   for name in total_losses_accum}
                prev_it = iterations
                print(f"Epoch {epoch}, Iteration {iterations}, Loss {total_losses_accum['Loss']}")
                for name, loss_value in total_losses_accum.items():
                    if loss_value > 0:  # Only print losses that are used
                        print(f"{name}: {round(loss_value, 5)}", end=", ")
                        if not "Reg" in name:
                            writer.add_scalar(f'{name}/train', loss_value, iterations)
                        else:
                            writer.add_scalar(f'{name}', loss_value, iterations)
                # print the annealing values
                print(f"Edge: {round(edge_annealing[iterations], 5)}, Elasticity: {round(elasticity_annealing[iterations], 5)}, Curvature: {round(curvature_annealing[iterations], 5)}, KLD: {round(kld_annealing[iterations], 5)}")
                print(f"Time: {round(t2 - t1, 5)} \n")
                
                t1 = time.time()
                total_losses_accum = {name: 0.0 for name in total_losses_accum}
            
            # Save checkpoint periodically
            if iterations % 480 == 0:
                model.save_checkpoint(f"{os.path.join(folder,config['name'])}.pth", epoch, iterations, optimizer)

        if iterations != prev_it:
            t2 = time.time()
            total_losses_accum = {name: total_losses_accum[name] / (iterations - prev_it) * image.shape[0]   for name in total_losses_accum}
            prev_it = iterations
            print(f"Epoch {epoch}, Iteration {iterations}, Loss {total_losses_accum['Loss']}")
            for name, loss_value in total_losses_accum.items():
                if loss_value > 0:  # Only print losses that are used
                    print(f"{name}: {round(loss_value, 5)}", end=", ")
                    if not "Reg" in name:
                        writer.add_scalar(f'{name}/train', loss_value, iterations)
                    else:
                        writer.add_scalar(f'{name}', loss_value, iterations)
            print(f"Time: {round(t2 - t1, 5)} \n")
            
            t1 = time.time()
            total_losses_accum = {name: 0.0 for name in total_losses_accum}
        
        validation_count += 1
        
        if validation_count % validate_every == 0:
            # Validation at end of epoch
            val_results = validate(val_loader, model, config, device, edge_matrices, organ_ids, organ_order, organ_counts, edge_adjacencies, dice_loss_fun, bce_loss_fun, circ_organ_order)
            val_loss = val_results['total_loss']
            t2 = time.time()
            
            # Log validation metrics
            writer.add_scalar('Loss/val', val_loss, iterations)
            writer.add_scalar('Chamfer/val', val_results['chamfer'], iterations)
            writer.add_scalar('Dice/val', val_results['graph_dice'], iterations)
            writer.add_scalar('BCE/val', val_results['graph_bce'], iterations)

            if config['use_dual']:
                writer.add_scalar('Conv/Dice/val', val_results['conv_dice'], iterations)
                writer.add_scalar('Conv/BCE/val', val_results['conv_bce'], iterations)
            
            val_loss = val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_checkpoint(f"{os.path.join(folder,config['name'])}_best.pth", epoch, iterations, optimizer)
                
            print(f"Validation Loss: {val_loss}")
            print(f"Best Validation Loss: {best_val_loss}")
            print(f"Time: {t2 - t1} \n")
            t1 = time.time()

        scheduler.step()
        
    return True

def validate(val_loader, model, config, device, edge_matrices, organ_ids, organ_order, organ_counts, edge_adjacencies, dice_loss_fun, bce_loss_fun, circ_organ_order=None):
    model.eval()
    # Initialize accumulators for various metrics
    total_loss_accum = 0
    chamfer_accum = 0
    graph_dice_accum = 0
    graph_bce_accum = 0
    conv_dice_accum = 0
    conv_bce_accum = 0
    num_batches = 0
    
    with torch.no_grad():
        for sample_batched in val_loader:
            image = sample_batched['image'].to(device)
            target = sample_batched['landmarks'].to(device)
            rasters_target = sample_batched['raster'].to(device).float()
            organs_target, counts_target = sample_batched['organs'], sample_batched['counts']
            
            # Determine what to use as input
            model_input = rasters_target if config['raster_as_input'] else image
            
            # Forward pass
            model_output = model(model_input)

            if config['use_dual']:
                graph_output ,positions, segmentation = model_output
            else:
                graph_output, positions = model_output
                segmentation = None

            total_loss = 0.0
            
            # Process graph decoder output
            graph_tensors = [graph_output] + positions
            
            # Number of batches in this batch
            B = len(target)
            
            # Graph-specific losses without regularization
            weights = {
                'Chamfer': config['chamfer_w'],
                'Reg/Edge': 0,
                'Reg/Elasticity': 0,
                'Reg/Curvature': 0
            }
            
            # Call the updated compute_losses function
            graph_loss, total_losses, available_flat, batch_available = compute_losses(
                graph_tensors, edge_matrices, organ_order, organ_counts,
                organ_ids, target, organs_target, counts_target, weights, edge_adjacencies
            )
            
            # Rasterize for this batch 
            if config['naive']:
                rasters_pred = model.raster_naive(graph_tensors[0], organ_ids[0], config['inputsize'])
            else:
                rasters_pred = model.raster_non_naive(graph_tensors[0], organ_ids[0], circ_organ_order, config['inputsize'])
                
            # Initialize empty lists to collect predictions and targets
            all_raster_preds = []
            all_raster_targets = []

            if config['use_dual']:
                all_seg_preds = []
                all_seg_targets = []
            
            for b in range(B):
                available_organs = batch_available[b]
                
                # Collect only available organs' predictions and targets
                all_raster_preds.append(rasters_pred[b][available_organs])
                all_raster_targets.append(rasters_target[b])
                
                # Collect segmentation predictions and targets
                if config['use_dual']:
                    all_seg_preds.append(segmentation[b][available_organs])
                    all_seg_targets.append(rasters_target[b])
            
            # Concatenate all collected tensors
            all_raster_preds = torch.cat(all_raster_preds, dim=0)
            all_raster_targets = torch.cat(all_raster_targets, dim=0)

            if config['use_dual']:
                all_seg_preds = torch.cat(all_seg_preds, dim=0)
                all_seg_targets = torch.cat(all_seg_targets, dim=0)
            
            # Compute losses once on the combined tensors
            graph_dice = dice_loss_fun(all_raster_preds, all_raster_targets) 
            graph_bce = bce_loss_fun(all_raster_preds, all_raster_targets)

            if config['use_dual']:
                conv_dice = dice_loss_fun(all_seg_preds, all_seg_targets) 
                conv_bce = bce_loss_fun(all_seg_preds, all_seg_targets)
            
            # Add dice and BCE to graph loss (using tensors directly)
            graph_loss += config['dice_w'] * graph_dice + config['bce_w'] * graph_bce
            total_loss += graph_loss
            
            # Accumulate metrics
            graph_dice_accum += graph_dice.item()
            graph_bce_accum += graph_bce.item()
            chamfer_accum += total_losses.get('Chamfer', 0)
            total_loss_accum += total_loss.item()

            if config['use_dual']:
                conv_dice_accum += conv_dice.item()
                conv_bce_accum += conv_bce.item()
            
            num_batches += 1
            
    # Normalize accumulated metrics by number of batches
    metrics = {
        'total_loss': total_loss_accum / num_batches,
        'chamfer': chamfer_accum / num_batches,
        'graph_dice': graph_dice_accum / num_batches,
        'graph_bce': graph_bce_accum / num_batches
    }

    if config['use_dual']:
        metrics['conv_dice'] = conv_dice_accum / num_batches
        metrics['conv_bce'] = conv_bce_accum / num_batches
    
    return metrics