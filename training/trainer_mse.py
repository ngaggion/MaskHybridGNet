import os
import time

import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from training.utils import initialize_edge_matrices_and_organ_counts, compute_losses, check_nan
from losses.dice import OneClassDiceLoss
from torch.nn import BCELoss 
import json

from models.layers import Pool

### Trainer
def trainer(train_dataset, val_dataset, model, config, start_epoch, start_iterations, checkpoint=None): 
    torch.manual_seed(420)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config['val_batch_size'], num_workers=1
    )

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']
    )

    if config['resume']:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    DATASET = config['DATASET']
    scheduler = StepLR(optimizer, step_size=config['stepsize'], gamma=config['gamma'])

    dice_loss_fun = OneClassDiceLoss()
    bce_loss_fun = BCELoss()
    
    edge_matrices, organ_ids, organ_order, organ_counts = initialize_edge_matrices_and_organ_counts(DATASET, device)

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
        "raster_as_input": config['raster_as_input']
    }
    with open(f"{folder}/hyperparameters.json", "w") as f:
        json.dump(hyperparameters, f, indent=4)

    print('Training ...')

    iterations = start_iterations
    best_val_loss = float('inf')

    pool = Pool()
    downsamples = model.downsample_matrices
    
    for epoch in range(start_epoch, config['epochs']):
        model.train()

        t1 = time.time()
        loss_fns = ['MSE']
        total_losses_accum = {name: 0.0 for name in loss_fns}
        prev_it = iterations
        total_losses_accum['Loss'] = 0.0
        total_losses_accum['Dice'] = 0.0
        total_losses_accum['BCE'] = 0.0
        total_losses_accum['Reg/KLD'] = 0.0
            
        for sample_batched in train_loader:
            image, target = sample_batched['image'].to(device), sample_batched['landmarks'].to(device)
            
            n_o = target.shape[1]
            
            targets = [target]
            for i in range(0, len(downsamples)):
                targets.append(pool(targets[-1].view(1, -1, 2), downsamples[i]).view(1, n_o, -1, 2))
            targets = [target] + targets
            
            rasters_target = sample_batched['raster'].to(device).float()
            organs_target, counts_target = sample_batched['organs'], sample_batched['counts']
            
            if config['raster_as_input']:
                out_tensor, deep_supervisions = model(rasters_target)            
            else:
                out_tensor, deep_supervisions = model(image)
            
            out_tensors = [out_tensor] + deep_supervisions
                
            optimizer.zero_grad()

            weights = {'MSE': config['mse_w']}
            total_loss, total_losses, available = compute_losses(out_tensors, organ_order, organ_counts, organ_ids, targets, organs_target, weights)
            
            if check_nan(total_loss, "total_loss"):
                return

            for name in total_losses_accum:
                if name in total_losses:
                    total_losses_accum[name] = total_losses_accum[name] + total_losses[name]

            if iterations >= config['warm_up_it']:
                rasters_pred = model.raster(out_tensors[0][0, :, :], organ_ids[0], config['inputsize'])
                dice = dice_loss_fun(rasters_pred[0, available], rasters_target[0])
                bce = bce_loss_fun(rasters_pred[0, available], rasters_target[0])
                total_loss += config['dice_w'] * dice + config['bce_w'] * bce
                total_losses_accum['Dice'] = total_losses_accum['Dice'] + dice.item()
                total_losses_accum['BCE'] = total_losses_accum['BCE'] + bce.item()
            
            kld_loss = -0.5 * torch.mean(
                torch.mean(1 + model.log_var - model.mu ** 2 - model.log_var.exp(), dim=1), dim=0
            )

            total_loss += config['kld_w'] * kld_loss
            total_losses_accum['Reg/KLD'] = total_losses_accum['Reg/KLD'] + kld_loss.item()
            total_losses_accum['Loss'] = total_losses_accum['Loss'] + total_loss.item()
            
            total_loss.backward()
            optimizer.step()
            iterations += 1

            if iterations % 250 == 0:
                t2 = time.time()
                total_losses_accum = {name: total_losses_accum[name] / (iterations - prev_it) for name in total_losses_accum}
                prev_it = iterations
                print(f"Epoch {epoch}, Iteration {iterations}, Loss {total_losses_accum['Loss']}")
                for name, loss_value in total_losses_accum.items():
                    print(f"{name}: {round(loss_value, 5)}", end=", ")
                    if not "Reg" in name:
                        writer.add_scalar(f'{name}/train', loss_value, iterations)
                    else:
                        writer.add_scalar(f'{name}', loss_value, iterations)
                print(f"Time: {round(t2 - t1, 5)} \n")
                
                t1 = time.time()
                total_losses_accum = {name: 0.0 for name in total_losses_accum}
            
            if iterations % 500 == 0:
                model.save_checkpoint(f"{os.path.join(folder,config['name'])}.pth", epoch, iterations, optimizer)
            
            scheduler.step()
        
        val_loss, mse, dice, bce = validate(val_loader, model, config, device, edge_matrices, organ_ids, organ_order, organ_counts, dice_loss_fun, bce_loss_fun)
        t2 = time.time()
        writer.add_scalar('Loss/val', val_loss / len(val_dataset), iterations)
        writer.add_scalar('MSE/val', mse / len(val_dataset), iterations)
        writer.add_scalar('Dice/val', dice / len(val_dataset), iterations)
        writer.add_scalar('BCE/val', bce / len(val_dataset), iterations)
        
        val_loss = val_loss / len(val_dataset)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_checkpoint(f"{os.path.join(folder,config['name'])}_best.pth", epoch, iterations, optimizer)
            
        print(f"Validation Loss: {val_loss}")
        print(f"Best Validation Loss: {best_val_loss}")
        print(f"Time: {t2 - t1} \n")
        t1 = time.time()
        

def validate(val_loader, model, config, device, edge_matrices, organ_ids, organ_order, organ_counts, dice_loss_fun, bce_loss_fun):
    model.eval()
    total_loss_accum = 0
    mse_accum = 0
    dice_accum = 0
    bce_accum = 0
    
    with torch.no_grad():
        for sample_batched in val_loader:
            image, target = sample_batched['image'].to(device), sample_batched['landmarks'].to(device)
            rasters_target = sample_batched['raster'].to(device).float()
            organs_target, counts_target = sample_batched['organs'], sample_batched['counts']
            
            if config['raster_as_input']:
                out_tensors, _ = model(rasters_target)            
            else:
                out_tensors, _ = model(image)

            out_tensors = [out_tensors]
            edge_matrices = [edge_matrices[0]]
            organ_ids = [organ_ids[0]]
            organ_counts = [organ_counts[0]]
            target = [target]

            weights = {'MSE': config['mse_w']}
            total_loss, total_losses, available = compute_losses(out_tensors, organ_order, organ_counts, organ_ids, target, organs_target, weights)

            rasters_pred = model.raster(out_tensors[0][0, :, :], organ_ids[0], config['inputsize'])
            dice = dice_loss_fun(rasters_pred[0, available], rasters_target[0])
            bce = bce_loss_fun(rasters_pred[0, available], rasters_target[0])
            total_loss += config['dice_w'] * dice + config['bce_w'] * bce

            total_loss_accum += total_loss.item()
            mse_accum += total_losses['MSE']
            dice_accum += dice.item()
            bce_accum += bce.item()

    return total_loss_accum, mse_accum, dice_accum, bce_accum


import numpy as np

def compute_losses(out_tensors, organ_order, organ_counts, organ_ids, targets, organs_target, weights):
    total_losses = {name: 0.0 for name in weights}
    
    for out_tensor, organ_count, organ_id, target in zip(out_tensors, organ_counts, organ_ids, targets):
        B = out_tensor.shape[0]
        tensor = torch.zeros((B * organ_count.shape[0], organ_count.max(), 2), device=out_tensor.device)
        
        for b in range(B):
            for i in range(organ_count.shape[0]):
                tensor[b * organ_count.shape[0] + i, :organ_count[i], :] = out_tensor[b, organ_id == organ_order[i], :]
                    
        available = []

        idx = 0 
        b = 0
        for organ in organs_target[0].numpy().tolist():
            order = np.where(organ_order == organ)[0][0]
            available.append(order)
            idx += 1

        available = torch.tensor(available, device=out_tensor.device, dtype=torch.long, requires_grad=False)

        tensor = tensor[available]
        mse = torch.mean((tensor - target[0]) ** 2)
        total_losses['MSE'] += mse
    
    total_loss = 0
    for name in total_losses:
        total_loss += weights[name] * total_losses[name]
        total_losses[name] = total_losses[name].item()
    
    return total_loss, total_losses, available