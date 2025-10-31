from data.dataset import ToTensorWithSegBatched as ToTensor
from data.transforms import Scale, AugColor, RandomScaleCentered, Rotate
from torchvision import transforms
from models.layers import Pool 
from losses.dice import OneClassDiceLoss
from torch.nn import BCELoss 
import cv2
import json
import torch
import numpy as np
import os 
import time
from training.utils import initialize_edge_matrices_and_organ_counts, precompute_edge_adjacency, preprocess_edge_matrix
from losses.contour import edge_length_loss

def pretrainer(model, config):
    DATASET = config['dataset']

    transforms_list = [Scale(config['inputsize']), RandomScaleCentered(config['inputsize'])]

    transforms_list.append(Rotate(45))
    transforms_list.extend([AugColor(0.40), ToTensor()])

    transformation = transforms.Compose(transforms_list)

    # Add the first resolution to the list of resolutions for deep supervision
    config['resolutions'] = [config['resolutions'][0]] + config['resolutions']

    torch.manual_seed(420)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    
    tensorboard = "../Training"
    folder = os.path.join(tensorboard, config['name'])
    os.makedirs(folder, exist_ok=True)

    if not config['naive']:
        # check if atlas exists
        if not os.path.exists("%s/NonNaive/atlas_image.png" %DATASET):
            print("Atlas image not found in NonNaive folder. Please run the atlas generation script first.")
            print("Exiting pretrainer, training will be performed without pretraining.")
            return model
        
        og_image = cv2.imread("%s/NonNaive/atlas_image.png" %DATASET,0 ).astype('float') / 255.0
        og_landmarks = np.load("%s/NonNaive/atlas_pos.npy" %DATASET, allow_pickle=True)
    else:
        if not os.path.exists("%s/Naive/atlas_image.png" %DATASET):
            print("Atlas image not found in NonNaive folder. Please run the atlas generation script first.")
            print("Exiting pretrainer, training will be performed without pretraining.")
            return model
        
        og_image = cv2.imread("%s/Naive/atlas_image.png" %DATASET, 0).astype('float') / 255.0
        og_landmarks = np.load("%s/Naive/atlas_pos.npy" %DATASET, allow_pickle=True)
        
    print(f"Image shape: {og_image.shape}")
    print(f"Landmarks shape: {og_landmarks.shape}")
        
    og_landmarks = np.concatenate([og_landmarks, np.ones((og_landmarks.shape[0], 1))], axis=1)

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=0.0001, weight_decay=0.00001
    )

    # Initialize edge matrices and organ data
    t0 = time.time()
    print('Initializing edge matrices and organ counts...')
    edge_matrices, edge_adjacencies, organ_ids, organ_order, organ_counts = initialize_edge_matrices_and_organ_counts(DATASET, device, config)
    t1 = time.time()
    print(f"Initialization took {t1 - t0:.2f} seconds")
    
    print('Pre training with mean squared error...')

    pool = Pool()
    downsamples = model.downsample_matrices
    mse = torch.nn.MSELoss(reduction='mean')
    dice_loss_fun = OneClassDiceLoss()
    bce_loss_fun = BCELoss()

    organs = config['organs']
    
    if not config['naive']:
        organ_id = "%s/NonNaive/adj_full_organ_id.npy" %DATASET
        organ_id = np.load(organ_id)[:,0].tolist()
        
        with open(f"{DATASET}/NonNaive/organ_order_full.json", "r") as f:
            organ_order = json.load(f)
    else:
        organ_id = np.load("%s/Naive/adj_full_organ_id.npy" % DATASET)[:,0]       
        organ_order = {}
        for organ in organs:
            organ_order[organ] = [i for i, id_ in enumerate(organ_id) if id_ == int(organ)]
    
    model.train()   
    t0 = time.time()
    
    for epoch in range(0, 25000):
        sample = {
            'image': og_image.copy(),
            'landmarks': og_landmarks.copy()
        }

        sample_batched = transformation(sample)
        
        image, target = sample_batched['image'].to(device), sample_batched['landmarks'].to(device)
        target = target[:, :model.n_nodes[0]]
        target = target
                
        targets = [target]
        for i in range(0, len(downsamples)):
            targets.append(pool(targets[-1].view(1, -1, 2), downsamples[i]).view(1, -1, 2))
        targets = [target] + targets
        
        rasters_target = sample_batched['raster'].to(device).float()
            
        if not config['use_dual'] and not config['raster_as_input']:
            out_tensor, deep_supervisions = model(image.unsqueeze(0))
        else:
            # Create the rasters for the dual model
            rasters = []

            for i in range(0, len(organs)):
                organ = organs[i]
                organ_ord = organ_order[organ]
                organ_landmarks = target.cpu().numpy() * config['inputsize']
                organ_landmarks = organ_landmarks.astype('int')
                organ_landmarks = organ_landmarks[0, organ_ord, :]
                h, w = image.shape[1], image.shape[2]
                
                raster = cv2.drawContours(np.zeros((h, w)), [organ_landmarks.astype('int')], -1, 1, -1)
                rasters.append(raster)
                
            raster_targets = np.stack(rasters, axis = 0)
            rasters_target = torch.from_numpy(raster_targets).float().to(device)
            rasters_target = rasters_target.unsqueeze(0)
            
            if config['raster_as_input']:
                out_tensor, deep_supervisions = model(rasters_target)
            else:        
                out_tensor, deep_supervisions, seg = model(image.unsqueeze(0))
        
        out_tensors = [out_tensor] + deep_supervisions
                
        optimizer.zero_grad()
        
        total_loss = 0.0
        
        for i in range(len(out_tensors)):
            out_tensor = out_tensors[i]
            target_tensor = targets[i]
            
            # Compute the loss
            loss = mse(out_tensor, target_tensor) * 100
            
            if epoch > 10000:
                average_edge, elasticity, curvature = edge_length_loss(out_tensor, edge_matrices[i], edge_adjacencies[i])
                reg = 50 * elasticity + 50 * curvature + average_edge
            else:
                reg = 0.0
                
            total_loss += loss + reg
            
        kld_loss = -0.5 * torch.mean(
            torch.mean(1 + model.log_var - model.mu ** 2 - model.log_var.exp(), dim=1), dim=0
        )
        
        if epoch % 1000 == 0:
            if not config['use_dual']:
                print(f"Epoch {epoch} - Loss: {total_loss.item():.4f}")
            else:
                dice = dice_loss_fun(seg[0], rasters_target[0])
                bce = bce_loss_fun(seg[0], rasters_target[0])
                total_loss += dice + bce
                print(f"Epoch {epoch} - Loss: {total_loss.item():.4f} - Dice Loss: {dice.item():.4f}",
                    f" BCE Loss: {bce.item():.4f}")
            t1 = time.time()
            print(f"Time elapsed: {t1 - t0:.2f} seconds")
            t0 = time.time()
            
            model.save_checkpoint(f"{os.path.join(folder,config['name'])}_initial.pth", 0, 0, optimizer)
                
        total_loss += 1e-5 * kld_loss
        
        total_loss.backward()
        optimizer.step()
    
    model.save_checkpoint(f"{os.path.join(folder,config['name'])}_initial.pth", 0, 0, optimizer)
    
    return model