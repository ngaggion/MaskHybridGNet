import numpy as np
import torch
from pathlib import Path
import os
from medpy.metric.binary import dc, hd, __surface_distances
import cv2
import pandas as pd

from models.hybridgnet_se_resnext import Hybrid
from models.hybridgnet_se_resnext_dual import HybridDual
from models.utils import load_config, normalize_representation, is_independent, adjacency_data_dir
from data.dataset import LandmarksDataset, ToTensor, TestDataset, ToTensorTest, ToTensorWithSeg
from data.transforms import Scale, ScaleImageOnly
from torchvision import transforms
from utils.image_processing import process_image, create_mask, create_mask_independent

def calculate_metrics(pred, gt):
    dc_val = dc(pred, gt)
    try:
        hd_val = hd(pred, gt, voxelspacing=(1, 1))
        d1 = __surface_distances(pred, gt, voxelspacing=(1, 1))
        d2 = __surface_distances(gt, pred, voxelspacing=(1, 1))
        assd_val = np.mean(np.concatenate((d1, d2)))
    except:
        hd_val = np.nan
        assd_val = np.nan

    return dc_val, hd_val, assd_val

def eval_test_set(dataset_path, model_path, parameters, results_path, save_masks = False):
    
    representation = normalize_representation(parameters)
    if representation == "independent":
        organ_order = None
    else:
        DATASET = dataset_path
        adj_path = adjacency_data_dir(DATASET, representation)
        with open(f"{DATASET}/{adj_path}/organ_order_full.json", "r") as f:
            organ_order = json.load(f)
    
    config, D_t, U_t, A_t = load_config(dataset_path, parameters)
    
    if config["use_dual"]:
        model = HybridDual(config, D_t, U_t, A_t).to(config['device'])
    else:    
        model = Hybrid(config, D_t, U_t, A_t).to(config['device'])

    if model_path:
        model.load_checkpoint(model_path, config['device'])
    
    model.eval()

    organ_id_path = adjacency_data_dir(config['DATASET'], config['representation'])
    organ_id = np.load(Path(config['DATASET']) / organ_id_path / "adj_full_organ_id.npy")[:,0]
        
    DATASET = config['DATASET']
    images = np.loadtxt("%s/test.txt"%DATASET, dtype = str)
    test_dataset = LandmarksDataset(images, "%s/images"%DATASET, "%s/landmarks"%DATASET, 
                               transform = transforms.Compose([Scale(config['inputsize']), ToTensorWithSeg()]))
    
    output_table = []

    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        name = test_dataset.images[i]
        image = sample['image'].unsqueeze(0).to(config['device'])
        raster = sample['raster'].unsqueeze(0).to(config['device'])
        
        image_path = os.path.join("%s/images"%DATASET, name)
        og_image = cv2.imread(image_path, 0)
        
        h, w = og_image.shape
        _, size, pad_w, pad_h = process_image(og_image, h, w)

        with torch.no_grad():
            if config['raster_as_input']:
                out = model(raster)[0]
            else:
                out = model(image)[0]
            out = out.cpu().numpy()[0]
            out = out * size
            out = np.round(out, 0).astype(int)
            out[:,0] -= pad_w
            out[:,1] -= pad_h

        mask = np.zeros((len(organ_id), h, w), dtype=np.uint8)
        for organ in np.unique(config['organs']).astype(int):
            if organ_order is not None:
                index_train = organ_order[str(organ)]
            else:
                index_train = organ_id == int(organ)
            contour = out[index_train]
            cv2.drawContours(mask[int(organ)], [contour], -1, 1, thickness=cv2.FILLED)

        og_landmarks = os.path.join("%s/landmarks"%DATASET, name.replace(".png", ".json"))
        if os.path.exists(og_landmarks):    
            with open(og_landmarks, 'r') as f:
                GT_landmarks = json.load(f)

        GT = np.zeros((len(organ_id), h, w), dtype=np.uint8)
        for organ in GT_landmarks.keys():
            organ = int(organ)
            contour = np.array(GT_landmarks[str(organ)])
            cv2.drawContours(GT[organ], [contour], -1, 1, thickness=cv2.FILLED)
        
        results = []
        organs = config['organs']
        organ_names = config['organ_names']

        available_organs = GT_landmarks.keys() 
        available_organs = [int(organ) for organ in available_organs]

        for organ_num, organ_name in zip(organs, organ_names):
            if int(organ_num) not in available_organs:
                continue
            dc_val, hd_val, assd_val = calculate_metrics(mask[int(organ_num)], GT[int(organ_num)])
            results.append({
                "image": name,
                "organ": organ_name,
                "dc": dc_val,
                "hd": hd_val,
                "assd": assd_val,
            })

        output_table.extend(results)

        if save_masks:
            out_path = os.path.join(results_path, "segmentations", name)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, mask)
        
        print("Processed image %d/%d"%(i+1, len(test_dataset)), end = "\r")

    output_table = pd.DataFrame(output_table)
    os.makedirs(results_path, exist_ok=True)
    output_table.to_csv(Path(results_path) / "results.csv", index=False)


def predict_test_set(dataset_path, model_path, parameters, results_path, image_list = None, independent = False, landmarks = False):
    DATASET = dataset_path    
    
    representation = normalize_representation(parameters)
    adj_path = adjacency_data_dir(DATASET, representation)
    if representation == "independent":
        organ_id = np.load("%s/%s/adj_full_organ_id.npy" % (DATASET, adj_path))[:,0]
        organ_order = np.unique(organ_id)
        
        circ_organ_order = {}
        for i, org in enumerate(organ_order):
            # put all the idxs of the organ in the dict
            circ_organ_order[str(int(org))] = np.where(organ_id == org)[0].tolist()
    else:
        # Load organ IDs
        organ_id = np.load("%s/%s/adj_full_organ_id.npy" % (DATASET, adj_path))[:,0]

        unique_organs = set()
        for org_str in organ_id:
            for org in str(org_str).split('-'):
                if org:  # Skip empty strings
                    unique_organs.add(int(org))

        organ_order = sorted(list(unique_organs))
        organ_order = [str(org) for org in organ_order]
        
        with open(f"{DATASET}/{adj_path}/organ_order_full.json", "r") as f:
            circ_organ_order = json.load(f)
    print("Organ order:", organ_order)
            
    config, D_t, U_t, A_t = load_config(dataset_path, parameters)
    
    if config["use_dual"]:
        model = HybridDual(config, D_t, U_t, A_t).to(config['device'])
    else:    
        model = Hybrid(config, D_t, U_t, A_t).to(config['device'])

    if model_path:
        model.load_checkpoint(model_path, config['device'])
    
    model.eval()

    DATASET = config['DATASET']
    if image_list:
        images = image_list
        img_path = None
        print("Using provided image list for testing.")
        print("Number of images:", len(images))
    else:
        images = np.loadtxt("%s/test.txt"%DATASET, dtype = str)
        img_path = "%s/images"%DATASET
    
    test_dataset = TestDataset(images, img_path,
                               transform = transforms.Compose([ScaleImageOnly(config['inputsize']), ToTensorTest()]))
    
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        name = test_dataset.images[i]
        image = sample['image'].unsqueeze(0).to(config['device'])
        
        if test_dataset.img_path is not None:
            image_path = os.path.join(test_dataset.img_path, name)
        else:
            image_path = name

        og_image = cv2.imread(image_path, 0)

        h, w = og_image.shape
        _, size, pad_w, pad_h = process_image(og_image, h, w)

        with torch.no_grad():
            out = model(image)[0]
            out = out.cpu().numpy()[0]
            out = out * size
            out = np.round(out, 0).astype(int)
            out[:,0] -= pad_w
            out[:,1] -= pad_h
            
        if landmarks:
            landmarks_dict = {}
            if organ_order is not None:
                for organ in circ_organ_order.keys():
                    organ_name = config['organ_names'][int(organ) - 1]
                    index_train = circ_organ_order[organ]
                    landmarks_dict[organ_name] = out[index_train].tolist()

            out_path = os.path.join(results_path, name.split("/")[-1].replace(".png", ".json"))
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'w') as f:
                json.dump(landmarks_dict, f)
        
        if not independent:
            mask = create_mask(out, circ_organ_order, h, w)
        else:
            mask = create_mask_independent(out, circ_organ_order, h, w)

        if independent:
            out_path = os.path.join(results_path, "segmentations", name.split("/")[-1].replace(".png", ".npy"))
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.save(out_path, mask)
        else:
            out_path = os.path.join(results_path, name.split("/")[-1])
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, mask)
        
        print("Processed image %d/%d"%(i+1, len(test_dataset)), end = "\r")

from data.dataset import ToTensorWithSeg
import json

def predict_landmarks_from_seg(dataset_path, model_path, parameters, results_path):
    config, D_t, U_t, A_t = load_config(dataset_path, parameters)
    config['raster_as_input'] = True
    
    model = Hybrid(config, D_t, U_t, A_t).to(config['device'])
    print("Image Encoder filters", model.encoder.filters + [model.encoder.filters[-1]])
    print("Bottleneck latents", model.encoder.latents)
    print("Graph convolutional filters", config['filters'][::-1])

    if model_path:
        model.load_checkpoint(model_path, config['device'])
    
    model.eval()
    organ_id = np.load(Path(config['DATASET']) / "adj_full_organ_id.npy")[:,0]
    
    DATASET = config['DATASET']

    images_1 = np.loadtxt("%s/train.txt"%DATASET, dtype = str)
    images_2 = np.loadtxt("%s/val.txt"%DATASET, dtype = str)
    images_3 = np.loadtxt("%s/test.txt"%DATASET, dtype = str)
    images = np.concatenate([images_1, images_2, images_3])
    img_path = "%s/images"%DATASET
    label_path = "%s/landmarks"%DATASET
    
    test_dataset = LandmarksDataset(images, img_path, label_path,
                               transform = transforms.Compose([Scale(config['inputsize']), ToTensorWithSeg()]))
    
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        name = test_dataset.images[i]
        image = sample['raster'].unsqueeze(0).to(config['device'])
        
        if test_dataset.img_path is not None:
            image_path = os.path.join(test_dataset.img_path, name)
        else:
            image_path = name

        og_image = cv2.imread(image_path, 0)

        h, w = og_image.shape
        _, size, pad_w, pad_h = process_image(og_image, h, w)

        with torch.no_grad():
            out = model(image)[0]
            out = out.cpu().numpy()[0]
            out = out * size
            out = np.round(out, 0).astype(int)
            out[:,0] -= pad_w
            out[:,1] -= pad_h
        
        landmarks = {}
            
        for organ in np.unique(organ_id).astype(int):
            index_train = organ_id == int(organ)       
            organ = int(organ)
            landmarks[organ] = out[index_train].tolist()
        
        out_path = os.path.join(results_path, name.replace(label_path, results_path).replace(".png", ".json"))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(landmarks, f)    
            
        print("Processed image %d/%d"%(i+1, len(test_dataset)), end = "\r")


def eval_test_set_xray(dataset_path, model_path, parameters, results_path, save_masks = False):
    
    representation = normalize_representation(parameters)
    if representation == "independent":
        organ_order = None
    else:
        DATASET = dataset_path
        adj_path = adjacency_data_dir(DATASET, representation)
        with open(f"{DATASET}/{adj_path}/organ_order_full.json", "r") as f:
            organ_order = json.load(f)
    
    config, D_t, U_t, A_t = load_config(dataset_path, parameters)
    
    if config["use_dual"]:
        model = HybridDual(config, D_t, U_t, A_t).to(config['device'])
    else:    
        model = Hybrid(config, D_t, U_t, A_t).to(config['device'])

    print("Image Encoder filters", model.encoder.filters + [model.encoder.filters[-1]])
    print("Bottleneck latents", model.encoder.latents)
    print("Graph convolutional filters", config['filters'][::-1])

    if model_path:
        model.load_checkpoint(model_path, config['device'])
    
    model.eval()

    organ_id_path = adjacency_data_dir(config['DATASET'], config['representation'])
    organ_id = np.load(Path(config['DATASET']) / organ_id_path / "adj_full_organ_id.npy")[:,0]
        
    DATASET = config['DATASET']
    images = np.loadtxt("%s/test.txt"%DATASET, dtype = str)
    test_dataset = LandmarksDataset(images, "%s/images"%DATASET, "%s/landmarks"%DATASET, 
                               transform = transforms.Compose([Scale(config['inputsize']), ToTensorWithSeg()]))
    
    output_table = []

    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        name = test_dataset.images[i]
        image = sample['image'].unsqueeze(0).to(config['device'])
        raster = sample['raster'].unsqueeze(0).to(config['device'])
        
        image_path = os.path.join("%s/images"%DATASET, name)
        og_image = cv2.imread(image_path, 0)
        
        h, w = og_image.shape
        _, size, pad_w, pad_h = process_image(og_image, h, w)

        with torch.no_grad():
            if config['raster_as_input']:
                out = model(raster)[0]
            else:
                out = model(image)[0]
            out = out.cpu().numpy()[0]
            out = out * size
            out = np.round(out, 0).astype(int)
            out[:,0] -= pad_w
            out[:,1] -= pad_h

        mask = np.zeros((4, h, w), dtype=np.uint8)
        k = 1
        for organ in np.unique(config['organs']).astype(int):
            if organ_order is not None:
                index_train = organ_order[str(organ)]
            else:
                index_train = organ_id == int(organ)
            contour = out[index_train]
            cv2.drawContours(mask[k], [contour], -1, 1, thickness=cv2.FILLED)
            if int(organ) == 2 or int(organ) == 3:
                k += 1

        og_landmarks = os.path.join("%s/landmarks"%DATASET, name.replace(".png", ".json"))
        if os.path.exists(og_landmarks):    
            with open(og_landmarks, 'r') as f:
                GT_landmarks = json.load(f)

        GT = np.zeros((4, h, w), dtype=np.uint8)
        k = 1
        for organ in GT_landmarks.keys():
            organ = int(organ)
            contour = np.array(GT_landmarks[str(organ)])
            cv2.drawContours(GT[k], [contour], -1, 1, thickness=cv2.FILLED)
            if int(organ) == 2 or int(organ) == 3:
                k += 1
        
        results = []
        organs = [1,2,3]
        organ_names = ["Lungs", "Heart", "Clavicles"]

        available_organs = GT_landmarks.keys() 
        available_organs = [int(organ) for organ in available_organs]
        # remove 2 and 5 from available organs
        available_organs = [organ for organ in available_organs if organ not in [2, 5]]
        
        for t in range(len(available_organs)):
            if available_organs[t] == 3:
                available_organs[t] = 2
            if available_organs[t] == 4:
                available_organs[t] = 3        
        
        for organ_num, organ_name in zip(organs, organ_names):
            if int(organ_num) not in available_organs:
                continue            
            dc_val, hd_val, assd_val = calculate_metrics(mask[int(organ_num)], GT[int(organ_num)])
            results.append({
                "image": name,
                "organ": organ_name,
                "dc": dc_val,
                "hd": hd_val,
                "assd": assd_val,
            })

        output_table.extend(results)

        if save_masks:
            out_path = os.path.join(results_path, "segmentations", name)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, mask)
        
        print("Processed image %d/%d"%(i+1, len(test_dataset)), end = "\r")

    output_table = pd.DataFrame(output_table)
    os.makedirs(results_path, exist_ok=True)
    output_table.to_csv(Path(results_path) / "results.csv", index=False)

def eval_test_set_sunnybrook(dataset_path, model_path, parameters, results_path, save_masks = False):
    
    representation = normalize_representation(parameters)
    if representation == "independent":
        organ_order = None
    else:
        DATASET = dataset_path
        adj_path = adjacency_data_dir(DATASET, representation)
        with open(f"{DATASET}/{adj_path}/organ_order_full.json", "r") as f:
            organ_order = json.load(f)
    
    config, D_t, U_t, A_t = load_config(dataset_path, parameters)
    
    if config["use_dual"]:
        model = HybridDual(config, D_t, U_t, A_t).to(config['device'])
    else:    
        model = Hybrid(config, D_t, U_t, A_t).to(config['device'])

    print("Image Encoder filters", model.encoder.filters + [model.encoder.filters[-1]])
    print("Bottleneck latents", model.encoder.latents)
    print("Graph convolutional filters", config['filters'][::-1])

    if model_path:
        model.load_checkpoint(model_path, config['device'])
    
    model.eval()

    organ_id_path = adjacency_data_dir(config['DATASET'], config['representation'])
    organ_id = np.load(Path(config['DATASET']) / organ_id_path / "adj_full_organ_id.npy")[:,0]
        
    DATASET = config['DATASET']
    images = np.loadtxt("%s/test.txt"%DATASET, dtype = str)
    test_dataset = LandmarksDataset(images, "%s/images"%DATASET, "%s/landmarks"%DATASET, 
                               transform = transforms.Compose([Scale(config['inputsize']), ToTensorWithSeg()]))
    
    organs = config['organs']
    organ_names = config['organ_names']
    
    output_table = []

    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        name = test_dataset.images[i]
        image = sample['image'].unsqueeze(0).to(config['device'])
        raster = sample['raster'].unsqueeze(0).to(config['device'])
        
        image_path = os.path.join("%s/images"%DATASET, name)
        og_image = cv2.imread(image_path, 0)
        
        h, w = og_image.shape
        _, size, pad_w, pad_h = process_image(og_image, h, w)

        with torch.no_grad():
            if config['raster_as_input']:
                out = model(raster)[0]
            else:
                out = model(image)[0]
            out = out.cpu().numpy()[0]
            out = out * size
            out = np.round(out, 0).astype(int)
            out[:,0] -= pad_w
            out[:,1] -= pad_h

        mask = np.zeros((h, w), dtype=np.uint8)
        for organ in [2, 1]: # Draws epicardium, then endocardium inside
            if organ_order is not None:
                index_train = organ_order[str(organ)]
            else:
                index_train = organ_id == int(organ)
            contour = out[index_train]
            cv2.drawContours(mask, [contour], -1, organ, thickness=cv2.FILLED)

        og_landmarks = os.path.join("%s/landmarks"%DATASET, name.replace(".png", ".json"))
        if os.path.exists(og_landmarks):    
            with open(og_landmarks, 'r') as f:
                GT_landmarks = json.load(f)

        GT = np.zeros((h, w), dtype=np.uint8)
        for organ in ["2", "1"]:  # Draws epicardium, then endocardium inside
            if organ not in GT_landmarks:
                continue
            organ = int(organ)
            contour = np.array(GT_landmarks[str(organ)])
            cv2.drawContours(GT, [contour], -1, organ, thickness=cv2.FILLED)
        
        results = []

        for organ_num, organ_name in zip(organs, organ_names):
            if organ_num not in GT_landmarks:
                continue
            
            dc_val, hd_val, assd_val = calculate_metrics(mask == int(organ_num), GT == int(organ_num))
            results.append({
                "image": name,
                "organ": organ_name,
                "dc": dc_val,
                "hd": hd_val,
                "assd": assd_val,
            })

        output_table.extend(results)

        if save_masks:
            out_path = os.path.join(results_path, "segmentations", name)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, mask)
        
        print("Processed image %d/%d"%(i+1, len(test_dataset)), end = "\r")

    output_table = pd.DataFrame(output_table)
    os.makedirs(results_path, exist_ok=True)
    output_table.to_csv(Path(results_path) / "results.csv", index=False)