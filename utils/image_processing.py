import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pathlib import Path
from typing import Union, Tuple, Dict, Any
from skimage import transform
import cv2
import json

def draw_organ(ax, array, color = 'b', radius = 9):
    N = array.shape[0]
    for i in range(0, N):
        x, y = array[i,:]
        circ = plt.Circle((x, y), radius=radius, color=color, fill = True)
        ax.add_patch(circ)
    return

def draw_lines(ax, array, color = 'b'):
    N = array.shape[0]
    for i in range(0, N):
        x1, y1 = array[i-1,:]
        x2, y2 = array[i,:]
        ax.plot([x1, x2], [y1, y2], color=color, linestyle='-', linewidth=1.5)
    return

def sitk_load(filepath: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Loads an image using SimpleITK and returns the image and its metadata.

    Args:
        filepath: Path to the image.

    Returns:
        - np.ndarray: The image array.
        - Dict[str, Any]: Collection of metadata.
    """
    # Load image and save info
    image = sitk.ReadImage(str(filepath))
    info = {"origin": image.GetOrigin(), "spacing": image.GetSpacing(), "direction": image.GetDirection()}

    # Extract numpy array from the SimpleITK image object
    im_array = np.squeeze(sitk.GetArrayFromImage(image))

    return im_array, info

def process_image(array, h, w, size = 512):
    if h > w:
        pad_w, pad_h = (h-w)//2, 0
    else:
        pad_h, pad_w = (w-h)//2, 0
    
    array_padded = np.pad(array, ((pad_h, pad_h), (pad_w, pad_w)))
    size = max(h, w)
    array_resized = transform.resize(array_padded, (size, size))
    return np.expand_dims(array_resized, axis=0), size, pad_w, pad_h

def create_mask(out, circ_organ_order, h, w):
    n_organs = len(circ_organ_order) + 1
    mask = np.zeros((n_organs, h, w))
    organs = list(circ_organ_order.keys())
    for organ in organs:
        index_train = circ_organ_order[organ]
        mask[int(organ)] = cv2.drawContours(mask[int(organ)], [out[index_train]], -1, 1, -1)
    mask[0, :, :] = 1 - np.sum(mask[1:, :, :], axis=0)
    return np.argmax(mask, axis=0)


def create_mask_independent(out, circ_organ_order, h, w):
    # creates a mask where each organ is in a different channel
    n_organs = len(circ_organ_order)
    mask = np.zeros((n_organs, h, w))
    for organ, indices in circ_organ_order.items():
        mask[int(organ)] = cv2.drawContours(mask[int(organ)], [out[indices]], -1, 1, -1)
    return mask

def generate_landmark_contours(config):
    """
    Generates landmark contours for the organs provided in the config.
    
    Args:
    config (dict): Configuration dictionary containing paths and other settings.
    image_list_path (str): Path to the text file containing image file paths.
    mask_list_path (str): Path to the text file containing mask file paths.
    
    Returns:
    str: Path to the landmarks directory.
    """

    image_list_path = Path(config['output_path']) / 'image_list.txt'
    mask_list_path = Path(config['output_path']) / 'mask_list.txt'

    print(f'Loading image list from {image_list_path}')
    print(f'Loading mask list from {mask_list_path}')
    
    landmarks_dir = Path(config['output_path']) / 'landmarks'
    landmarks_dir.mkdir(parents=True, exist_ok=True)
    
    with open(image_list_path, 'r') as f:
        image_list = f.read().splitlines()
    with open(mask_list_path, 'r') as f:
        mask_list = f.read().splitlines()
        
    print(f'Generating landmarks for {len(image_list)} images.')
    
    for image_path, mask_path in zip(image_list, mask_list):
        # Load image and mask
        # You might need to adjust this depending on your file format
        mask = cv2.imread(str(Path(config['output_path']) / 'masks' / mask_path), cv2.IMREAD_GRAYSCALE)
        
        landmarks = {}
        for organ in config['organs']:
            organ_mask = (mask == int(organ)).astype(np.uint8)
            contours, _ = cv2.findContours(organ_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                landmarks[organ] = largest_contour.reshape(-1, 2).tolist()
        
        # Save landmarks
        landmark_path = landmarks_dir / Path(image_path).with_suffix('.json')
        landmark_path.parent.mkdir(parents=True, exist_ok=True)
        with open(landmark_path, 'w') as f:
            json.dump(landmarks, f)
    
    return str(landmarks_dir)

def get_contour_lengths(config, landmarks_path, threshold=0):
    """
    Reads all the landmark contours and obtains the contour lengths.
    
    Args:
    config (dict): Configuration dictionary containing paths and other settings.
    landmarks_path (str): Path to the landmarks directory.
    
    Returns:
    dict: A dictionary where keys are organ names and values are lists of contour lengths.
    """
    contour_lengths = {organ: [] for organ in config['organs']}
    landmarks_dir = Path(landmarks_path)
    
    for landmark_file in landmarks_dir.rglob('*.json'):
        removed = 0

        with open(landmark_file, 'r') as f:
            landmarks = json.load(f)
        
        to_pop = []
        
        for organ, contour in landmarks.items():
            if organ in config['organs']:
                lenght = len(contour)
                if lenght > threshold:
                    contour_lengths[organ].append(len(contour))
                else:
                    to_pop.append(organ)
                    removed += 1
        
        if removed > 0:
            for organ in to_pop:
                landmarks.pop(organ)
            with open(landmark_file, 'w') as f:
                json.dump(landmarks, f)
            print(f'Removed {removed} contours from {landmark_file}')

    
    return contour_lengths