import numpy as np
import cv2

def pad_to_square_sequence(input_sequence):
    """ Pad the input image sequence to make it square.
    Args:
        input_sequence (numpy array): The input image sequence TxHxW. All images in the sequence must have the same size.
    Returns:
        img (numpy array): The padded image sequence.
        padding (tuple): Padding information for the original image.
    """
    T, h, w = input_sequence.shape[:3]
    
    if h > w:
        padw = (h - w) 
        auxw = padw % 2
        img = np.pad(input_sequence, ((0, 0), (0, 0), (padw//2, padw//2 + auxw)), 'constant')
        
        padh = 0
        auxh = 0
        
    else:
        padh = (w - h) 
        auxh = padh % 2
        img = np.pad(input_sequence, ((0, 0), (padh//2, padh//2 + auxh), (0, 0)), 'constant')

        padw = 0
        auxw = 0
        
    return img, (padh, padw, auxh, auxw)

def preprocess_sequence(input_sequence, image_size=512):
    """ Preprocess the input image to fit the model requirements.
    Args:
        input_sequence (numpy array): The input image sequence TxHxW. All images in the sequence must have the same size.
        image_size (int): The desired size of the output image.
    Returns:
        img (numpy array): The preprocessed image.
        padding (tuple): Padding information for the original image.
    """
    img, padding = pad_to_square_sequence(input_sequence)

    T, h, w = img.shape[:3]
    if h != image_size or w != image_size:
        img = np.array([cv2.resize(img[t], (image_size, image_size), interpolation = cv2.INTER_CUBIC) for t in range(T)])

    return img, (h, w, padding)

def removePreprocess_sequence(output, info, image_size=512):
    """ Remove the preprocessing applied to the output.
    Args:
        output (numpy array): The output from the model with sequence as batch.
        info (tuple): Information about the original image size and padding.
        image_size (int): The size of the image after preprocessing.
    Returns:
        output (numpy array): The output adjusted to the original image size.
    """
    h, w, padding = info

    if h != image_size or w != image_size:
        output = output * h
    else:
        output = output * image_size
    
    padh, padw, auxh, auxw = padding
    
    output[:, :, 0] = output[:, :, 0] - padw//2
    output[:, :, 1] = output[:, :, 1] - padh//2
    
    return output   



def pad_to_square(img):
    h, w = img.shape[:2]
    
    if h > w:
        padw = (h - w) 
        auxw = padw % 2
        img = np.pad(img, ((0, 0), (padw//2, padw//2 + auxw)), 'constant')
        
        padh = 0
        auxh = 0
        
    else:
        padh = (w - h) 
        auxh = padh % 2
        img = np.pad(img, ((padh//2, padh//2 + auxh), (0, 0)), 'constant')

        padw = 0
        auxw = 0
        
    return img, (padh, padw, auxh, auxw)

def preprocess(input_img, image_size=512):
    """ Preprocess the input image to fit the model requirements.
    Args:
        input_img (numpy array): The input image to preprocess.
        image_size (int): The desired size of the output image.
    Returns:
        img (numpy array): The preprocessed image.
        padding (tuple): Padding information for the original image.
    """
    img, padding = pad_to_square(input_img)
    
    h, w = img.shape[:2]
    if h != image_size or w != image_size:
        img = cv2.resize(img, (image_size, image_size), interpolation = cv2.INTER_CUBIC)

    return img, (h, w, padding)

def removePreprocess(output, info, image_size=512):
    """ Remove the preprocessing applied to the output.
    Args:
        output (numpy array): The output from the model.
        info (tuple): Information about the original image size and padding.
        image_size (int): The size of the image after preprocessing.
    Returns:
        output (numpy array): The output adjusted to the original image size.
    """
    h, w, padding = info

    if h != image_size or w != image_size:
        output = output * h
    else:
        output = output * image_size
    
    padh, padw, auxh, auxw = padding
    
    output[:, 0] = output[:, 0] - padw//2
    output[:, 1] = output[:, 1] - padh//2
    
    return output   