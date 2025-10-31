import cv2
import numpy as np
from skimage import transform

class Scale(object):
    def __init__(self, size = 512):
        self.size = size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        ids = landmarks[:,2]
        landmarks = landmarks[:,:2]
        
        h, w = image.shape[:2]

        # if not square, make it square
        if h > w:
            pad = (h - w) // 2
            image = np.pad(image, ((0,0), (pad, pad)), mode='constant', constant_values=0)
            landmarks[:,0] += pad
        elif w > h:
            pad = (w - h) // 2
            image = np.pad(image, ((pad, pad), (0,0)), mode='constant', constant_values=0)
            landmarks[:,1] += pad

        h, w = image.shape[:2]
        
        landmarks[:,0] = landmarks[:,0] * self.size / w
        landmarks[:,1] = landmarks[:,1] * self.size / h
        
        img = transform.resize(image, (self.size, self.size))
        
        landmarks = np.hstack([landmarks, ids.reshape(-1,1)])
        
        return {'image': img, 'landmarks': landmarks}


class ScaleImageOnly(object):
    def __init__(self, size = 512):
        self.size = size

    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]

        # if not square, make it square
        if h > w:
            pad = (h - w) // 2
            image = np.pad(image, ((0,0), (pad, pad)), mode='constant', constant_values=0)
        elif w > h:
            pad = (w - h) // 2
            image = np.pad(image, ((pad, pad), (0,0)), mode='constant', constant_values=0)

        h, w = image.shape[:2]
                
        img = transform.resize(image, (self.size, self.size))
                
        return {'image': img}

class RandomScaleCentered(object):
    def __init__(self, size = 512):
        self.size = size
        
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']       
        
        ids = landmarks[:,2]
        landmarks = landmarks[:,:2]
        
        # Set limits for the scaling factor
        min_x = np.min(landmarks[:,0]) 
        max_x = self.size - np.max(landmarks[:,0])
        
        dist_to_border_x = np.min([min_x, max_x])
        width = self.size - 2 * dist_to_border_x 
        
        min_y = np.min(landmarks[:,1])
        max_y = self.size - np.max(landmarks[:,1])
        
        dist_to_border_y = np.min([min_y, max_y])
        height = self.size - 2 * dist_to_border_y
        
        max_var_x = min(self.size / width, 1.20)
        min_var_x = 0.70
        
        max_var_y = min(self.size / height, 1.20)
        min_var_y = 0.70
                                
        varx = np.random.uniform(min_var_x, max_var_x)
        vary = np.random.uniform(min_var_y, max_var_y)
                
        landmarks[:,0] = landmarks[:,0] * varx
        landmarks[:,1] = landmarks[:,1] * vary
        
        h, w = image.shape[:2]
        new_h = np.round(h * vary).astype('int')
        new_w = np.round(w * varx).astype('int')

        img = transform.resize(image, (new_h, new_w))
                
        if new_h > self.size:
            # crop centered
            start_h = (new_h - self.size) // 2
            img = img[start_h:start_h+self.size, :]
            landmarks[:,1] -= start_h
        elif new_h < self.size:
            # pad
            pad_h_left = (self.size - new_h) // 2
            pad_h_right = self.size - new_h - pad_h_left
            img = np.pad(img, ((pad_h_left, pad_h_right), (0,0)), mode='constant', constant_values=0)
            landmarks[:,1] += pad_h_left
        
        if new_w > self.size:
            # crop centered
            start_w = (new_w - self.size) // 2
            img = img[:, start_w:start_w+self.size]
            landmarks[:,0] -= start_w
        elif new_w < self.size:
            # pad
            pad_w_left = (self.size - new_w) // 2
            pad_w_right = self.size - new_w - pad_w_left
            img = np.pad(img, ((0,0), (pad_w_left, pad_w_right)), mode='constant', constant_values=0)
            landmarks[:,0] += pad_w_left
        
        if img.shape[0] != self.size or img.shape[1] != self.size:
            print('Original', [new_h,new_w])
            print('Salida', img.shape)
            raise Exception('Error')
        
        landmarks = np.hstack([landmarks, ids.reshape(-1,1)])
            
        return {'image': img, 'landmarks': landmarks}
        

class Rotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        try:
            image, landmarks = sample['image'], sample['landmarks']
            
            original_image = image.copy()
            original_landmarks = landmarks.copy()
            
            # Get original image dimensions
            h, w = image.shape[:2]
            
            ids = landmarks[:, 2]
            landmarks = landmarks[:, :2]
            
            # Get a random angle on a normal distribution, with the given standard deviation
            angle = np.random.normal(0, self.angle / 3)
            
            # Compute the padding size based on the image diagonal length
            diagonal = int(np.ceil(np.sqrt(h ** 2 + w ** 2)))
            pad_x = (diagonal - w) // 2
            pad_y = (diagonal - h) // 2
            
            # Pad the image
            padded_image = cv2.copyMakeBorder(image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0)
            padded_h, padded_w = padded_image.shape[:2]
            
            # Rotate the padded image
            center = (padded_w // 2, padded_h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            rotated_image = cv2.warpAffine(padded_image, rotation_matrix, (padded_w, padded_h))
            
            # Compute the rotated landmarks
            landmarks += np.array([pad_x, pad_y])  # Account for padding
            ones = np.ones(shape=(len(landmarks), 1))
            landmarks_hom = np.hstack([landmarks, ones])
            rotated_landmarks = np.dot(rotation_matrix, landmarks_hom.T).T
            
            # Crop the center of the rotated image to the original size
            x_start = (padded_w - w) // 2
            y_start = (padded_h - h) // 2
            cropped_image = rotated_image[y_start:y_start+h, x_start:x_start+w]
            
            # Adjust landmarks to the cropped image coordinates
            adjusted_landmarks = rotated_landmarks - np.array([x_start, y_start])
            
            # Check if the cropped image has the correct shape
            if cropped_image.shape[:2] != (h, w):
                print("Warning: Cropped image has incorrect shape. Attempting to resize.")
                cropped_image = cv2.resize(cropped_image, (w, h))
                
                # Check if resize was successful
                if cropped_image.shape[:2] != (h, w):
                    print("Error: Unable to resize image to correct shape. Returning original sample.")
                    return {'image': original_image, 'landmarks': original_landmarks}
                
                adjusted_landmarks *= np.array([w, h]) / np.array(cropped_image.shape[:2])
            
            # Final check on image shape
            if cropped_image.shape != original_image.shape:
                print("Error: Final image shape does not match original. Returning original sample.")
                return {'image': original_image, 'landmarks': original_landmarks}
            
            # Combine landmarks with their ids
            final_landmarks = np.hstack([adjusted_landmarks, ids.reshape(-1,1)])
            
            return {'image': cropped_image, 'landmarks': final_landmarks}

        except Exception as e:
            print(f"Error during rotation: {str(e)}. Returning original sample.")
            return {'image': original_image, 'landmarks': original_landmarks}
        

class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        original_image = image.copy()
        original_landmarks = landmarks.copy()

        _, w = image.shape[:2]

        if np.random.random() < self.prob:
            # Flip the image
            flipped_image = cv2.flip(image, 1)  # 1 means horizontal flip

            # Flip the landmarks
            flipped_landmarks = landmarks.copy()
            flipped_landmarks[:, 0] = w - landmarks[:, 0]  # Flip x-coordinates

            # Check if the flipped image has the correct shape
            if flipped_image.shape != original_image.shape:
                print("Error: Flipped image shape does not match original. Returning original sample.")
                return {'image': original_image, 'landmarks': original_landmarks}

            return {'image': flipped_image, 'landmarks': flipped_landmarks}
        else:
            # If not flipping, return the original sample
            return sample
        

class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        original_image = image.copy()
        original_landmarks = landmarks.copy()

        h, w = image.shape[:2]

        if np.random.random() < self.prob:
            # Flip the image
            flipped_image = cv2.flip(image, 0)  # 1 means horizontal flip

            # Flip the landmarks
            flipped_landmarks = landmarks.copy()
            flipped_landmarks[:, 1] = h - landmarks[:, 1]  # Flip y-coordinates

            # Check if the flipped image has the correct shape
            if flipped_image.shape != original_image.shape:
                print("Error: Flipped image shape does not match original. Returning original sample.")
                return {'image': original_image, 'landmarks': original_landmarks}

            return {'image': flipped_image, 'landmarks': flipped_landmarks}
        else:
            # If not flipping, return the original sample
            return sample

class RandomTranspose(object):
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        original_image = image.copy()
        original_landmarks = landmarks.copy()
        h, w = image.shape[:2]

        if np.random.random() < self.prob:
            # Transpose the image
            transposed_image = cv2.transpose(image)

            # Transpose the landmarks
            transposed_landmarks = landmarks.copy()
            transposed_landmarks[:, 0] = landmarks[:, 1]
            transposed_landmarks[:, 1] = landmarks[:, 0]

            return {'image': transposed_image, 'landmarks': transposed_landmarks}
        else:
            # If not transposing, return the original sample
            return sample        

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return np.float32(cv2.LUT(image.astype('uint8'), table))


class AugColor(object):
    def __init__(self, gammaFactor):
        self.gammaf = gammaFactor

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        # Gamma
        gamma = np.random.uniform(1 - self.gammaf, 1 + self.gammaf / 2)
        
        image = adjust_gamma(image * 255, gamma) / 255
        
        # Adds a little noise
        image = image + np.random.normal(0, 1/128, image.shape)
        
        return {'image': image, 'landmarks': landmarks}