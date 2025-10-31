import cv2
import numpy as np
import pandas as pd
import os
from evaluation.utils import calculate_metrics

def fit_ellipse(im):    
    # find contours
    contours, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_sizes = [len(contour) for contour in contours]
    ellipse = cv2.fitEllipse(contours[np.argmax(contour_sizes)])
    return ellipse
     
df_pixel_size = pd.read_csv('../Fetal/HC18/training_set_pixel_size_and_HC.csv')
submission = {'filename': [], 'center_x_mm': [], 'center_y_mm': [], 'semi_axes_a_mm': [],
             'semi_axes_b_mm': [], 'angle_rad': [], 'Index': [], 'HC': [] , 'Pred HC': [],
             'dc': [], 'hd': []}

file_names = sorted([file for _,_,files in os.walk('../Fetal/HC18/training_set_results/') for file in files])

for i, file_name in enumerate(file_names):
    im = cv2.imread('../Fetal/HC18/training_set_results/'+file_name, 0)
    im = (im == 2).astype(np.uint8)
    ellipse = fit_ellipse(im)
    im_mask = cv2.ellipse(np.zeros_like(im),ellipse,(255,255,255),-1)
    #image_name = file_name.split('.')[0] + '_Annotation_fit.png'
    #cv2.imwrite('../Fetal/HC18/training_set/'+image_name, im_mask)
    
    pixel_size = df_pixel_size.loc[df_pixel_size['filename'] == file_name, 'pixel size(mm)'].iloc[0]
    submission['filename'].extend([file_name])
    submission['Index'].extend([int(file_name.split('_')[0])])
    center_x,  center_y = ellipse[0]
    submission['center_x_mm'].extend([pixel_size*center_x])
    submission['center_y_mm'].extend([pixel_size*center_y])
    semi_axes_b, semi_axes_a = ellipse[1]
    if semi_axes_b > semi_axes_a:
        semi_axes_b = semi_axes_b + semi_axes_a
        semi_axes_a = semi_axes_b - semi_axes_a
        semi_axes_b -= semi_axes_a
    submission['semi_axes_a_mm'].extend([semi_axes_a*pixel_size/2])
    submission['semi_axes_b_mm'].extend([semi_axes_b*pixel_size/2])
    angle = ellipse[2]
    if angle < 90:
        angle += 90
    else:
        angle -= 90
    submission['angle_rad'].extend([np.deg2rad(angle)])
    submission['HC'].extend([df_pixel_size.loc[df_pixel_size['filename'] == file_name, 'head circumference (mm)'].iloc[0]])
    # pred HC is the circumference of the ellipse
    pred_HC = 2 * np.pi * np.sqrt((semi_axes_a*pixel_size/2) * (semi_axes_b*pixel_size/2))
    submission['Pred HC'].extend([pred_HC])

    gt = cv2.imread('../Fetal/HC18/training_set/'+file_name.replace('.png', '_Annotation.png'), 0)

    contours, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gt = np.zeros_like(gt)
    cv2.drawContours(gt, contours, -1, 2, -1)

    # estimate dice and hausdorff distance
    dc, hd, _ = calculate_metrics(im == 1, gt == 2)

    submission['dc'].extend([dc])
    submission['hd'].extend([hd])

submission = pd.DataFrame(submission)
submission = submission[['filename', 'center_x_mm', 'center_y_mm', 
                         'semi_axes_a_mm', 'semi_axes_b_mm', 'angle_rad', 'Index', 'HC', 'Pred HC', 'dc', 'hd']]
submission = submission.sort_values(['Index'])
submission = submission.drop('Index', axis=1)
submission.to_csv("../Fetal/HC18/Train_HC18.csv", index=False)
print('Required .csv file generated')
     
