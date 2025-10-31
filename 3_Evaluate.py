from evaluation.utils import eval_test_set 
from evaluation.utils import eval_test_set_sunnybrook
from evaluation.utils import eval_test_set_sunnybrook_tta
import json
import os
import warnings
warnings.filterwarnings("ignore")

"""
DATASET = "../Dataset/CXRAY/Landmarks_3_10"

NAMES = ["ChestXRay_10_NN_no_raster",
         "ChestXRay_10_dual_NN_no_raster",
         "ChestXRay_10_no_raster",
         "ChestXRay_10_dual_no_raster",
         "ChestXRay_10",
         "ChestXRay_10_dual",
         "ChestXRay_10_NN",
         "ChestXRay_10_dual_NN"      
]
"""

"""
DATASET = "../Dataset/CAMUS/Landmarks_3_10"

NAMES = ["CAMUS_NN",
         "CAMUS_NN_dual",
         "CAMUS_NN_no_raster",
         "CAMUS_NN_dual_no_raster",
         "CAMUS_s",
         "CAMUS_dual"
         "CAMUS_atlas_seg_NN",
         "CAMUS_atlas_seg",]



DATASET = "../Dataset/Merged/Landmarks_3_10"

NAMES = ["HC18",
         "Merged",
         "PSFHS",
         "JNU-IFM"]

DATASET = "../Dataset/PAXRay_Front/Landmarks_3_10"

NAMES = ["PRF_dual_no_raster_NN_v3"]

for NAME in NAMES:
    model_checkpoint = "../Training/%s/%s.pth" % (NAME, NAME)
    parameters = json.load(open("../Training/%s/hyperparameters.json"%NAME))
    results_path = "../Results/PRF/%s" % NAME
    os.makedirs(results_path, exist_ok=True)

    eval_test_set(DATASET, model_checkpoint, parameters, results_path)


"""

NAMES = ["Sunnybrook_dual_4",
         "Sunnybrook_2",
         "Sunnybrook_3",
         "Sunnybrook_4",
         "Sunnybrook_dual_2",
         "Sunnybrook_dual_3"]

for NAME in NAMES:
    
    if "2" in NAME:
        DATASET = "../Dataset/Sunnybrook/Landmarks_2_10"
    elif "3" in NAME:
        DATASET = "../Dataset/Sunnybrook/Landmarks_3_10"
    elif "4" in NAME:
        DATASET = "../Dataset/Sunnybrook/Landmarks_4_10"
    
    model_checkpoint = "../Trained/SunnyBrook/%s/%s.pth" % (NAME, NAME)
    parameters = json.load(open("../Trained/SunnyBrook/%s/hyperparameters.json"%NAME))
    results_path = "../Results/Sunnybrook/%s" % NAME
    os.makedirs(results_path, exist_ok=True)

    eval_test_set_sunnybrook_tta(DATASET, model_checkpoint, parameters, results_path, use_tta=True)