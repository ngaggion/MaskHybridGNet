from evaluation.utils import predict_landmarks_from_seg
import json
import os

DATASET = "../Dataset/CAMUS/Landmarks_3_10"
NAME = "CAMUS_Seg_2_Graph"

model_checkpoint = "../Training/%s/%s.pth" % (NAME, NAME)
parameters = json.load(open("../Training/%s/hyperparameters.json"%NAME))

results_path = "../Dataset/CAMUS/Landmarks_3_10/landmarks2"

predict_landmarks_from_seg(DATASET, model_checkpoint, parameters, results_path)