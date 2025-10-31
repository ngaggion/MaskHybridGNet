from evaluation.utils import predict_test_set
import json
import os

DATASET = "../Dataset/CAMUS/Landmarks_3_10"
NAME = "CAMUS_NN_no_raster"
model_checkpoint = "../Training/%s/%s_best.pth" % (NAME, NAME)
parameters = json.load(open("../Training/%s/hyperparameters.json"%NAME))

results_path = "../CAMUS_Landmarks/%s" % NAME

predict_test_set(DATASET, model_checkpoint, parameters, results_path, landmarks = True)
