from evaluation.utils import predict_test_set
import json
import os

DATASET = "../Dataset/CAMUS/Landmarks_3_10"
NAME = "CAMUS_NN_dual"
model_checkpoint = "../Trained/CAMUS_Trains/%s/%s.pth" % (NAME, NAME)
parameters = json.load(open("../Trained/CAMUS_Trains/%s/hyperparameters.json"%NAME))
parameters["naive"] = False
parameters["use_dual"] = True
parameters["dual"] = True

image_test_folder = "/home/ngaggion/Documents/Gradio/Chest-x-ray-HybridGNet-Segmentation/datasets/heart"
results_path = "/home/ngaggion/Documents/Gradio/Chest-x-ray-HybridGNet-Segmentation/datasets/heart/results"

if not os.path.exists(results_path):
    os.makedirs(results_path)
    
image_list = []
for image in os.listdir(image_test_folder):
    # if is folder, skip
    if os.path.isdir(os.path.join(image_test_folder, image)):
        continue
    # if is json, skip
    if not image.endswith(".png"):
        continue
    image_list.append(image_test_folder + "/" + image)

predict_test_set(DATASET, model_checkpoint, parameters, results_path, image_list = image_list, independent = False, landmarks=True)