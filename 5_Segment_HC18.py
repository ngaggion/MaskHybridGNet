from evaluation.utils import predict_test_set
import json
import os

DATASET = "../Dataset/Merged/Landmarks_3_10"
NAME = "HC18"
model_checkpoint = "../Training/%s/%s.pth" % (NAME, NAME)
parameters = json.load(open("../Training/%s/hyperparameters.json"%NAME))
parameters["naive"] = True
parameters["dual"] = True

image_test_folder = "../Fetal/HC18/training_set"
results_path = "../Fetal/HC18/training_set_results"
if not os.path.exists(results_path):
    os.makedirs(results_path)
image_list = []
for image in os.listdir(image_test_folder):
    if not "Annotation" in image:
        image_list.append(image_test_folder + "/" + image)

predict_test_set(DATASET, model_checkpoint, parameters, results_path, image_list = image_list, independent = False)

image_test_folder = "../Fetal/HC18/test_set"
results_path = "../Fetal/HC18/test_set_results"
if not os.path.exists(results_path):
    os.makedirs(results_path)
image_list = []
for image in os.listdir(image_test_folder):
    if not "Annotation" in image:
        image_list.append(image_test_folder + "/" + image)

predict_test_set(DATASET, model_checkpoint, parameters, results_path, image_list = image_list, independent = False)
