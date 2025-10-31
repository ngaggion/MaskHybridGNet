from evaluation.utils import predict_test_set
import json
import os

DATASET = "../Dataset/CXRAY/Landmarks_3_10"
NAME = "ChestXRay_10_dual_NN_prod"
model_checkpoint = "../Training/%s/%s.pth" % (NAME, NAME)
parameters = json.load(open("../Training/%s/hyperparameters.json"%NAME))

image_test_folder = "../CheXmask/FinalDatabase/Images"
results_path = "../CheXmask/FinalDatabase/Preds"
if not os.path.exists(results_path):
    os.makedirs(results_path)
image_list = []
for image in os.listdir(image_test_folder):
    if not "Annotation" in image:
        image_list.append(image_test_folder + "/" + image)

predict_test_set(DATASET, model_checkpoint, parameters, results_path, image_list = image_list, landmarks = True)
