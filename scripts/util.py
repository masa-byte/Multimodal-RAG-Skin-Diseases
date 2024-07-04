import json
import pandas as pd
import torch
import os
from PIL import Image

# Functions which are used in multiple scripts are defined here
# Those used in only one script are defined in the script itself


def read_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        return data


def write_json_file(data, file_path, mode="w"):
    with open(file_path, mode) as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def read_csv_file(file_path):
    data = pd.read_csv(file_path)
    return data


def get_model_path():
    save_directory = "../models"
    os.makedirs(save_directory, exist_ok=True)
    model_path = os.path.join(save_directory, "align-model-trained.pth")
    return model_path


def get_optimizer_path():
    save_directory = "../models"
    os.makedirs(save_directory, exist_ok=True)
    optimizer_path = os.path.join(save_directory, "align-optimizer-trained.pth")
    return optimizer_path


def load_model(model, model_path, optimizer=None, optimizer_path=None):
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Model loaded from", model_path)
    if (
        optimizer is not None
        and optimizer_path is not None
        and os.path.exists(optimizer_path)
    ):
        optimizer.load_state_dict(torch.load(optimizer_path))
        print("Optimizer loaded from", optimizer_path)


def get_train_image_directory():
    image_directory = "../augmented-data/train"
    return image_directory


def get_disease_skin_symptoms():
    path = "../data-info/disease-skin-symptoms.json"
    data = read_json_file(path)
    return data


def get_disease_descriptions():
    path = "../data-info/disease-descriptions.json"
    data = read_json_file(path)
    return data
