import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np


# this function only resizes the original image
def level1_augment_image(image_path, save_path):
    image = Image.open(image_path)

    transform = transforms.Compose([transforms.Resize((224, 224))])

    augmented_image = transform(image)
    augmented_image.save(save_path)


# this function resizes the original image and augments it
def level2_augment_image(image_path, save_path):
    image = Image.open(image_path)

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation((0, 360)),
            transforms.ColorJitter(brightness=0.5),
            transforms.Resize((224, 224)),
        ]
    )

    augmented_image = transform(image)
    augmented_image.save(save_path)


# this function augments the original image and takes a random crop of the image
def level3_augment_image(image_path, save_path):
    image = Image.open(image_path)

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation((0, 360)),
            transforms.ColorJitter(brightness=0.5),
            transforms.RandomResizedCrop((224, 224)),
        ]
    )

    augmented_image = transform(image)
    augmented_image.save(save_path)


# this function will call level1_augment_image, for ALL images in the subdirectory
def level1_augment_subdirectory(subdirectory, save_path):
    print(f"Level 1 augmenting {subdirectory}")
    for image in os.listdir(subdirectory):
        image_path = os.path.join(subdirectory, image)
        sp = os.path.join(save_path, image.split(".")[0] + "_resized.jpg")

        level1_augment_image(image_path, sp)


# this function will call level2_augment_image, for NUM_OF_IMAGES_TO_AUGMENT images in the subdirectory
def level2_augment_subdirectory(subdirectory, save_path, num_of_images_to_augment):
    print(f"Level 2 augmenting {subdirectory}")
    images = os.listdir(subdirectory)
    num_images = len(images)

    random_indices = np.random.choice(num_images, num_of_images_to_augment)

    i = 0
    for index in random_indices:
        image = images[index]
        image_path = os.path.join(subdirectory, image)
        sp = os.path.join(save_path, image.split(".")[0] + f"_augmented{i}.jpg")

        level2_augment_image(image_path, sp)

        i += 1


# this function will call level3_augment_image, for NUM_OF_IMAGES_TO_AUGMENT images in the subdirectory
def level3_augment_subdirectory(subdirectory, save_path, num_of_images_to_augment):
    print(f"Level 3 augmenting {subdirectory}")
    images = os.listdir(subdirectory)
    num_images = len(images)

    random_indices = np.random.choice(num_images, num_of_images_to_augment)

    i = 0
    for index in random_indices:
        image = images[index]
        image_path = os.path.join(subdirectory, image)
        sp = os.path.join(save_path, image.split(".")[0] + f"_cropped_augmented{i}.jpg")

        level3_augment_image(image_path, sp)

        i += 1


def process_subdirectory(subdirectory, save_path):
    print(f"Processing {subdirectory}")
    num_of_images = len(os.listdir(subdirectory))

    if num_of_images >= 200:
        level1_augment_subdirectory(subdirectory, save_path)
    elif num_of_images >= 100:
        level1_augment_subdirectory(subdirectory, save_path)
        level2_augment_subdirectory(subdirectory, save_path, 200 - num_of_images)
    else:
        level1_augment_subdirectory(subdirectory, save_path)
        level2_augment_subdirectory(subdirectory, save_path, 150 - num_of_images)
        level3_augment_subdirectory(subdirectory, save_path, 50)


# this function will create test samples for the subdirectory
def create_test_samples(subdirectory, save_path):
    print(f"Creating test samples for {subdirectory}")
    level2_augment_subdirectory(subdirectory, save_path, 25)
    level3_augment_subdirectory(subdirectory, save_path, 25)


if __name__ == "__main__":
    original_data_directory = "data/train"
    augmented_data_directory = "augmented-data/train"

    subdirectories = os.listdir(original_data_directory)

    # for training
    for subdirectory in subdirectories:
        save_path = os.path.join(augmented_data_directory, subdirectory)
        os.makedirs(save_path, exist_ok=True)

        process_subdirectory(
            os.path.join(original_data_directory, subdirectory), save_path
        )

    # for testing
    augmented_data_directory = "augmented-data/test"
    os.makedirs(augmented_data_directory, exist_ok=True)

    for subdirectory in subdirectories:
        save_path = os.path.join(augmented_data_directory, subdirectory)
        os.makedirs(save_path, exist_ok=True)

        create_test_samples(
            os.path.join(original_data_directory, subdirectory), save_path
        )
