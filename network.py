#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pprint
import random

def validation_split_check(value):
    fvalue = float(value)
    if fvalue != 0 and fvalue < 0 or fvalue > 1:
        raise argparse.ArgumentTypeError("%s is not a valid validation split value (must be between 0 and 1)" % value)
    return fvalue

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    WHITE  = '\33[37m'

# function for debugging. prints a given message and then exits the script
# call like:
#   dier('debug_message')
# or
#   dier(object)

def dier (msg):
    pprint.pprint(msg)
    sys.exit(1)


if not os.environ.get('VIRTUAL_ENV'):
    print(f"Not running inside a virtual environment. Make sure you called `{bcolors.BOLD}{bcolors.WHITE}bash install.sh{bcolors.ENDC}` first one time and this script over `{bcolors.BOLD}{bcolors.WHITE}bash run.sh{bcolors.ENDC}` (always)")
    sys.exit(1)

def bold_white_print (msg):
    print(bcolors.BOLD + bcolors.WHITE + str(msg) + bcolors.ENDC)

import site
import json
import re
import os.path
import re
import PIL
import pandas as pd
import pprint
import numpy as np
import PIL
from tqdm import tqdm
import pickle

import argparse

parser = argparse.ArgumentParser(description='Example Neural Network for OmniOpt-Hackathon')

parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train the model (default: 10)')
parser.add_argument('--visualization-steps', type=int, default=1000, help='Number of visualization steps (default: 1000)')
parser.add_argument('--visualization-lr', type=float, default=0.05, help='Visualization Learning Rate (default: 0.05)')
parser.add_argument('--dataset', type=str, default="full", help='Path of the training data directory. Possibilities are full for the full dataset and tiny_train_data for testing the network.')
parser.add_argument('--no-shuffle', dest='shuffle', action='store_false', help='Disable shuffling (default: True)')
parser.add_argument('--validation-split', type=validation_split_check, default=0.1, help='Validation split ratio (default: 0.1)')
parser.add_argument('--visualize', action='store_true', help='Disable visualization')
parser.add_argument('--debug', action='store_true', help='Show debug messages')
parser.add_argument('--start-firefox-if-ssh', action='store_true', help='Show visualization images in firefox if ssh -X is available (or local X-Server is installed)')
parser.add_argument('--pretrained', type=str, help='Path to a pretrained model. This skips training and allows to visualize any model.')

args = parser.parse_args()

import tensorflow as tf

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def debug (msg):
    if args.debug:
        print("DEBUG:", end = '')
        pprint.pprint(msg);

hyperparameters = {
    "epochs": args.epochs,
    "batch_size": 10,
    "shuffle": args.shuffle,
    "validation_split": args.validation_split,
    "seed": 42,
    "learning_rate": 0.001,

    "width": 220,
    "height": 220,

    "activation_function": 'relu',

    "number_of_convolutions": 4,
    "conv_filters": 16,
    "padding_name": "same", # other possible option: valid
    "max_pooling_size": 2,

    "number_of_dense_layers": 4,
    "dense_neurons":  32
}

debug(hyperparameters)

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_global_determinism(seed=42):
    set_seeds(seed=seed)

    #tf.keras.utils.set_random_seed(1)
    #tf.config.experimental.enable_op_determinism()

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

# Call the above function with seed value
set_global_determinism(hyperparameters["seed"])

# Get the path of the current Python script
script_path = os.path.abspath(__file__)

# Get the directory containing the script
dir_path = os.path.dirname(script_path)

# Change the current working directory to the script's directory
os.chdir(dir_path)

# Create a folder called 'runs' if it doesn't exist
if not os.path.exists('runs'):
    os.makedirs('runs', exist_ok=True)

# Find the number of existing run subfolders
run_folders = [folder for folder in os.listdir('runs') if os.path.isdir(os.path.join('runs', folder))]
num_runs = len(run_folders)

# Create a new run subfolder with the next available number
run_folder = f'run_{num_runs + 1}'
os.makedirs(os.path.join('runs', run_folder), exist_ok=True)

# Print out the name of the new folder
bold_white_print(f'Run folder: runs/{run_folder}')

# folders to load data from

dataset = "datasets/" + args.dataset + "/"
if not os.path.exists(dataset):
    print(f"{bcolors.FAIL}The folder {bcolors.WHITE}{dataset}{bcolors.ENDC}{bcolors.FAIL} does not exist{bcolors.ENDC}")
    sys.exit(2)
labels_path = f"{dataset}train/labels/"
if not os.path.exists(labels_path):
    print(f"{bcolors.FAIL}The folder {bcolors.WHITE}{labels_path}{bcolors.ENDC}{bcolors.FAIL} does not exist{bcolors.ENDC}")
    sys.exit(2)

# variables containing the training info after loading it

# Set the output file path
output_file = f'runs/{run_folder}/hyperparameters.json'

# Write the variables to the output file
with open(output_file, 'w') as f:
    json.dump(hyperparameters, f, indent=4)

bold_white_print(f"Printed hyperparameters to {bcolors.UNDERLINE}{output_file}")

labels_list = []
labels = []
images = []
boxes = []


# get all images and labels

for x in os.listdir(labels_path):
    img_path = str((labels_path + x)).replace("labels","images").replace("txt","jpg")
    label_path = str(labels_path + x)
    if os.path.exists(labels_path) and os.path.exists(img_path):
        labels.append(labels_path+x)
        images.append(img_path)

bold_white_print("Loaded " + str(len(labels)) + " labels and " + str(len(images)) + " images")


# read the file with the label ids

labels_txt = f"{dataset}labels.txt"
if not os.path.exists(labels_txt):
    print(f"{bcolors.FAIL}{labels_txt} does not exist{bcolors.ENDC}")
    sys.exit(3)

with open(dataset + 'labels.txt', 'r') as f:
    for line in f:
        res, n = re.subn('[" ",\n]', '', line)
        labels_list.append(res.split(":")[1])

bold_white_print(labels_list)

# print label/bounding box (from txt file) and image

# Get list of label files
label_files = [os.path.join(labels_path, f) for f in os.listdir(labels_path) if f.endswith('.txt')]

if not os.path.exists(f"{dataset}/x.pkl") or not not os.path.exists(f"{dataset}/y.pkl"):
    # Load the images and annotations

    X = []
    Y = []
    Y_labels_onehot = []
    bold_white_print("Loading images into memory")
    for idx, image_path in enumerate(tqdm(images)):
        try:
            img = PIL.Image.open(image_path).resize((hyperparameters["height"], hyperparameters["width"]))
            if str(img.mode) != "RGB":
                continue
            objects_string = ""

            with open(labels[idx], 'r') as f:
                # each line represents one object with label and bounding box
                for line in f:
                    # for convinience print the name of the phenomenon instead the label id
                    #objects_string += line.replace(line[0], labels_list[int(line[0])], 1) + "\n"
                    line_split = line.split(" ")
                    line_split[0] = labels_list[int(line_split[0])]
                    objects_string += " ".join(line_split)

            boxes = []
            for line in objects_string.split('\n'):
                if line:
                    class_name, x, y, w, h = line.split()
                    if class_name in labels_list:
                        # Convert the coordinates from relative to absolute
                        x1 = int((float(x) - float(w) / 2) * img.size[0])
                        y1 = int((float(y) - float(h) / 2) * img.size[1])
                        x2 = int((float(x) + float(w) / 2) * img.size[0])
                        y2 = int((float(y) + float(h) / 2) * img.size[1])
                        boxes.append((labels_list.index(class_name), x1, y1, x2, y2))
            # Append the image and labels to the X and Y tensors
            X.append(np.array(img))
            Y.append(np.array(boxes))
            one_hot_vector = tf.one_hot(labels_list.index(class_name), depth=len(labels_list))
            Y_labels_onehot.append(one_hot_vector)
        except:
            continue

if args.pretrained:
    # Load the pretrained model from the specified path
    model = tf.keras.models.load_model(args.pretrained)
elif args.epochs != 0:
    # Convert the lists to numpy arrays
    X = np.array(tf.convert_to_tensor(X), dtype=float)
    Y_labels_onehot = np.array(Y_labels_onehot)

    debug(f"Loading data from {dataset}/x.pkl")
    with open(f'{dataset}/x.pkl', 'wb') as f:
        pickle.dump(X, f)

    debug(f"Loading data from {dataset}/y.pkl")
    with open(f'{dataset}/y.pkl', 'wb') as f:
        pickle.dump(Y_labels_onehot, f)

    X = []
    Y_labels_onehot = []
    with open(dataset + 'x.pkl', 'rb') as f:
        X = pickle.load(f)

    with open(dataset + 'y.pkl', 'rb') as f:
        Y_labels_onehot = pickle.load(f)

    with open(dataset + 'x2.pkl', 'wb') as f:
        pickle.dump(X, f)

    # Normalize image data from 0-255 (integer) to 0-1 (float)

    X = X / 255.0


    # Create a simple neural network
    model = tf.keras.Sequential()

    for i in range(0, hyperparameters["number_of_convolutions"] + 1):
        try:
            model.add(tf.keras.layers.Conv2D(hyperparameters["conv_filters"], (3, 3), padding=hyperparameters["padding_name"], activation=hyperparameters["activation_function"], input_shape=(hyperparameters["width"], hyperparameters["height"], 3)))
        except:
            print(f"{bcolors.FAIL}There was an error adding more convolution layers. This may mean that the input width and height are too small or the number of convolutional layers is too high.{bcolors.ENDC}")
            sys.exit(1)

        try:
            model.add(tf.keras.layers.MaxPooling2D((hyperparameters["max_pooling_size"], hyperparameters["max_pooling_size"])))
        except:
            print(f"{bcolors.FAIL}There was an error adding a MaxPooling2D layer. This may mean that the input width and height are too small or the number of convolutional layers is too high.{bcolors.ENDC}")

            sys.exit(1)

    model.add(tf.keras.layers.Flatten())

    for i in range(0, hyperparameters["number_of_dense_layers"] + 1):
        model.add(tf.keras.layers.Dense(hyperparameters["dense_neurons"], activation=hyperparameters["activation_function"]))

    model.add(tf.keras.layers.Dense(len(labels_list), activation='softmax'))


    optimizer = tf.keras.optimizers.Adam(
            learning_rate=hyperparameters["learning_rate"]
            )

    # Compile the model with a suitable loss function

    model.compile(optimizer=optimizer, loss='CategoricalCrossentropy')

    model.summary()


    bold_white_print("X-Shape: " + str(X.shape))
    bold_white_print("Y-Shape: " + str(Y_labels_onehot.shape))


    # Shuffle X and Y data the same way, so that each epoch has a bunch of different iamges of each category

    indices = tf.range(start=0, limit=tf.shape(X)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)

    X = tf.gather(X, shuffled_indices)
    Y_labels_onehot = tf.gather(Y_labels_onehot, shuffled_indices)

    # Calculate the number of occurrences of each class
    num_occurrences = np.sum(Y_labels_onehot, axis=0)

    # Print the results
    for idx, label in enumerate(labels_list):
        print(f"{label}: {num_occurrences[idx]}")

    # Train the model
    history = model.fit(x=X, y=Y_labels_onehot, batch_size=hyperparameters["batch_size"], epochs=hyperparameters["epochs"], shuffle=hyperparameters["shuffle"], validation_split=hyperparameters["validation_split"])

    # Save the history object to a CSV file
    history_df = pd.DataFrame(history.history)
    history_csv = f"runs/{run_folder}/history.csv"
    history_df.to_csv(history_csv, index=False)
    bold_white_print(f"Saved history to {history_csv}")

    # Save model
    save_path = os.path.join("runs/" + str(run_folder), 'model.h5')
    bold_white_print(f"Saved model to {save_path}")
    model.save(save_path)

    # get all images and label
    path = labels_path.replace("train", "test")

    labels_files = []
    images_files = []
    if os.path.exists(path):
        for x in os.listdir(path):
            img_path = str((path + x)).replace("labels","images").replace("txt","jpg")
            label_path = str(path + x)
            if os.path.exists(label_path) and os.path.exists(img_path):
                labels_files.append(path+x)
                images_files.append(img_path)

        # Load the images and annotations
        X = []
        Y = []
        Y_labels_onehot = []
        predicted_Y_onehot = []

        bold_white_print("Evaluating performance on test dataset")
        for idx, image_path in enumerate(tqdm(images_files)):
            try:
                img = PIL.Image.open(image_path).resize((hyperparameters["height"], hyperparameters["width"]))
                if str(img.mode) != "RGB":
                    continue

                with open(labels_files[idx], 'r') as f:
                    for line in f:
                        line_split = line.split(" ")
                        line_split[0] = labels_list[int(line_split[0])]


                img_np = np.expand_dims(np.asarray(img), 0)
                prediction = model.predict(img_np, verbose=0)
                predicted_Y_onehot.append(prediction)

                # Append the image and labels to the X and Y tensors

                for line in objects_string.split('\n'):
                    if line:
                        class_name, x, y, w, h = line.split()

                        one_hot_vector = tf.one_hot(labels_list.index(class_name), depth=len(labels_list))
                        Y_labels_onehot.append(one_hot_vector)

                predicted_label = 0
                max_label = prediction[0][0]
                for i in range(0, len(prediction[0])):
                    if prediction[0][i] > max_label:
                        max_label = prediction[0][i]
                        predicted_label = i

                class_name = labels_list[predicted_label]

                subdir_path = f"runs/{run_folder}/test/{class_name}/"
                os.makedirs(subdir_path, exist_ok=True)

                img_file_name = images_files[idx]
                basename_file = os.path.basename(img_file_name)
                img_path = subdir_path + basename_file

                img.save(img_path)  # Save the image to the subdirectory

            except Exception as e:
                print("EXCEPTION:")
                print(bcolors.FAIL + str(e) + bcolors.ENDC)
                print("idx: ", str(idx))
                print("image_path: ", str(image_path))
                continue

        cce = tf.keras.losses.CategoricalCrossentropy()

        bold_white_print("CCE (the lower, the better):")
        cce = cce(np.expand_dims(Y_labels_onehot, 1), predicted_Y_onehot).numpy()
        bold_white_print(cce)

        print(f"RESULT: " + str(cce))
    else:
        print(f"{bcolors.WARNING}WARNING: path '%s' not found, cannot calculate CCE{bcolors.ENDC}" % path)


num_channels = 3

# Define the loss function
@tf.function
def compute_loss(input_image, filter_index):
    activation = model(input_image)
    return tf.gather(activation[0], filter_index)

# Define a function to generate the image that maximally activates the given filter
def generate_max_activation_image(filter_index, iterations=args.visualization_steps, step_size=args.visualization_lr):
    # Create a starting image with random noise
    input_image = tf.random.uniform((1, hyperparameters["width"], hyperparameters["height"], num_channels))
    input_image = tf.Variable(input_image)

    # Optimize the image to maximize activation of the given filter
    for iteration in tqdm(range(iterations)):
        with tf.GradientTape() as tape:
            loss = -compute_loss(input_image, filter_index)
        gradients = tape.gradient(loss, input_image)
        gradients /= tf.math.reduce_std(gradients) + 1e-8
        input_image.assign_add(gradients * step_size)
        input_image.assign(tf.clip_by_value(input_image, 0, 1))

    # Return the optimized image as a numpy array
    return input_image.numpy()[0]

stop_file = os.path.expanduser('~/stop_visualizing')

if args.visualize and not os.path.exists(stop_file):
    layer = len(model.layers) - 1 # last layer
    num_filters = model.layers[layer].output_shape[-1]
    vis_folder = f"runs/{run_folder}/visualizations"

    if not os.path.exists(vis_folder):
        os.makedirs(vis_folder, exist_ok=True)

    for neuron_idx in range(num_filters):
        label = labels_list[neuron_idx]
        if layer == len(model.layers) - 1:
            bold_white_print(f"Generating image for filter {label} (last layer, neuron: {neuron_idx})...")
        else:
            bold_white_print(f"Generating image for filter layer {layer}, neuron {neuron_idx}...")
        image = generate_max_activation_image(neuron_idx)
        img_dir = f"{vis_folder}/"
        image_path = f"{img_dir}{label}.png"
        os.makedirs(img_dir, exist_ok=True)
        bold_white_print(f"Written image to {image_path}")
        PIL.Image.fromarray((image * 255).astype(np.uint8)).save(image_path)
        if args.start_firefox_if_ssh and os.environ.get("DISPLAY"):
            print("SSH connection with X forwarding")
            firefox_cmd = "firefox {}".format(image_path)
            os.system(firefox_cmd)
