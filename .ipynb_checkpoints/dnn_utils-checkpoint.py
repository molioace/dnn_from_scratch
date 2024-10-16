import os
import h5py
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True) 

    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ

def sigmoid_backward(dA, cache):
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def load_images_from_folder(folder, image_size=(64, 64)):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            try:
                with Image.open(img_path) as img:
                    img = img.resize(image_size).convert('RGB')
                    img_array = np.array(img)
                    images.append(img_array)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return np.array(images)


def create_hdf5_file(cat_folder, dog_folder, output_file, image_size=(64, 64)):
    cat_images = load_images_from_folder(cat_folder, image_size)
    dog_images = load_images_from_folder(dog_folder, image_size)

    with h5py.File(output_file, 'w') as hdf5_file:
        hdf5_file.create_dataset('cats', data=cat_images, compression="gzip")
        hdf5_file.create_dataset('dogs', data=dog_images, compression="gzip")
    print(f"HDF5 file created at {output_file}")


def load_data_from_hdf5(hdf5_file):
    with h5py.File(hdf5_file, 'r') as file:
        cats = np.array(file['cats'])
        dogs = np.array(file['dogs'])
    return cats, dogs


def prepare_datasets(cats, dogs, test_size=0.2, random_state=42):
    y_cats = np.ones(len(cats))
    y_dogs = np.zeros(len(dogs))

    X = np.concatenate((cats, dogs), axis=0)
    y = np.concatenate((y_cats, y_dogs), axis=0)

    X_train_orig, X_test_orig, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train_orig, X_test_orig, y_train, y_test


def load_data(h5_train_path, h5_test_path):
    train_dataset = h5py.File(h5_train_path, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) 
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  

    test_dataset = h5py.File(h5_test_path, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) 

    classes = np.array(test_dataset["list_classes"][:])  

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes






