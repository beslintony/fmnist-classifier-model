import imageio.v3 as iio
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import tensorflow as tf ;

# Read the numpy files of FashionMNIST
def load_numpy_files():
    X = np.load("fashionMNIST_X.npy").astype(float)
    T = np.load("fashionMNIST_T.npy").astype(float)

    return X, T

# Plot some images against its labels
def plot_images(r, c, X):
    # Create a figure with a grid of subplots
    f, ax = plt.subplots(r, c, figsize=(r*c, r*r))
    ax = ax.ravel()

    # Get a random sample of 15 images and labels from the dataset
    sample_indices = np.random.randint(0, len(X), 15)
    sample_images = X[sample_indices]
    sample_labels = np.argmax(T[sample_indices], axis=1)

    # Plot the images and labels in the subplots
    for i, (image, label) in enumerate(zip(sample_images, sample_labels)):
        ax[i].imshow(image)
        ax[i].set_title(f'Label: {label}')
        ax[i].axis('off')

    # Show the figure
    plt.show()

# Fix brightness in the center of the picture
def fix_images(x):
    h = int(x.shape[1] / 2)
    w = int(x.shape[2] / 2)

    # Fill the center with 0
    x[:,h,w] = 0

    return x

# Plot the labels count of the dataset
def plot_label_counts(T):
    # Count the number of occurrences of each label
    label_counts = np.sum(T, axis=0)

    f,ax=plt.subplots(1,1)

    # Plot the distribution of the labels
    ax.bar(np.arange(10), label_counts)
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.show()

def get_largest_label(T):
    # Find the label with the largest occurrence in each row
    labels = np.argmax(T, axis=1)
    counts = np.bincount(labels)

    # Find the index of the label with the maximum count
    max_label = np.argmax(counts)
    max_count = counts[max_label]

    return max_count

# Oversample the dataset to fix the dataset imbalances
def get_oversample(X, T):
    # Create an empty list to store the oversampled samples
    oversampled_X = []
    oversampled_T = []


    # Determine the desired number of samples per class
    # target_count = 13198
    samples = get_largest_label(T)
    print("Largest occurence of label is ", samples)
    target_count = samples


    # Iterate over each class
    for i in range(10):
        # Get the indices of the samples for this class
        class_indices = np.where(np.argmax(T, axis=1) == i)[0]
        # Calculate the number of samples to oversample
        oversample_count = target_count - len(class_indices)

        print("Class ", i ," indices shape: ", class_indices.shape)
        # Randomly select samples to oversample
        oversample_indices = np.random.choice(class_indices, oversample_count, replace=True)
        # Add the oversampled samples to the list
        oversampled_X.append(X[oversample_indices])
        oversampled_T.append(T[oversample_indices])

    # Concatenate the oversampled samples with the original samples
    oversample_X = np.concatenate((X, np.concatenate(oversampled_X)), axis=0)
    oversample_T = np.concatenate((T, np.concatenate(oversampled_T)), axis=0)

    return oversample_X, oversample_T

# Split the dataset into train and test
def split_dataset(X, T, size):
    # Create an array of indices for the data
    indices = np.arange(X.shape[0])

    # Shuffle the indices
    np.random.shuffle(indices)

    # Compute the number of samples for the test set
    test_size = int(X.shape[0] * size)

    # Split the indices into training and testing sets
    X_train_indices = indices[test_size:]
    X_test_indices = indices[:test_size]

    # Use the indices to split the data into training and testing sets
    X_train = X[X_train_indices]
    X_test = X[X_test_indices]
    T_train = T[X_train_indices]
    T_test = T[X_test_indices]

    return X_train, X_test, T_train, T_test

# Create confusion matrix
def create_confusion_matrix(predicted_labels, actual_labels):
    # Initialize an empty confusion matrix
    confusion_matrix = np.zeros((10, 10))
    
    # Iterate over the predicted labels and actual labels
    for predicted, actual in zip(predicted_labels, actual_labels):
        # Increment the count for the corresponding predicted and actual labels
        confusion_matrix[predicted][actual] += 1
    return confusion_matrix

# Plot confusion matrix
def plot_CM(confusion_matrix):
    f,ax=plt.subplots(1,1)
    # Plot the confusion matrix
    ax.matshow(confusion_matrix)

    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.title('Confusion Matrix')

    plt.colorbar()
    plt.show()

# Train the model
def train_model(X_train, X_test, T_train, T_test):
    # reshape to NHWC format for initial convLayers!
    X_train = X_train.reshape(-1,28,28,1)
    X_test = X_test.reshape(-1,28,28,1)

    print(tf.shape(X_train), tf.shape(X_test), tf.shape(T_train), tf.shape(T_test))

    # This is a complete DNN, should get ~98% on MNIST test set
    # comment out the lines with !! to get a linear classifier
    model = tf.keras.Sequential() ;
    model.add(tf.keras.layers.Conv2D(64, 3)) ; # 26,26,64
    model.add(tf.keras.layers.MaxPool2D()) ; # 13,13,64
    model.add(tf.keras.layers.ReLU()) ;     # 13,13,64
    model.add(tf.keras.layers.Conv2D(64, 4)) ; # 10,10,64
    model.add(tf.keras.layers.MaxPool2D()) ; # 5,5,64
    model.add(tf.keras.layers.ReLU()) ;     # 5,5,64
    model.add(tf.keras.layers.Reshape(target_shape=(25*64,))) ;     # 25*64        
    model.add(tf.keras.layers.Dense(100)) ; # 
    model.add(tf.keras.layers.ReLU()) ;     # !!
    model.add(tf.keras.layers.Dense(100)) ; # 
    model.add(tf.keras.layers.ReLU()) ;     # !!
    model.add(tf.keras.layers.Dense(10)) ;
    model.add(tf.keras.layers.Softmax()) ;

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), 
        loss = tf.keras.losses.CategoricalCrossentropy(), 
        metrics = [tf.keras.metrics.CategoricalAccuracy()]) ;

    print(X_train.shape, X_test.shape, T_train.shape, T_test.shape)

    model.fit(tf.cast(X_train, tf.float32),tf.cast(T_train, tf.float32),epochs=3, batch_size=100) ;
    Y = model.predict(tf.cast(X_test, tf.float32))

    actual_labels = np.argmax(T_test, axis=1) #Get the targets in the form of 1d vector that represents the columns of the Confusion Matrix
    predicted_labels = np.argmax(Y,axis=1) #Get the predicted values in the form of 1d vector that represents the rows of the Confusion Matrix

    confusion_matrix = create_confusion_matrix(predicted_labels, actual_labels)
    print("Confusion Matrix")
    print(confusion_matrix)
    plot_CM(confusion_matrix)

if __name__ == "__main__":
    X, T = load_numpy_files()

    # Prints the shape of the loaded data and label set
    print(X.shape)
    print(T.shape)

    plot_images(3,5, X)
    
    fix_images(X)
    plot_images(3,5, X)

    plot_label_counts(T)

    oversample_X, oversample_T = get_oversample(X, T)
    plot_label_counts(oversample_T)

    X_train, X_test, T_train, T_test = split_dataset(oversample_X, oversample_T, 0.2)
    print(X_train.shape, X_test.shape, T_train.shape, T_test.shape)

    train_model(X_train, X_test, T_train, T_test)
