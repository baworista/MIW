import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Filter classes (animals and vehicles)
animal_classes = [2, 3, 4, 5, 6, 7]  # Birds, cats, deer, dogs, frogs, horses
vehicle_classes = [0, 1, 8, 9]  # Airplanes, automobiles, ships, trucks

# Relabel classes: 0 for animals, 1 for vehicles
y_train = np.where(np.isin(y_train, animal_classes), 0, 1)
y_test = np.where(np.isin(y_test, animal_classes), 0, 1)

# Split training data into 3:7 ratio
num_train_samples = int(0.3 * x_train.shape[0])
x_train, x_valid = x_train[:num_train_samples], x_train[num_train_samples:]
y_train, y_valid = y_train[:num_train_samples], y_train[num_train_samples:]

print("Training set shape:", x_train.shape, y_train.shape)
print("Validation set shape:", x_valid.shape, y_valid.shape)
print("Test set shape:", x_test.shape, y_test.shape)


def create_cnn_model(num_conv_layers):
    '''
    Convolutional layers has filters and find different patterns in images,
    poling layer is the layer which reduces amount of filters,
    than the flatter layer is making from two-dimensional array one dimensional array
    and then dense is just normal layers of neural network
    which process data and the output is one neuron that classifies.
    '''
    model = Sequential()
    input_shape = x_train.shape[1:]

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))

    for i in range(num_conv_layers - 1):
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Single neuron with sigmoid activation

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_and_evaluate_model(num_conv_layers):
    model = create_cnn_model(num_conv_layers)
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_valid, y_valid), batch_size=64)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy with {num_conv_layers} convolutional layer(s): {test_acc:.4f}")
    return history, test_acc


history_1, acc_1 = train_and_evaluate_model(1)
history_2, acc_2 = train_and_evaluate_model(2)
history_3, acc_3 = train_and_evaluate_model(3)

def plot_training_history(history, num_conv_layers):
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Training and Validation Accuracy with {num_conv_layers} Conv Layer(s)')
    plt.legend()
    plt.show()

plot_training_history(history_1, 1)
plot_training_history(history_2, 2)
plot_training_history(history_3, 3)

print(f"Accuracy with 1 conv layer: {acc_1:.4f}")
print(f"Accuracy with 2 conv layers: {acc_2:.4f}")
print(f"Accuracy with 3 conv layers: {acc_3:.4f}")

# Choose the model with the highest accuracy
optimal_layers = np.argmax([acc_1, acc_2, acc_3]) + 1
print(f"The optimal number of convolutional layers is: {optimal_layers}")
