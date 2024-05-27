import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# GPU configuration
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU is available and memory growth is set.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Running on CPU.")



# Download MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Define data shape
input_shape = x_train.shape[1:]

# Define autocoder
input_img = Input(shape=input_shape)
x = Flatten()(input_img)
encoded = Dense(64, activation='relu')(x)
decoded = Dense(np.prod(input_shape), activation='sigmoid')(encoded)
decoded = Reshape(input_shape)(decoded)

# Create autocoder
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train autocoder
print("Autocoder(enc/dec) training starts...")
autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

# Take encoder
encoder = Model(input_img, encoded)

# Take encoded representation
encoded_imgs_train = encoder.predict(x_train)
encoded_imgs_test = encoder.predict(x_test)

# Replace decoder with new deep neural networks and classifier
encoded_input = Input(shape=(64,))
dense = Dense(64, activation='relu')(encoded_input)
output = Dense(10, activation='softmax')(dense)

# Creating new classification model
classifier = Model(encoded_input, output)
classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Take 10% of data to train
num_labels = int(0.1 * len(y_train))
indices = np.random.choice(len(y_train), num_labels, replace=False)
x_train_small = encoded_imgs_train[indices]
y_train_small = y_train[indices]

# Train new model
print("Classifier(dec/ ) training starts...")
classifier.fit(x_train_small, y_train_small, epochs=50, batch_size=128, validation_data=(encoded_imgs_test, y_test))

# Evaluation
score = classifier.evaluate(encoded_imgs_test, y_test)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# Alternative: encoded representation clustering
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(encoded_imgs_train)

# Centroids
centroids = kmeans.cluster_centers_

# Label other data
def assign_labels(encoded_data, centroids):
    labels = []
    for data in encoded_data:
        distances = np.linalg.norm(centroids - data, axis=1)
        labels.append(np.argmin(distances))
    return np.array(labels)

test_labels = assign_labels(encoded_imgs_test, centroids)

# Classificator evaluation
print(f'Clustering Accuracy: {accuracy_score(y_test, test_labels)}')

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reduce dimensions for visualization
pca = PCA(n_components=2)
encoded_imgs_train_pca = pca.fit_transform(encoded_imgs_train)
centroids_pca = pca.transform(centroids)

# Visualize clusters
plt.figure(figsize=(8, 6))
plt.scatter(encoded_imgs_train_pca[:, 0], encoded_imgs_train_pca[:, 1], c=clusters, cmap='viridis', s=20, alpha=0.5)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='x', s=100, label='Centroids')
plt.title('Clusters of Encoded Representations')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.legend()
plt.show()
