import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.layers import Conv2D, MaxPooling2D, \
                                    Dense, Dropout, Input, Softmax, Reshape, Add, \
                                    GlobalAveragePooling2D, Flatten

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load MNIST dataset
def load_mnist_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return X_train, X_test, y_train, y_test

# Preprocess image
def preprocess_image(image, img_height, img_width):
    image = tf.image.resize(image, [img_height, img_width])
    image = image / 255.0  # Normalize to [0, 1]
    return image

def load_and_preprocess_from_path_label(image, label, img_height, img_width):
    image = preprocess_image(image, img_height, img_width)
    return image, label

def create_dataset(images, labels, batch_size, img_height, img_width):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(lambda x, y: load_and_preprocess_from_path_label(x, y, img_height, img_width),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def create_student_model(img_height=28, img_width=28):
    inputs = Input(shape=(img_height, img_width, 1))  # Modify input shape
    
    x = Conv2D(4, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Flatten()(x)
    
    # Fully connected layer + Dropout
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    outputs = Dense(10, activation='softmax')(x)  # Modify output layer's units
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Calculate model's FLOPs and parameters
def calculate_flops_and_params(model):
    # Calculate parameters
    model.summary()
    total_params = model.count_params()
    param_size_kb = total_params * 4 / 1024  # Each parameter is 4 bytes, convert to KB
    print(f"Total Parameters: {total_params}")
    print(f"Parameter Size: {param_size_kb:.2f} KB")
    
    # Calculate FLOPs
    concrete_func = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete_func.get_concrete_function(tf.TensorSpec([1, IMG_HEIGHT, IMG_WIDTH, 1], model.inputs[0].dtype))
    
    # Use TensorFlow's convert_variables_to_constants_v2 to freeze the graph
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    graph_def = frozen_func.graph.as_graph_def()

    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
        print(f"FLOPs: {flops.total_float_ops}")

# Plot loss and accuracy every 10 epochs and save
def plot_loss_and_accuracy(train_loss, val_loss, train_acc, val_acc, epoch):
    plt.figure(figsize=(12, 4))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend()
    plt.title("Loss")

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracy")

    # Save plot
    plot_path = f'./student_training_plots/epoch_{epoch + 1}.png'
    os.makedirs('./student_training_plots', exist_ok=True)  # Ensure directory exists
    plt.savefig(plot_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, epoch):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(10)])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix at Epoch {epoch + 1}')
    
    # Adjust layout to ensure labels are not cut off
    plt.tight_layout()
    
    plot_path = f'./student_confusion_matrices/epoch_{epoch + 1}.png'
    os.makedirs('./student_confusion_matrices', exist_ok=True)  # Ensure directory exists
    plt.savefig(plot_path)
    plt.close()

@tf.function
def train_step(X_batch, y_batch, model, loss_fn, optimizer, train_acc_metric):
    with tf.GradientTape() as tape:
        logits = model(X_batch, training=True)
        loss_value = loss_fn(y_batch, logits)
    
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y_batch, logits)
    return loss_value

@tf.function
def val_step(X_batch, y_batch, model, loss_fn, val_acc_metric):
    val_logits = model(X_batch, training=False)
    val_loss = loss_fn(y_batch, val_logits)
    val_acc_metric.update_state(y_batch, val_logits)
    return val_loss

def train_model(model, train_dataset, test_dataset, epochs, optimizer, loss_fn, train_acc_metric, val_acc_metric, checkpoint_filepath):
    best_val_acc = 0.0
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_loss = 0.0
        num_batches = 0

        # Training loop
        for X_batch, y_batch in tqdm(train_dataset, desc="Training", leave=False):
            
            loss_value = train_step(X_batch, y_batch, model, loss_fn, optimizer, train_acc_metric)
            epoch_loss += loss_value.numpy()
            num_batches += 1

        train_loss = epoch_loss / num_batches
        train_acc = train_acc_metric.result().numpy()
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        train_acc_metric.reset_states()

        # Validation
        for X_batch, y_batch in tqdm(test_dataset, desc="Validation", leave=False):
            val_loss = val_step(X_batch, y_batch, model, loss_fn, val_acc_metric).numpy()

        val_acc = val_acc_metric.result().numpy()
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        val_acc_metric.reset_states()

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        # Model checkpoint
        if val_acc > best_val_acc:
            print("Saving best model weights...")
            model.save(checkpoint_filepath)
            best_val_acc = val_acc

        # Save plot and confusion matrix every 10 epochs
        if (epoch + 1) % 10 == 0:
            plot_loss_and_accuracy(train_loss_history, val_loss_history, train_acc_history, val_acc_history, epoch)
            evaluate_and_plot_confusion_matrix(model, test_dataset, epoch)

def evaluate_and_plot_confusion_matrix(model, test_dataset, epoch):
    y_pred_classes = []
    y_true_classes = []
    for X_batch, y_batch in test_dataset:
        val_logits = model(X_batch, training=False)
        y_pred_classes.extend(np.argmax(val_logits, axis=1))
        y_true_classes.extend(np.argmax(y_batch, axis=1))
    plot_confusion_matrix(y_true_classes, y_pred_classes, epoch)

def test_model(model, X_test, y_test, checkpoint_filepath):
    model.load_weights(checkpoint_filepath)
    results = model.evaluate(X_test, y_test, verbose=2)
    test_loss, test_acc = results[0], results[1]
    print(f'\nTest accuracy: {test_acc}')

    # Calculate confusion matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true_classes, y_pred_classes)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(10)])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig('./student_test_confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    # Set parameters
    IMG_HEIGHT = 28
    IMG_WIDTH = 28
    
    epochs = 100
    batch_size = 512
    checkpoint_filepath = './best_student_weights.h5'
    initial_learning_rate = 0.001
    train = True



    # Load MNIST data
    X_train, X_test, y_train, y_test = load_mnist_data()
    
    # Create CNN model
    model = create_student_model()

    # Calculate model's FLOPs and parameters
    calculate_flops_and_params(model)
    
    if train:
        
        
        # Create training and testing datasets
        train_dataset = create_dataset(X_train, y_train, batch_size, 28, 28)
        test_dataset = create_dataset(X_test, y_test, batch_size, 28, 28)
        
        # Set learning rate decay
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True)

        optimizer = Adam(learning_rate=lr_schedule)

        # Custom training loop
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        train_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        val_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=initial_learning_rate), 
            loss=loss_fn, 
            metrics=[
                train_acc_metric,
                val_acc_metric
            ]
        )
        # Train model
        train_model(model, train_dataset, test_dataset, epochs, optimizer, loss_fn, train_acc_metric, val_acc_metric, checkpoint_filepath)

        # Evaluate model
        test_model(model, X_test, y_test, checkpoint_filepath)