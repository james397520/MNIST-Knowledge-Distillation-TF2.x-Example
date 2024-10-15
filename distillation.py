import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import CategoricalAccuracy

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Load MNIST dataset
def load_mnist_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Extract the i-th image and reshape
    image = X_train[0].reshape(28, 28)
    
    # Set save path and filename
    image_path = os.path.join(f'mnist_image_{0}.png')
    
    # Plot image and remove axes
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    # Save image
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return X_train, X_test, y_train, y_test

# Plot and save every 10 epochs
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
    plot_path = f'./distillation_training_plots/epoch_{epoch + 1}.png'
    os.makedirs('./distillation_training_plots', exist_ok=True)  # Ensure directory exists
    plt.savefig(plot_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, epoch):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(10)])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix at Epoch {epoch + 1}')
    
    # Adjust layout to ensure labels are not cut off
    plt.tight_layout()
    
    plot_path = f'./distillation_confusion_matrices/epoch_{epoch + 1}.png'
    os.makedirs('./distillation_confusion_matrices', exist_ok=True)  # Ensure directory exists
    plt.savefig(plot_path)
    plt.close()

def create_dataset(images, labels, batch_size, img_height, img_width):
    # Create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(lambda x, y: (tf.image.resize(x, [img_height, img_width]), y),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def distillation_loss(y_true, y_pred, teacher_pred, temperature=3.0, alpha=0.5):
    y_pred_soft = tf.nn.softmax(y_pred / temperature)
    teacher_pred_soft = tf.nn.softmax(teacher_pred / temperature)
    soft_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(teacher_pred_soft, y_pred_soft))
    hard_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))
    return alpha * soft_loss + (1 - alpha) * hard_loss

# Create student model
def create_student_model(img_height=28, img_width=28):
    inputs = Input(shape=(img_height, img_width, 1))
    x = Conv2D(4, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_student_model(teacher_model, student_model, train_dataset, val_dataset, checkpoint_filepath, epochs=50):
    optimizer = Adam(learning_rate=0.001)
    train_acc_metric = CategoricalAccuracy()
    val_acc_metric = CategoricalAccuracy()

    # Initialize history lists
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    best_val_acc = 0.0  # Initialize best_val_acc

    @tf.function
    def train_step(X_batch, y_batch):
        with tf.GradientTape() as tape:
            teacher_pred = teacher_model(X_batch, training=False)
            student_pred = student_model(X_batch, training=True)

            loss_value = distillation_loss(y_batch, student_pred, teacher_pred)

        grads = tape.gradient(loss_value, student_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, student_model.trainable_weights))
        train_acc_metric.update_state(y_batch, student_pred)
        return loss_value

    @tf.function
    def val_step(X_batch, y_batch):
        val_pred = student_model(X_batch, training=False)
        val_loss = distillation_loss(y_batch, val_pred, teacher_model(X_batch, training=False))
        val_acc_metric.update_state(y_batch, val_pred)
        return val_loss

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_loss = 0.0
        num_batches = 0

        for X_batch, y_batch in train_dataset:
            loss_value = train_step(X_batch, y_batch)
            epoch_loss += loss_value.numpy()
            num_batches += 1

        train_loss = epoch_loss / num_batches
        train_acc = train_acc_metric.result().numpy()
        train_acc_metric.reset_states()

        # Validation
        val_epoch_loss = 0.0
        val_num_batches = 0
        for X_batch, y_batch in val_dataset:
            val_loss = val_step(X_batch, y_batch).numpy()
            val_epoch_loss += val_loss
            val_num_batches += 1

        val_loss = val_epoch_loss / val_num_batches
        val_acc = val_acc_metric.result().numpy()
        val_acc_metric.reset_states()

        # Update history
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Model checkpoint
        if val_acc > best_val_acc:
            print("Saving best model weights...")
            student_model.save_weights(checkpoint_filepath)
            best_val_acc = val_acc

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        if (epoch + 1) % 10 == 0:
            plot_loss_and_accuracy(train_loss_history, val_loss_history, train_acc_history, val_acc_history, epoch)
            # Calculate confusion matrix
            y_pred_classes = []
            y_true_classes = []
            for X_batch, y_batch in val_dataset:
                val_logits = student_model(X_batch, training=False)
                y_pred_classes.extend(np.argmax(val_logits, axis=1))
                y_true_classes.extend(np.argmax(y_batch, axis=1))
            plot_confusion_matrix(y_true_classes, y_pred_classes, epoch)
            
            # Visualize results
            # Get a batch of predictions
            X_batch, _ = next(iter(val_dataset))
            teacher_pred = teacher_model(X_batch, training=False)
            student_pred = student_model(X_batch, training=False)
            visualize_predictions(X_batch, teacher_pred, student_pred)  # Pass teacher_pred and student_pred

def visualize_predictions(images, teacher_predictions, student_predictions):
    
    num_images = min(5, len(images))  # Display up to 5 images
    for i in range(num_images):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        image = images[i].numpy().astype("float32") * 255.0
        plt.imshow(image.astype("uint8"), cmap='gray')  # Display in grayscale
        plt.title("Input Image")

        plt.subplot(1, 3, 2)
        plt.bar(range(len(teacher_predictions[i])), teacher_predictions[i], width=0.5)  # Adjust bar width
        plt.ylim(0, 1)  # Set y-axis range
        plt.title("Teacher Prediction")

        plt.subplot(1, 3, 3)
        plt.bar(range(len(student_predictions[i])), student_predictions[i], width=0.5)  # Adjust bar width
        plt.ylim(0, 1)  # Set y-axis range
        plt.title("Student Prediction")

        # Save figure
        plot_path = f'./distillation_visualizations/image_{i + 1}.png'
        os.makedirs('./distillation_visualizations', exist_ok=True)  # Ensure directory exists
        plt.savefig(plot_path)
        plt.close()

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
    plt.savefig('./distillation_confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    # Load pre-trained teacher model
    teacher_model = load_model('./best_teacher_weights.h5')

    # Create student model
    IMG_HEIGHT = 28
    IMG_WIDTH = 28
    epochs = 10
    batch_size = 512
    checkpoint_filepath = './best_distillation_weights.h5'

    # Use MNIST dataset
    X_train, X_test, y_train, y_test = load_mnist_data()  # Use load_mnist_data function

    student_model = create_student_model(IMG_HEIGHT, IMG_WIDTH)

    # Compile student model
    student_model.compile(optimizer=Adam(learning_rate=0.001), 
                          loss='categorical_crossentropy', 
                          metrics=['accuracy'])

    # Create training and testing datasets
    train_dataset = create_dataset(X_train, y_train, batch_size, IMG_HEIGHT, IMG_WIDTH)
    test_dataset = create_dataset(X_test, y_test, batch_size, IMG_HEIGHT, IMG_WIDTH)

    # Train student model using distillation
    train_student_model(teacher_model, student_model, train_dataset, test_dataset, checkpoint_filepath, epochs)
    # Evaluate model
    test_model(student_model, X_test, y_test, checkpoint_filepath)