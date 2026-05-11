import os
import argparse

import numpy as np
import keras
import matplotlib

# GitHub Actions sets CI=true; switch to non-interactive backend before pyplot loads
if os.getenv('CI'):
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Model
from keras.layers import (
    Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout,
    BatchNormalization, RandomFlip, RandomTranslation,
)
from keras.datasets import cifar10
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def load_and_preprocess_data(ci_mode=False):
    """Load CIFAR-10, normalize pixels to [0, 1] and one-hot encode labels.

    In CI mode we use a tiny slice of the data so the smoke test
    finishes in seconds rather than minutes.
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    if ci_mode:
        x_train, y_train = x_train[:500], y_train[:500]
        x_test, y_test = x_test[:100], y_test[:100]

    # Models generally train better with float data in [0, 1]
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # 3 -> [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def build_model():
    """Build a three-block CNN for CIFAR-10.

    The network has more than 10 layers, so we use he_uniform as the
    kernel initializer — it keeps activations well-scaled at init and
    prevents the vanishing gradient problem that plain random init causes
    in deeper relu stacks.

    Data augmentation layers (RandomFlip, RandomTranslation) are embedded
    in the model and are only active during training, so inference is unaffected.
    BatchNormalization after each conv block accelerates convergence and acts
    as an additional regularizer.
    """
    init = 'he_uniform'

    inputs = Input(shape=(32, 32, 3))

    # Data augmentation — only active when model.fit(...) is called
    x = RandomFlip('horizontal')(inputs)
    x = RandomTranslation(height_factor=0.1, width_factor=0.1)(x)

    # First conv block — low-level edges and colour cues
    x = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

    # Second conv block — mid-level object parts
    x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    # Third conv block — high-level semantic features
    x = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)

    # Classifier head
    x = Flatten()(x)
    x = Dense(512, activation='relu', kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )
    return model


def build_callbacks():
    """Return a list of training callbacks."""
    return [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1
        ),
    ]


def visualize_architecture(model, save_path='model_plot.png'):
    """Print the model summary and save the hand-drawn architecture diagram."""
    model.summary()
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'draw_architecture',
        os.path.join(os.path.dirname(__file__), 'draw_architecture.py'),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f'Architecture diagram saved to {save_path}')


def train_model(model, x_train, y_train, x_test, y_test, epochs=40, batch_size=64, callbacks=None):
    """Run the training loop and return the history object."""
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose='auto'
    )
    return history


def plot_training_history(history, save_path='history_plot.png'):
    """Plot the accuracy and loss curves from training history and save to file."""
    df = pd.DataFrame(history.history)
    ax = df.plot(title='Training History')
    ax.set_xlabel('Epoch')
    fig = ax.get_figure()
    fig.savefig(save_path)
    plt.close()
    print(f'Training history saved to {save_path}')


def evaluate_model(model, x_test, y_test):
    """Evaluate the model on test data and print loss and accuracy."""
    loss, accuracy = model.evaluate(x_test, y_test, verbose='auto')
    print(f'Test loss:     {loss:.4f}')
    print(f'Test accuracy: {accuracy:.4f}')
    return loss, accuracy


def save_model(model, path='cifar10_model.keras'):
    """Export the trained model weights and architecture to a .keras file."""
    model.save(path)
    print(f'Model saved to {path}')


def plot_confusion_matrix(model, x_test, y_test, labels, save_path='confusion_matrix.png'):
    """Run inference, build the confusion matrix, and save the heatmap."""
    predictions = np.argmax(model.predict(x_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(xticks_rotation='vertical', ax=ax, cmap='Blues')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f'Confusion matrix saved to {save_path}')


def show_misclassified(model, x_test, y_test, labels, n=5):
    """Save PNG files for n misclassified examples with their labels."""
    predictions = np.argmax(model.predict(x_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    incorrect = np.nonzero(predictions != y_true)[0]

    # Guard against fewer misclassifications than requested (e.g. in CI mode)
    n = min(n, len(incorrect))
    for i in range(n):
        idx = incorrect[i]
        plt.imshow(x_test[idx])
        plt.axis('off')
        plt.title(f'True: {labels[y_true[idx]]}  |  Predicted: {labels[predictions[idx]]}')
        plt.savefig(f'misclassified_{i + 1}.png', bbox_inches='tight')
        plt.close()
        print(f'Misclassified example {i + 1} saved')


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='CIFAR-10 CNN classifier')
    parser.add_argument(
        '--ci', action='store_true',
        help='Smoke-test mode: tiny dataset, 1 epoch — used by the CI pipeline'
    )
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=64)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    epochs = 1 if args.ci else args.epochs

    x_train, y_train, x_test, y_test = load_and_preprocess_data(ci_mode=args.ci)
    model = build_model()
    visualize_architecture(model)
    callbacks = [] if args.ci else build_callbacks()
    history = train_model(
        model, x_train, y_train, x_test, y_test,
        epochs=epochs, batch_size=args.batch_size, callbacks=callbacks
    )
    evaluate_model(model, x_test, y_test)
    save_model(model)
    plot_training_history(history)
    plot_confusion_matrix(model, x_test, y_test, LABELS)
    show_misclassified(model, x_test, y_test, LABELS)

#Test loss:     0.3451
#Test accuracy: 0.8840