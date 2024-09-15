import os
import io
import json
import boto3
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from keras import layers, models

def load_mnist_data(data_dir, limit=1000):
    """
    Load MNIST dataset from S3 bucket.

    Args:
        data_dir (str): S3 URI directory containing the images and labels.
        limit (int): Maximum number of images to load.

    Returns:
        tuple: (images, labels) where images is a numpy array of shape (num_images, 28, 28),
               and labels is a numpy array of shape (num_images,).
    """
    if data_dir is None:
        raise ValueError("Data directory is not provided or is None.")
    
    s3 = boto3.client('s3')
    images = []
    labels = []
    
    # Parse S3 URI
    bucket_name = data_dir.split('/')[2]
    prefix = data_dir.split('/')[3]

    # Load labels
    response = s3.get_object(Bucket=bucket_name, Key=f"{prefix}/labels.csv")
    labels_df = pd.read_csv(io.BytesIO(response['Body'].read()))
    
    # Load images (limiting to the first 'limit' entries)
    for _, row in labels_df.iterrows():
        print(len(images))
        if len(images) >= limit:
            break
        image_key = f"{prefix}/images/{row['filename']}"
        response = s3.get_object(Bucket=bucket_name, Key=image_key)
        image = Image.open(io.BytesIO(response['Body'].read())).convert('L')
        images.append(np.array(image))
        labels.append(row['label'])
    
    return np.array(images), np.array(labels)

def main():
    """
    Main function to set up argument parsing, load data, define and train a CNN model,
    evaluate the model, and save it.
    """
    parser = argparse.ArgumentParser()

    # SageMaker specific arguments
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS', '[]')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST', ''))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', ''))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST', ''))
    parser.add_argument('--num-gpus', type=int, default=os.environ.get('SM_NUM_GPUS', 0))

    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--trainlimit', type=int, default=3000)
    parser.add_argument('--testlimit', type=int, default=500)

    args = parser.parse_args()

    train_data_dir = args.train
    test_data_dir = args.test
    print(f"Training data directory: {train_data_dir}")
    print(f"Testing data directory: {test_data_dir}")

    train_limit = args.trainlimit
    test_limit  = args.testlimit
    print(train_limit, test_limit)

    # Load training and testing data
    train_images, train_labels = load_mnist_data(train_data_dir, limit=train_limit)
    test_images, test_labels = load_mnist_data(test_data_dir, limit=test_limit)

    # Normalize images
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255

    # Reshape images for CNN input
    train_images = train_images.reshape((-1, 28, 28, 1))
    test_images = test_images.reshape((-1, 28, 28, 1))

    # Convert labels to categorical
    train_labels = tf.keras.utils.to_categorical(train_labels, 10)
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)

    # Define the CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(train_images, train_labels,
              epochs=args.epochs,
              batch_size=args.batch_size,
              validation_split=0.1,
              verbose=1)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'\nTest Loss: {test_loss}\nTest accuracy: {test_acc}')

    # Save the model
    model_path = os.path.join(args.model_dir, 'model.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    main()
