import os
from dotenv import load_dotenv
import tensorflow as tf
import boto3
import io
from botocore.exceptions import NoCredentialsError
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

# Load environment variables from .env file
load_dotenv(dotenv_path= "C:\Dhiraj\Aws-Resources\Train-And-Infer-Custom-Model-On-Sagemaker\.env")

# Define the S3 bucket name
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Initialize the S3 client
s3 = boto3.client('s3')

def upload_to_s3(file_obj, s3_key):
    """
    Uploads a file object to an S3 bucket.

    Args:
        file_obj (io.BytesIO): The file-like object to upload.
        s3_key (str): The S3 key (path) where the file will be stored.
    """
    try:
        s3.upload_fileobj(file_obj, BUCKET_NAME, s3_key)
        print(f"Upload Successful: {s3_key}")
    except NoCredentialsError:
        print("Credentials not available")

def upload_mnist_data(images, labels, data_type):
    """
    Prepares and uploads MNIST data (images and labels) to S3.

    Args:
        images (numpy.ndarray): Array of MNIST images.
        labels (numpy.ndarray): Array of MNIST labels.
        data_type (str): Type of dataset ('train' or 'test').
    """
    label_data = []

    def process_and_upload_image(args):
        """
        Processes an image and uploads it to S3.

        Args:
            args (tuple): Tuple containing index and (image, label).

        Returns:
            tuple: Filename and label of the uploaded image.
        """
        idx, (image, label) = args
        image_uint8 = image.astype('uint8')
        img = Image.fromarray(image_uint8)
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        filename = f"mnist_{idx}.png"
        s3_key = f"{data_type}/images/{filename}"
        
        upload_to_s3(img_buffer, s3_key)
        return filename, label

    # Process and upload images concurrently
    with ThreadPoolExecutor(max_workers=20) as executor:
        label_data = list(executor.map(process_and_upload_image, enumerate(zip(images, labels))))

    # Prepare and upload labels CSV
    csv_content = "filename,label\n" + "\n".join([f"{filename},{label}" for filename, label in label_data])
    csv_buffer = io.BytesIO(csv_content.encode())
    upload_to_s3(csv_buffer, f"{data_type}/labels.csv")
    
    print(f"Uploaded {data_type} data to S3.")

if __name__ == "__main__":
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    
    # Upload datasets to S3
    print("Uploading MNIST data to S3...")
    upload_mnist_data(train_images[:10000], train_labels[:10000], 'train') # Only Uploading few samples for speed reason
    upload_mnist_data(test_images[:1000], test_labels[:1000], 'test') # Only Uploading few samples for speed reason
    
    print("Data upload completed.")
