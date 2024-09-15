import os
from dotenv import load_dotenv
from sagemaker.estimator import Estimator

# Load environment variables from a .env file
load_dotenv()

# Retrieve environment variables
image_uri = os.getenv("ECR_IMAGE_URI")
role = os.getenv("SAGEMAKER_EXECUTION_ROLE")
train_data = os.getenv("TRAIN_S3_URI")
test_data = os.getenv("TEST_S3_URI")

# Set up the SageMaker Estimator for the training job
estimator = Estimator(
    image_uri=image_uri,
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    hyperparameters={
        'epochs': 25,
        'batch-size': 64,
        'learning-rate': 0.001,
        'trainlimit' : 5000,
        #'testlimit' : 500
    },
    environment={
        "SM_CHANNEL_TRAIN": train_data,
        "SM_CHANNEL_TEST": test_data
    }
)

# Start the training job
estimator.fit({'train': train_data, 'test': test_data})
