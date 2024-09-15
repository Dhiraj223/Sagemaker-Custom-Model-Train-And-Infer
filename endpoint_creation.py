import os
import sagemaker
from dotenv import load_dotenv
from sagemaker.model import Model

# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
image_uri = os.getenv("ECR_IMAGE_URI")
role = os.getenv("SAGEMAKER_EXECUTION_ROLE")
model_data = os.getenv("MODEL_ARTIFACT_URI")
endpoint_name = "mnist-endpoint"

# Create a SageMaker Model instance
model = Model(
    image_uri=image_uri,              # ECR image URI for the model container
    role=role,                        # IAM role for SageMaker
    entry_point='MnistExample\inference.py',       # Script for handling inference
    model_data=model_data,            # S3 location of the model weights
    sagemaker_session=sagemaker.Session()  # SageMaker session
)

# Deploy the model to a SageMaker endpoint
predictor = model.deploy(
    instance_type='ml.m5.large',      # Instance type for hosting the model
    initial_instance_count=1,      # Number of instances for the endpoint
    endpoint_name=endpoint_name    # Name of the deployed endpoint
)

# Output the deployed endpoint name
print(f"Model deployed to endpoint: {endpoint_name}")
