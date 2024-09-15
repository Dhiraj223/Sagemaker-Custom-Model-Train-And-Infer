import boto3
from botocore.exceptions import ClientError

def delete_sagemaker_endpoint(endpoint_name):
    """Delete a SageMaker endpoint by name."""
    sagemaker_client = boto3.client('sagemaker')
    try:
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        print(f"SageMaker endpoint '{endpoint_name}' has been deleted.")
    except ClientError as e:
        print(f"Error deleting SageMaker endpoint '{endpoint_name}': {e}")

def delete_ecr_repository(repo_name):
    """Delete an ECR repository by name along with all images."""
    ecr_client = boto3.client('ecr')
    try:
        # Delete all images in the repository before deleting the repo
        images = ecr_client.list_images(repositoryName=repo_name)
        if 'imageIds' in images and images['imageIds']:
            ecr_client.batch_delete_image(repositoryName=repo_name, imageIds=images['imageIds'])
            print(f"Deleted images from ECR repository '{repo_name}'.")

        ecr_client.delete_repository(repositoryName=repo_name, force=True)
        print(f"ECR repository '{repo_name}' has been deleted.")
    except ClientError as e:
        print(f"Error deleting ECR repository '{repo_name}': {e}")

def delete_s3_bucket(bucket_name):
    """Delete an S3 bucket and all its contents."""
    s3_client = boto3.client('s3')
    try:
        # Delete all objects in the bucket before deleting the bucket
        bucket = boto3.resource('s3').Bucket(bucket_name)
        bucket.objects.delete()  # Deletes all objects in the bucket
        print(f"Deleted all objects in S3 bucket '{bucket_name}'.")

        # Now delete the bucket
        s3_client.delete_bucket(Bucket=bucket_name)
        print(f"S3 bucket '{bucket_name}' has been deleted.")
    except ClientError as e:
        print(f"Error deleting S3 bucket '{bucket_name}': {e}")

def main():
    endpoint_name = input('your-sagemaker-endpoint-name : ')
    repo_name = input("your-ecr-repository-name : ")
    bucket_name = input("your-s3-bucket-name : ")

    # Cleanup SageMaker endpoint
    # delete_sagemaker_endpoint(endpoint_name)

    # Cleanup ECR repository
    # delete_ecr_repository(repo_name)

    # Cleanup S3 bucket
    # delete_s3_bucket(bucket_name)

if __name__ == "__main__":
    main()
