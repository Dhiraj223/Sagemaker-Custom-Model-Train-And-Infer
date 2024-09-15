import boto3
import json
import base64
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# Initialize a SageMaker runtime client
sagemaker_runtime = boto3.client('runtime.sagemaker')

# Function to load and encode an image
def load_and_encode_image(image_path):
    """
    Loads an image from the given path and encodes it in base64 format.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        str: Base64-encoded image data.
    """
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
    return base64_image

# Function to invoke the SageMaker endpoint
def invoke_endpoint(endpoint_name, image_path):
    """
    Invokes the SageMaker endpoint with the base64-encoded image.

    Parameters:
        endpoint_name (str): Name of the SageMaker endpoint.
        image_path (str): Path to the image file.

    Returns:
        dict: Prediction result from the SageMaker endpoint.
    """
    # Load and encode the image
    base64_image = load_and_encode_image(image_path)

    # Prepare the payload as JSON
    payload = json.dumps({"image": base64_image})

    # Invoke the SageMaker endpoint
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=payload
    )

    # Decode the response
    response_body = response['Body'].read().decode('utf-8')
    return json.loads(response_body)

# Function to display the image with the prediction label
def display_image_with_prediction(image_path, prediction):
    """
    Displays the image with the predicted label.

    Parameters:
        image_path (str): Path to the image file.
        prediction (str): The predicted label to display with the image.
    """
    # Load the image
    img = Image.open(image_path)

    # Create a plot
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')  # Hide axes

    # Display the prediction as the title of the plot
    plt.title(f"Prediction: {prediction}", fontsize=16, color='blue')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Get the image path from user input
    image_path = input("Enter the path to your image file: ")
    endpoint_name = 'mnist-endpoint'

    # Invoke the endpoint and get the prediction result
    result = invoke_endpoint(endpoint_name, image_path)

    # Extract the prediction 
    prediction = result.get('predicted_digit', 'Unknown')

    # Print the prediction result
    print("Prediction result:", result)

    # Display the image with the prediction
    display_image_with_prediction(image_path, prediction)
