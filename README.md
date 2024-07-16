![download (1)](https://github.com/user-attachments/assets/28629d2e-57dc-4e45-898c-ad8b81875f6a)
# Hugging-face-on-AWS-SAGE-MAKER
Step by Step guide to deploy hugging face on AWS Sagemaker

## Deploying Hugging Face Models Using AWS SageMaker: A Step-by-Step Guide

AWS SageMaker is a powerful tool for managing the entire lifecycle of machine learning projects. This guide will walk you through deploying Hugging Face models on AWS SageMaker.

## Step 1: Set Up SageMaker Studio

1. **Access AWS SageMaker**: Log in to your AWS Management Console and search for AWS SageMaker. Click on "Getting Started" to initiate the setup.
2. **Create a Domain**: Navigate to the "Domains" section and create a new domain. Choose the "Setup for single user" option for testing purposes.
3. **Configure IAM Role**: SageMaker will automatically create an IAM role with a full access policy, ensuring you have the necessary permissions. Ensure public internet access and standard encryption are selected.
4. **Initialize SageMaker Studio**: Once the domain is created, launch SageMaker Studio, which provides an integrated development environment for machine learning.
<img width="1158" alt="1" src="https://github.com/user-attachments/assets/3b62c090-36a6-44e3-8186-3f85c8f3ebb3">

5. **Under the User Profile**: By Default, it has already created a user. You have the option to launch
A user profile represents a single user within a domain. It is the main way to reference a user for the purposes of sharing, reporting, and other user-oriented features.

<img width="1165" alt="2" src="https://github.com/user-attachments/assets/dc68217d-69a0-4d8f-9afe-df46adc2d893">

## Step 2: Set Up Your Environment

1. **Launch JupyterLab**: In SageMaker Studio, click on JupyterLab to start your environment
Select an instance as per the model requirement. I have take ml.m5.2x.large instance and 10GB storage, Click on Run Space

<img width="1512" alt="3" src="https://github.com/user-attachments/assets/6dfda2dd-c70d-4d96-a772-dd637414635b">


2. **Install SageMaker SDK**: Open a new  notebook and install the SageMaker SDK by running:
   ```python
   !pip install sagemaker -U
   ```
3. Create a SageMaker Session: Import necessary libraries and create a SageMaker session.
   ```python
    import sagemaker
    import boto3
    
    sess=sagemaker.Session()
    
    # sagemaker session bucket -> used for uploading data, models and logs
    # sagemaker will automatically create this bucket if it does not exist
    sagemaker_session_bucket=None
    if sagemaker_session_bucket is None and sess is not None:
        sagemaker_session_bucket = sess.default_bucket()
    
    # Role Management
    
    try:
        role = sagemaker.get_execution_role()
    except ValueError:
        iam = boto3.client("iam")
        role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']
    
    session = sagemaker.Session(default_bucket=sagemaker_session_bucket)
    
    print(f'sagemaker role arn:{role}')
    print(f'sagemaker session region:{sess.boto_region_name}')
   ```

## Step 3: Deploy a Hugging Face Model

1. **Select a Model**: Choose a Hugging Face model from the Hugging Face Hub. For example, you can use `distilbert-base-uncased-distilled-squad` for question-answering tasks.
2. **Configure the Model**:
   ``` python
    from sagemaker.huggingface.model import HuggingFaceModel
    
    # Hub model configuration <https://huggingface.co/models>
    
    hub = {
        'HF_MODEL_ID': 'distilbert-base-uncased-distilled-squad', #model id
        'HF_TASK': 'question-answering'                           #NLP Task you want to use for prediction
    }
    
    #model case
    huggingface_model = HuggingFaceModel(
    	env=hub,        #configuration for loading model from hub
        role=role,      #IAM role with permissions to create an endpoint
        transformers_version='4.26', 
        pytorch_version='1.13',
        py_version='py39',
    )
   ```
  3. **Deploy the Model**:
     ```python
    #deploy model to Sagemaker inference
    
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.xlarge'
    )
     ```
## Step 4: Test the Deployed Model

1. **Prepare Input Data**: Create the input data for the model.
``` python
#test the deployed model

data = {
    'inputs': {
        'question': 'What is used for inference?',
        'context': 'My name is Pranjal and I live in Utah. This model is used with sagemaker for inference'
    }
}
```
2.**Make Predictions**:
``` python
#request
predictor.predict(data)
```
3.**Outputs**:

<img width="933" alt="out1" src="https://github.com/user-attachments/assets/26d386b3-75b7-466e-9de0-4aeac81e7de5">

<img width="763" alt="out2" src="https://github.com/user-attachments/assets/334b3a16-1b62-4b11-b0fd-573a568df174">

## Conclusion

Deploying Hugging Face models on AWS SageMaker allows you to leverage powerful machine learning models in a scalable and managed environment. From the above steps, you can efficiently deploy and manage your models, ensuring to harness the full potential of AWS SageMaker.


