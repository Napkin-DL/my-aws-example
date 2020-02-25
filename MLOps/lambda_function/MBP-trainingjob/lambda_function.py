import boto3
import os
import json
import datetime
from time import gmtime, strftime
from boto3.session import Session

region = boto3.session.Session().region_name

sagemaker = boto3.client('sagemaker')
code_pipeline = boto3.client('codepipeline')

def lambda_handler(event, context):
    try:
        
        print("event : {}".format(event))
        
        experiments_name = 'MBP-experiments-v3'
        
        
        train_start = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        train_start_calc = datetime.datetime.now()
    
        codepipeline_job = event['CodePipeline.job']['id']
        print('[INFO]CODEPIPELINE_JOB:', codepipeline_job)
        print('[INFO]TRAIN_START:', train_start)
        
        
        userParamText = event['CodePipeline.job']['data']['actionConfiguration']['configuration']['UserParameters']
        
        # "UserParameters": " {\"traincompute\": \"ml.c4.2xlarge\" , \"traininstancevolumesize\": 10, \"traininstancecount\": 1} "
        user_param = json.loads(userParamText)
        # user_param = {"traincompute" : "ml.p2.8xlarge",
        #       "traininstancevolumesize" : 10,
        #       "traininstancecount" : 1}
        job_name = 'mbp-mlops-pytorch-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        print('[INFO]TRAINING_JOB_NAME:', job_name)
        
    
        event['job_name'] = job_name
        event['stage'] = 'Training'
        event['status'] = 'InProgress'
        event['message'] = 'training job "{} started."'.format(job_name)
 
        # Get path to the most recent image built in the previous BUILD stage of the pipeline
        # Get Account Id from lambda function arn
        LambdaArn = context.invoked_function_arn
        print("lambda arn: ", LambdaArn)
        # Get Account ID from lambda function arn in the context
        AccountID = context.invoked_function_arn.split(":")[4]
        print("Account ID=", AccountID)
        create_experiments(experiments_name)
        create_training_job(user_param, job_name, experiments_name, train_start, AccountID)
        
        write_job_info_s3(event)
        put_job_success(event, train_start_calc)

    except Exception as e:
        print(e)
        print('[ERROR] Unable to create training job.')
        event['message'] = str(e)
        put_job_failure(event)

    return event

def create_experiments(experiments_name):
    try:
        print("[INFO] Creating experiments")
        create_experiments_params = \
        {
            "ExperimentName" : experiments_name,
            "DisplayName" : 'MBP-experiments-v3',
            "Description" : "classifier to defect on images"
        }
        response_exp = sagemaker.create_experiment(**create_experiments_params) 
    except:
        print("[INFO] Already exists experiments_name")
        pass

def create_training_job(user_param, job_name, experiments_name, train_start, AccountID):

    try:
        print("[INFO]CODEPIPELINE_USER_PARAMETERS:", user_param)

        # Environment variable containing S3 bucket for storing the model artifact
        model_artifact_bucket = os.environ['S3Bucket'] + "/ModelArtifact"
        print("[INFO]MODEL_ARTIFACT_BUCKET:", model_artifact_bucket)

        # Environment variable containing S3 bucket containing training data
        data_bucket = os.environ['S3Bucket'] + "/DAGM"
        print("[INFO]TRAINING_DATA_BUCKET:", data_bucket)

    
        ECRRepository = os.environ['ECRRepository']
        container_path = AccountID + '.dkr.ecr.' + region + ".amazonaws.com/" + ECRRepository + ":latest"
        print('[INFO]CONTAINER_PATH:', container_path)
     
 
        # Role to pass to SageMaker training job that has access to training data in S3, etc
        SageMakerRole = os.environ['SageMakerExecutionRole']
        
        train_instance_type = user_param['traincompute']
        train_volume_size = user_param['traininstancevolumesize']
        train_instance_count = user_param['traininstancecount']
        print('[INFO]TRAIN_INSTANCE_TYPE:', train_instance_type)
        print('[INFO]TRAIN_VOLUME_SIZE:', train_volume_size)
        print('[INFO]TRAIN_INSTANCE_COUNT:', train_instance_count)
        
        trial_name = "{}-{}".format(train_start, job_name)
    
        create_trials_params = \
        {
            "ExperimentName" : experiments_name,
            "TrialName" : trial_name
        }

        tensorboard_output = "s3://{}/{}/tensorboard".format(model_artifact_bucket, job_name)
        model_output= "s3://{}/{}/model".format(model_artifact_bucket, job_name)
        checkpoint_s3_uri = "s3://{}/{}/checkpoint".format(model_artifact_bucket, job_name)

        
        create_training_params = \
        {
            "RoleArn": SageMakerRole,
            "TrainingJobName": job_name,
            "AlgorithmSpecification": {
                "TrainingImage": container_path,
                "TrainingInputMode": "File"
            },
            "ResourceConfig": {
                "InstanceCount": train_instance_count,
                "InstanceType": train_instance_type,
                "VolumeSizeInGB": train_volume_size
            },
            "InputDataConfig": [
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": "s3://{}".format(data_bucket),
                            "S3DataDistributionType": "FullyReplicated"
                        }
                    },
                    "ContentType": "application/x-image",
                    "CompressionType": "None"
                }
            ],
            "OutputDataConfig": {
                "S3OutputPath": model_output
            },
            "ExperimentConfig": {
                "ExperimentName": experiments_name,
                "TrialComponentDisplayName": "Training",
                "TrialName": trial_name
            },
            "HyperParameters": {
                "epochs": "1",
                "backend": "gloo",
                "pretrained_jobname": "DAGM-training-job-1579661636",
                "batch-size" : "128",
                "dropout": "0.8",
                "lr" : "0.001",
                "test-batch-size" : "1000",
                "optimizer": "sgd",
                "num_steps" : "50"
            },
            "EnableManagedSpotTraining": False,
            "StoppingCondition": {
              "MaxRuntimeInSeconds": 12*60*60,
            #   "MaxWaitTimeInSeconds": 12*60*60
            },
           "TensorBoardOutputConfig": { 
              "S3OutputPath": tensorboard_output
            },
           "CheckpointConfig": { 
               "S3Uri": checkpoint_s3_uri
           }
        }

        response_trials = sagemaker.create_trial(**create_trials_params)
        response = sagemaker.create_training_job(**create_training_params)

    except Exception as e:
        print(str(e))
        raise(e)
        
def write_job_info_s3(event):
    print(event)

    objectKey = event['CodePipeline.job']['data']['outputArtifacts'][0]['location']['s3Location']['objectKey']
    bucketname = event['CodePipeline.job']['data']['outputArtifacts'][0]['location']['s3Location']['bucketName']
    artifactCredentials = event['CodePipeline.job']['data']['artifactCredentials']
    artifactName = event['CodePipeline.job']['data']['outputArtifacts'][0]['name']
    
    # S3 Managed Key for Encryption
    S3SSEKey = os.environ['SSEKMSKeyIdIn']

    json_data = json.dumps(event)
    print(json_data)

    session = Session(aws_access_key_id=artifactCredentials['accessKeyId'],
                  aws_secret_access_key=artifactCredentials['secretAccessKey'],
                  aws_session_token=artifactCredentials['sessionToken'])
   

    s3 = session.resource("s3")
    object = s3.Object(bucketname, objectKey)
    print(object)
    object.put(Body=json_data, ServerSideEncryption='aws:kms', SSEKMSKeyId=S3SSEKey)
    
    print('[SUCCESS]Job Information Written to S3')

def put_job_success(event, train_start_calc):
    
    train_end = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    train_end_calc = datetime.datetime.now()
    print('[INFO]TRAIN_END_SUCCESS:', train_end)
    total_train_time = train_end_calc - train_start_calc
    print('[INFO]TOTAL_TRAIN_TIME:', total_train_time)
    print(event['message'])
    code_pipeline.put_job_success_result(jobId=event['CodePipeline.job']['id'])


def put_job_failure(event):
   
    print('[FAILURE]Putting job failure')
    print(event['message'])
    code_pipeline.put_job_failure_result(jobId=event['CodePipeline.job']['id'], failureDetails={'message': event['message'], 'type': 'JobFailed'})
    return event
