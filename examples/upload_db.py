import boto3
import os
from dotenv import load_dotenv
from botocore.exceptions import NoCredentialsError, ClientError
from tqdm import tqdm  # Import tqdm


class UploadProgress(tqdm):
    """
    Wrapper for tqdm to show progress of file uploads.
    """

    def __init__(self, total_size, *args, **kwargs):
        super().__init__(total=total_size, unit="B", unit_scale=True, *args, **kwargs)

    def __call__(self, bytes_transferred):
        self.update(bytes_transferred)


def upload_file_with_progress(file_path, bucket_name, object_name=None):
    """
    Upload a file to an S3 bucket with a progress bar.

    :param file_path: Path to the file to upload
    :param bucket_name: S3 bucket to upload to
    :param object_name: S3 object name. If not specified, file_path base name is used.
    :return: True if file was uploaded, else False
    """
    if object_name is None:
        object_name = os.path.basename(file_path)

    try:
        total_size = os.path.getsize(file_path)
        bucket = s3.Bucket(bucket_name)

        print(f"Uploading '{file_path}' to '{bucket_name}/{object_name}'...")
        with UploadProgress(
            total_size, desc=f"Uploading {os.path.basename(file_path)}"
        ) as progress:
            bucket.upload_file(file_path, object_name, Callback=progress)

        print(
            f"\nFile '{file_path}' uploaded successfully to '{bucket_name}/{object_name}'"
        )
        return True
    except NoCredentialsError:
        print(
            "\nCredentials not available. Make sure you have configured your AWS credentials."
        )
        return False
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        error_message = e.response.get("Error", {}).get("Message")
        print(f"\nS3 Client Error uploading file: [{error_code}] {error_message}")
        print(f"Full error response: {e.response}")
        return False
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        return False


db_file = "miriad_medical_minlm_500k_noidx.db"

load_dotenv()

s3 = boto3.resource(
    "s3",
    endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name="auto",
)

upload_file_with_progress(db_file, "loma-database", db_file)
