import boto3
import os
from dotenv import load_dotenv
from botocore.exceptions import NoCredentialsError, ClientError
from tqdm import tqdm  # Import tqdm


class DownloadProgress(tqdm):
    """
    Wrapper for tqdm to show progress of file downloads.
    """

    def __init__(self, total_size, *args, **kwargs):
        super().__init__(total=total_size, unit="B", unit_scale=True, *args, **kwargs)

    def __call__(self, bytes_transferred):
        self.update(bytes_transferred)


def download_file_with_progress(bucket_name, object_name, file_path=None):
    """
    Download a file from an S3 bucket with a progress bar.

    :param bucket_name: S3 bucket to download from
    :param object_name: S3 object name to download
    :param file_path: Local path to save the file. If not specified, object_name is used.
    :return: True if file was downloaded, else False
    """
    if file_path is None:
        file_path = os.path.basename(object_name)

    try:
        # Get object size for progress bar
        s3_client = boto3.client(
            "s3",
            endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name="auto",
        )

        response = s3_client.head_object(Bucket=bucket_name, Key=object_name)
        total_size = response["ContentLength"]

        bucket = s3.Bucket(bucket_name)

        print(f"Downloading '{bucket_name}/{object_name}' to '{file_path}'...")
        with DownloadProgress(
            total_size, desc=f"Downloading {os.path.basename(object_name)}"
        ) as progress:
            bucket.download_file(object_name, file_path, Callback=progress)

        print(
            f"\nFile '{bucket_name}/{object_name}' downloaded successfully to '{file_path}'"
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
        print(f"\nS3 Client Error downloading file: [{error_code}] {error_message}")
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

download_file_with_progress("loma-database", db_file, db_file)
