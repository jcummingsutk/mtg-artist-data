import glob
import os

from azure.storage.blob import BlobClient, BlobServiceClient

from mtg_data_code.config import load_env_vars

load_env_vars("config.yaml", "config_secret.yaml")
blobs = list(
    glob.glob(os.path.join("data", "model_training_and_eval", "*", "*", "*.jpg"))
)
blobs.append(os.path.join("data", "output_params.yaml"))

blob_service_client = BlobServiceClient.from_connection_string(
    os.environ["BLOB_CONNECTION_STRING"]
)
container_client = blob_service_client.get_container_client(
    os.environ["BLOB_CONTAINER_NAME"]
)

for blob_filename in blobs:
    with open(blob_filename, "rb") as f:
        blob_client = BlobClient.from_connection_string(
            conn_str=os.environ["BLOB_CONNECTION_STRING"],
            container_name=os.environ["BLOB_CONTAINER_NAME"],
            blob_name=blob_filename,
            max_block_size=1024 * 1024 * 4,
            max_single_put_size=1024 * 1024 * 8,
        )
        blob_client.upload_blob(f, overwrite=True, blob_type="BlockBlob")
