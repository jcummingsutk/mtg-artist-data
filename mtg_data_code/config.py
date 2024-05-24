import os
from typing import Any

import yaml


def load_env_vars(config_filename: str, config_secret_filename: str) -> dict[str, Any]:
    with open(config_filename, "r") as f:
        config = yaml.safe_load(f)
    os.environ["BLOB_CONTAINER_NAME"] = config["blob_container_name"]

    if os.path.exists(config_secret_filename):
        with open(config_secret_filename, "r") as fp:
            config_secret = yaml.safe_load(fp)
        os.environ["BLOB_CONNECTION_STRING"] = config_secret["blob_connection_string"]
