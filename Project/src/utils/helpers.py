# src/utils/helpers.py
import json
import logging

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
