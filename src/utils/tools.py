import os
import re
import time
import hashlib
from datetime import datetime
from loguru import logger
from filelock import FileLock, Timeout


def handle_status_file(path, file_id, operation='read', status='pending'):
    """
    Process status file operations: create, read, or update

    Args:
        path: Directory path
        file_id: File ID
        operation: 'read' or 'write'
        status: 'pending', 'success', 'error'

    Returns:
        str: Status if operation is 'read'
        None: If operation is 'write'

    Raises:
        Exception: When file lock acquisition fails or file operation fails
    """
    status_file = os.path.join(path, f"{file_id}.status")
    lock_file = f"{status_file}.lock"
    lock = FileLock(lock_file, timeout=10)

    try:
        with lock:
            if operation == 'read':
                if not os.path.exists(status_file):
                    logger.warning(f"Status file not found: {status_file}")
                    return None

                try:
                    with open(status_file, 'r') as f:
                        current_status = f.read().strip()
                        logger.info(f"Read status file: {status_file}, Status: {current_status}")
                        return current_status
                except Exception as e:
                    logger.error(f"Error reading status file {status_file}: {e}")
                    raise Exception(f"Error reading status file {status_file}: {e}")

            elif operation == 'write':
                try:
                    with open(status_file, 'w') as f:
                        f.write(status)
                    logger.info(f"Write status file: {status_file}, Status: {status}")
                    return None
                except Exception as e:
                    logger.error(f"Error writing status file {status_file}: {e}")
                    raise Exception(f"Error writing status file {status_file}: {e}")

            else:
                raise ValueError(f"Invalid operation: {operation}")

    except Timeout:
        error_msg = f"Can not obtain lock_file: {lock_file}"
        logger.error(error_msg)
        raise Exception(error_msg)
    except Exception as e:
        logger.error(f"Error occurred while processing status file: {e}")
        raise


def sanitize_filename(filename: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_\u4e00-\u9fff-]', '_', filename)


def get_partial_sha256_hash_from_text(text: str):
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    hash_object = hashlib.sha256()
    hash_object.update(text.encode('utf-8'))

    return hash_object.hexdigest()[0:16] + "_" + f"{timestamp}"


def get_partial_sha256_hash(filepath: str):
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    hash_object = hashlib.sha256()
    with open(filepath, 'rb') as file:
        chunk_size = 8192
        while chunk := file.read(chunk_size):
            hash_object.update(chunk)

    return hash_object.hexdigest()[0:16] + "_" + f"{timestamp}"


def calc_time():
    current_time = datetime.now().astimezone()
    rfc3339_str = current_time.isoformat()
    return rfc3339_str