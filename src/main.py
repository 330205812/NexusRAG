import os
import configparser
import uvicorn
import argparse
from loguru import logger
from fastapi import FastAPI
from services.service_manager import ServiceManager
from api.routes import setup_routes
import concurrent.futures
import atexit

API_EXECUTOR, LIGHT_EXECUTOR = None, None
app = FastAPI(title="知识库 API")


def init_thread_pools(config):
    global API_EXECUTOR, LIGHT_EXECUTOR

    api_max_workers = int(config.get('rag', 'api_max_workers', fallback=8))
    light_max_workers = int(config.get('rag', 'light_max_workers', fallback=8))

    API_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
        max_workers=api_max_workers,
        thread_name_prefix="api_worker"
    )

    LIGHT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
        max_workers=light_max_workers,
        thread_name_prefix="light_worker"
    )

    logger.info(
        f"Thread pools initialized with: API={api_max_workers}, LIGHT={light_max_workers}")


def close_thread_pools():
    global API_EXECUTOR, LIGHT_EXECUTOR

    if API_EXECUTOR:
        API_EXECUTOR.shutdown(wait=True)
    if LIGHT_EXECUTOR:
        LIGHT_EXECUTOR.shutdown(wait=True)

    logger.info("All thread pools have been shut down")


atexit.register(close_thread_pools)


def init_logging(config):
    log_dir = config.get('rag', 'log_dir', fallback='')
    if log_dir:
        log_file_path = os.path.join(log_dir, 'knowledgebase_service.log')
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(current_dir, 'log')
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, 'knowledgebase_service.log')

    logger.add(log_file_path, rotation='10MB', compression='zip')


def knowledgebase_serve(args: str):
    config = configparser.ConfigParser()
    config.read(args.config_path)

    init_logging(config)
    init_thread_pools(config)

    knowledgebase_bind_port = int(config.get('rag', 'knowledgebase_bind_port')) or None

    if not knowledgebase_bind_port:
        raise ValueError("knowledgebase_bind_port is not specified in the config file.")

    ServiceManager.initialize(config)
    setup_routes(app)

    logger.info(f"Starting knowledge base service on port: {knowledgebase_bind_port}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=knowledgebase_bind_port,
        timeout_keep_alive=300,
        limit_concurrency=100,
        timeout_graceful_shutdown=300,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_path',
        default='./config.ini',
        help='config path')
    return parser.parse_args()


def main():
    args = parse_args()
    knowledgebase_serve(args)


if __name__ == "__main__":
    main()