# -*- coding: utf-8 -*-
import configparser
import time

import requests
from loguru import logger
import argparse
import os


class PdfClient:
    def __init__(self, api_url):
        self.api_url = api_url

    def check_health(self):
        health_check_url = f"{self.api_url}/health"
        try:
            response = requests.get(health_check_url)
            if response.status_code == 200:
                logger.info("Server is healthy and ready to process requests.")
                return True
            else:
                logger.error(f"Server health check failed with status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check request failed: {e}")
            return False

    def ocr_pdf_client(self, path, output_dir):
        payload = {
            "path": str(path),
            "output_dir": str(output_dir),
        }

        try:
            response = requests.post(f"{self.api_url}/pdf_ocr", json=payload)
            output_dir = response.json()['output_path']
            response.raise_for_status()
            return output_dir if response.json()['status_code'] == 200 else None
        except requests.exceptions.RequestException as e:
            logger.error(f"OCR PDF API request failed: {e}")
            return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path',
        '-p',
        required=True
    )
    parser.add_argument(
        '--output_dir',
        '-o',
        required=True
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = configparser.ConfigParser()
    config.read(args.config_path)
    pdf_server = config.get('server', 'pdf_server')
    embedder = PdfClient(pdf_server)
    doc_analyze_start = time.time()

    if not os.path.isabs(args.output_dir):
        current_working_directory = os.getcwd()
        output_dir = os.path.join(current_working_directory, args.output_dir)

    else:
        output_dir = args.output_dir
    logger.info(f'output_dir:{output_dir}')

    try:
        res = embedder.ocr_pdf_client(path=args.path, output_dir=output_dir)
        if res:
            logger.info(f"output_dir: '{res}'")
        else:
            logger.warning("None")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error while making request to reranker service: {e}")
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
    doc_analyze_cost = time.time() - doc_analyze_start

    logger.info(f'解析当前pdf{args.path}耗时为:{doc_analyze_cost}')


if __name__ == "__main__":
    main()


