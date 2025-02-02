import requests
from typing import List
from loguru import logger
import numpy as np
import argparse
from urllib3.util import Retry
from requests.adapters import HTTPAdapter


class EmbedderClient:
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

    def embed_documents(self, texts):
        payload = {
            "texts": texts,
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(f"{self.api_url}/embed_documents", json=payload,
                                 headers=headers)
        if response.status_code == 200:
            result = response.json()
            return result.get('embedding_list', '')
        else:
            raise Exception(f"Embedder API request failed with status code {response.status_code}")

    def embed_query(self, query: str = None, image: str = None):
        payload = {}
        if query is not None:
            payload["query"] = query
        if image is not None:
            payload["image"] = image

        headers = {'Content-Type': 'application/json'}
        response = requests.post(f"{self.api_url}/embed_query", json=payload,
                                 headers=headers)
        if response.status_code == 200:
            result = response.json()
            return result.get('embedding', '')
        else:
            raise Exception(f"Embedder API request failed with status code {response.status_code}")

    def v1_embeddings(self, model: str, input: List[str]):
        if not model or not input:
            raise ValueError("Both model and input parameters are required")

        payload = {
            "model": model,
            "input": input
        }

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]
        )

        session = requests.Session()
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        headers = {'Content-Type': 'application/json'}
        try:
            response = session.post(
                f"{self.api_url}/v1/embeddings",
                json=payload,
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Embedder API request failed with status code {response.status_code}")

        except requests.exceptions.Timeout:
            raise Exception("Request timed out")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}")
        finally:
            session.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--url',
        default='http://127.0.0.1:5003',
    )
    parser.add_argument(
        '--query',
        default='你好，请介绍下自己',
    )
    parser.add_argument(
        '--texts',
        default=[
            "Milvus是一个强大的向量数据库系统,专为相似度搜索和人工智能应用而设计。",
            "Python是一种多功能编程语言,以其简洁性和可读性而闻名。",
            "机器学习是人工智能的一个子集,专注于开发能够从数据中学习并进行预测或决策的算法。",
            "向量数据库是专门为存储和查询高维向量而优化的数据库系统,常用于机器学习和人工智能应用。",
            "深度学习是机器学习的一个分支,使用多层神经网络来模拟人脑的学习过程。",
            "自然语言处理是人工智能的一个重要领域,致力于使计算机理解、解释和生成人类语言"
        ]
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    embedder = EmbedderClient(args.url)
    status = embedder.check_health()

    if not status:
        logger.error("Health check failed. The embedding server at"
                     " '{}' is not responding as expected.".format(args.url))
    else:
        logger.info("Health check succeeded. The embedding server at"
                    " '{}' is responding as expected.".format(args.url))
        try:
            # res = embedder.embed_query(query=args.query)
            # if res:
            #     embedding_np = np.array(res)
            #     logger.info(f"query: '{args.query}'")
            #     logger.info(f"Shape: {embedding_np.shape}")
            # else:
            #     logger.warning(f"Received empty embedding for query: '{args.query}'")

            # res2 = embedder.embed_documents(texts=args.texts)
            # if res2:
            #     embedding_np2 = np.array(res2)
            #     logger.info(f"texts: {args.texts}")
            #     logger.info(f"Shape: {embedding_np2.shape}")
            # else:
            #     logger.warning("No embeddings were generated.")

            res3 = embedder.v1_embeddings(model='test', input=args.texts)
            if res3:
                data = res3.get("data", [])
                if data:
                    for i, embedding_item in enumerate(data):
                        embedding = embedding_item.get("embedding", [])
                        logger.info(f"Text {i + 1}: '{args.texts[i]}'")
                        embedding_array = np.array(embedding)
                        logger.info(f"Embedding shape: {embedding_array.shape}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error while making request to embedder service: {e}")
        except Exception as e:
            logger.error(f"Unexpected error occurred: {e}")


if __name__ == "__main__":
    main()