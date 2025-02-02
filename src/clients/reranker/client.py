import requests
from typing import List, Dict
from loguru import logger
import argparse


class RerankerClient:
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

    def rerank_client(self, pairs):
        payload = {
            "pairs": pairs,
        }
        response = requests.post(f"{self.api_url}/rerank", json=payload)
        if response.status_code == 200:
            result = response.json()
            return result.get('scores', '')
        else:
            raise Exception(f"Reranker API request failed with status code {response.status_code}")

    def v1_rerank(self, query: str, documents: List[str], model: str) -> Dict:
        payload = {
            "query": query,
            "model": model,
            "documents": documents
        }
        response = requests.post(f"{self.api_url}/v1/rerank", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Reranker API request failed with status code {response.status_code}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--url',
        default='http://127.0.0.1:5001',
        )
    parser.add_argument(
        '--query',
        default="What is the capital of France?",
    )
    parser.add_argument(
        '--documents',
        default=[
            "Berlin",
            "Paris",
            "London",
        ],
        nargs='+',
    )
    parser.add_argument(
        '--model',
        default="BAAI/bge-reranker-v2-m3",
    )
    parser.add_argument(
        '--pairs',
        default=[
            ['你好，请介绍下自己', '我是一个AI助手，能够回答各种问题和提供帮助。'],
            ['你好，请介绍下自己', 'default 值现在是一个包含多个对的扁平列表。每两个连续的字符串形成一对。'],
            ['中国的首都是哪里？', '中国的首都是北京。'],
            ['中国的首都是哪里？', '如果你想接受一个由逗号分隔的字符串，然后将其分割成列表'],
            ['人工智能的发展趋势如何？', '人工智能正在快速发展，在各个领域都有广泛应用。'],
            ['人工智能的发展趋势如何？', '这样的日志记录可以帮助你更好地理解和调试嵌入过程。根据你的具体需求，你可能需要调整日志的详细程度。'],
            ['全球变暖的主要原因是什么？', '全球变暖主要是由人类活动产生的温室气体排放造成的。'],
            ['全球变暖的主要原因是什么？', '这个错误表明在尝试将响应序列化为 JSON 时出现了问题。'],
            ['如何保持健康的生活方式？', '保持均衡饮食、定期锻炼、充足睡眠和管理压力是保持健康生活方式的关键。'],
            ['如何保持健康的生活方式？', '根据错误信息，看起来你的响应对象不是一个标准的字典或者没有 __dict__ 属性。']
        ]
        )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    reranker = RerankerClient(args.url)
    status = reranker.check_health()
    if not status:
        logger.error("Health check failed. The reranker server at"
                       " '{}' is not responding as expected.".format(args.url))
    else:
        logger.info("Health check succeeded. The reranker server at"
                    " '{}' is responding as expected.".format(args.url))
        try:
            # res = reranker.rerank_client(pairs=args.pairs)
            # if res:
            #     logger.info(f"scores: '{res}'")
            # else:
            #     logger.warning("None")

            res1 = reranker.v1_rerank(
                query=args.query,
                documents=args.documents,
                model=args.model
            )
            logger.debug(res1)
            if res1:
                for result in res1:
                    document_index = result['index']
                    score = result['relevance_score']
                    document = args.documents[document_index]
                    logger.info(f"Query: '{args.query}', Answer: '{document}', Score: {score:.5f}")

            else:
                logger.warning("No results returned")

        except requests.exceptions.RequestException as e:
            logger.error(f"Error while making request to reranker service: {e}")
        except Exception as e:
            logger.error(f"Unexpected error occurred: {e}")


if __name__ == "__main__":
    main()