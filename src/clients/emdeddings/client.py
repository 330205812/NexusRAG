import requests
from typing import List
from loguru import logger
import numpy as np
import argparse
import asyncio
import aiohttp
from urllib3.util import Retry
from requests.adapters import HTTPAdapter


class AsyncEmbedderClient:
    def __init__(self, api_url: str, batch_size: int = 32):
        self.api_url = api_url
        self.timeout = 60
        self._session = None
        self._session_lock = asyncio.Lock()
        self.batch_size = max(1, batch_size)
        self._closed = False

    async def get_session(self):
        """Lazy load session"""
        if self._closed:
            raise RuntimeError("Session is closed")

        if self._session is None:
            async with self._session_lock:
                if self._session is None and not self._closed:
                    self._session = aiohttp.ClientSession()
        return self._session

    async def v1_embeddings(self, model: str, input: List[str]):
        if not model or not input:
            raise ValueError("Both model and input parameters are required")

        if not isinstance(input, list):
            raise ValueError("Input must be a list of strings")

        results = []

        for i in range(0, len(input), self.batch_size):
            batch = input[i:i + self.batch_size]

            payload = {
                "model": model,
                "input": batch
            }

            headers = {'Content-Type': 'application/json'}
            session = await self.get_session()

            for attempt in range(3):
                try:
                    async with session.post(
                            f"{self.api_url}/v1/embeddings",
                            json=payload,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            batch_result = await response.json()
                            results.extend(batch_result['data'])
                            break
                        else:
                            error_text = await response.text()
                            logger.error(f"Attempt {attempt + 1} failed with status {response.status}: {error_text}")
                            if response.status not in {500, 502, 503, 504}:
                                break
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.error(f"Attempt {attempt + 1} failed with error: {str(e)}")
                    if attempt == 2:
                        raise Exception(f"Request failed after 3 attempts: {str(e)}")

                await asyncio.sleep(2 ** attempt)

        return {
            'object': "list",
            'data': results,
            'model': model,
            'usage': {
                'prompt_tokens': sum(len(text.split()) for text in input),
                'total_tokens': sum(len(text.split()) for text in input)
            }
        }

    async def close(self):
        """Close session"""
        async with self._session_lock:
            if not self._closed and self._session is not None:
                try:
                    await self._session.close()
                except Exception as e:
                    logger.error(f"Error closing session: {e}")
                finally:
                    self._session = None
                    self._closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--url',
        default='http://localhost:5003',
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


async def async_main():
    args = parse_args()
    async_embedder = AsyncEmbedderClient(args.url)
    try:
        logger.info("Testing AsyncEmbedderClient:")
        res_async = await async_embedder.v1_embeddings(model='test', input=args.texts)
        if res_async:
            data = res_async.get("data", [])
            if data:
                for i, embedding_item in enumerate(data):
                    embedding = embedding_item.get("embedding", [])
                    logger.info(f"Async Text {i + 1}: '{args.texts[i]}'")
                    embedding_array = np.array(embedding)
                    logger.info(f"Async Embedding shape: {embedding_array.shape}")
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
    finally:
        await async_embedder.close()


if __name__ == "__main__":
    asyncio.run(async_main())