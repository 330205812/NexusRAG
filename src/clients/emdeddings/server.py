import os
import math
from pydantic import BaseModel
from typing import Optional
from enum import Enum
from loguru import logger
from typing import Any, List
import configparser
import uvicorn
import argparse
import asyncio
from fastapi import FastAPI, HTTPException, Request

app = FastAPI()


class QueryRequest(BaseModel):
    query: Optional[str] = None
    image: Optional[str] = None


class DocumentsRequest(BaseModel):
    texts: List[str]


class DistanceStrategy(str, Enum):
    EUCLIDEAN_DISTANCE = "EUCLIDEAN_DISTANCE"
    MAX_INNER_PRODUCT = "MAX_INNER_PRODUCT"
    UNKNOWN = 'UNKNOWN'

    @staticmethod
    def euclidean_relevance_score_fn(distance: float) -> float:
        return 1.0 - distance / math.sqrt(2)

    @staticmethod
    def max_inner_product_relevance_score_fn(similarity: float) -> float:
        return similarity


class Embedder:
    client: Any
    _type: str

    def __init__(self, model_path: str, batch_size: int = 32):
        self.batch_size = batch_size
        self.support_image = False
        # bce also use euclidean distance.
        self.distance_strategy = DistanceStrategy.EUCLIDEAN_DISTANCE
        self.bge = False
        if self.use_multimodal(model_path=model_path):
            from FlagEmbedding.visual.modeling import Visualized_BGE
            self.support_image = True
            vision_weight_path = os.path.join(model_path, 'Visualized_m3.pth')
            self.client = Visualized_BGE(
                model_name_bge=model_path,
                model_weight=vision_weight_path).eval()
            self.bge = True
            logger.info('Use BGE.')
        else:
            from sentence_transformers import SentenceTransformer
            self.client = SentenceTransformer(model_name_or_path=model_path).half()
            logger.info('Do not use BGE.')

    @classmethod
    def use_multimodal(self, model_path):
        """Check text2vec model using multimodal or not."""

        if 'bge-m3' not in model_path.lower():
            return False

        vision_weight = os.path.join(model_path, 'Visualized_m3.pth')
        if not os.path.exists(vision_weight):
            logger.warning(
                '`Visualized_m3.pth` (vision model weight) not exist')
            return False
        return True

    def v1_embeddings(self, model: str, input: List[str]):

        embeddings = self.client.encode(
            input,
            show_progress_bar=True,
            batch_size=self.batch_size,
            normalize_embeddings=True
        )

        data = []
        for emb in embeddings:
            data.append({
                "object": "embedding",
                "embedding": emb.tolist()
            })

        total_tokens = sum(len(text.split()) for text in input)

        res_dict = {
            'object': "list",
            'data': data,
            'model': model,
            'usage': {
                'prompt_tokens': total_tokens,
                'total_tokens': total_tokens
            }
        }

        return res_dict


def embedder_serve(args: str):
    global embedder
    config = configparser.ConfigParser()
    config.read(args.config_path)
    embedding_model_path = config.get('rag', 'embedding_model_path') or None
    embedding_bind_port = int(config.get('rag', 'embedding_bind_port')) or None

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if embedding_model_path and embedding_bind_port:
        embedder = Embedder(embedding_model_path)
    else:
        raise ValueError("embedding_model_path or embedding_bind_port is not specified in the config file.")

    uvicorn.run(app, host="0.0.0.0", port=embedding_bind_port)


@app.post("/v1/embeddings")
async def v1_embeddings(request: Request):
    try:
        body = await request.json()
        model = body.get("model")
        input_texts = body.get("input")

        if not model or not input_texts:
            raise HTTPException(status_code=400, detail="Missing model or input parameter")

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            embedder.v1_embeddings,
            model,
            input_texts
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_path',
        type=str,
        default='./config.ini',
        help='config目录')
    parser.add_argument(
        '--gpu_id',
        default=None)
    parser.add_argument(
        '--batch_size',
        default=64)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    embedder_serve(args)


if __name__ == "__main__":
    main()