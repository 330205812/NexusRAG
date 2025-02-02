import os
import math
import numpy as np
import torch
from pydantic import BaseModel
from typing import Optional
from enum import Enum
from loguru import logger
from typing import Any, List
import configparser
import uvicorn
import argparse
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

    def embed_query(self, text: str = None, image: str = None) -> np.ndarray:
        """Embed input text or image as feature, output np.ndarray with np.float32"""
        if self.bge:
            with torch.no_grad():
                feature = self.client.encode(text=text, image=image)
                return feature.cpu().numpy().astype(np.float32)
        else:
            if text is None:
                raise ValueError('This model only support text')
            emb = self.client.encode([text], show_progress_bar=True, normalize_embeddings=True)
            emb = emb.astype(np.float32)
            for norm in np.linalg.norm(emb, axis=1):
                assert abs(norm - 1) < 0.001
            return emb

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """

        texts = list(map(lambda x: x.replace("\n", " "), texts))

        embeddings = self.client.encode(
            texts,
            show_progress_bar=True,
            batch_size=self.batch_size,
            normalize_embeddings=True
        )

        return embeddings.tolist()

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


@app.post("/embed_query")
async def embed_query(request: QueryRequest):
    try:
        query = request.query
        image = request.image

        embedding = embedder.embed_query(text=query, image=image)

        logger.debug(f"query: {query}")
        logger.debug(f"embedding: {embedding}")
        logger.debug(f"embedding shape: {embedding.shape}")
        return {"embedding": embedding.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed_documents")
async def embed_documents(request: DocumentsRequest):
    try:
        texts = request.texts
        embedding_list = embedder.embed_documents(texts)

        embedding_list_np = np.array(embedding_list)
        logger.info(f"texts: {texts}")
        logger.info(f"Shape: {embedding_list_np.shape}")

        return {"embedding_list": embedding_list}

    except Exception as e:
        logger.error(f"Error in embed_documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/embeddings")
async def v1_embeddings(request: Request):
    try:
        body = await request.json()
        model = body.get("model")
        input_texts = body.get("input")

        if not model or not input_texts:
            raise HTTPException(status_code=400, detail="Missing model or input parameter")

        result = embedder.v1_embeddings(model, input_texts)
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
        default=32)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    embedder_serve(args)


if __name__ == "__main__":
    main()