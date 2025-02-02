import os
import configparser
import uvicorn
import argparse
from typing import List
from loguru import logger
from pydantic import BaseModel
from BCEmbedding import RerankerModel
from fastapi import FastAPI, HTTPException, Request


app = FastAPI()
bce_model, bge_tokenizer, bge_model, model_type = None, None, None, None


class RerankerQueryRequest(BaseModel):
    query: str
    model: str
    documents: List[str]


class RerankerResponse(BaseModel):
    ranked_passages: List[str]
    scores: List[float]


class RerankRequest(BaseModel):
    pairs: List[List[str]]


def get_model_type(model_path):
    keywords = ['bce', 'bge']
    for keyword in keywords:
        if keyword in model_path:
            return keyword
    raise ValueError("Unsupported model, please check the model path.")


def reranker_serve(args: str):
    global bce_model, bge_tokenizer, bge_model, model_type
    config = configparser.ConfigParser()
    config.read(args.config_path)
    reranker_model_path = config.get('rag', 'reranker_model_path') or None
    reranker_bind_port = int(config.get('rag', 'reranker_bind_port')) or None

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if reranker_model_path and reranker_bind_port:
        model_type = get_model_type(reranker_model_path)
        if model_type == 'bce':
            try:
                bce_model = RerankerModel(model_name_or_path=reranker_model_path, use_fp16=True)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            bge_tokenizer = AutoTokenizer.from_pretrained(
                reranker_model_path, trust_remote_code=True)
            bge_model = AutoModelForCausalLM.from_pretrained(
                reranker_model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16).eval().to('cuda')
    else:
        raise ValueError("reranker_model_path or reranker_bind_port is not specified in the config file.")

    uvicorn.run(app, host="0.0.0.0", port=reranker_bind_port)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/rerank")
async def rerank(request: RerankRequest):
    try:
        pairs = request.pairs

        if model_type == 'bce':
            scores = bce_model.compute_score(pairs)
        else:
            import torch
            with torch.no_grad():
                prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
                sep = '\n'
                prompt_inputs = bge_tokenizer(prompt,
                                              return_tensors=None,
                                              add_special_tokens=False)['input_ids']
                sep_inputs = bge_tokenizer(sep,
                                           return_tensors=None,
                                           add_special_tokens=False)['input_ids']
                inputs = []
                for query, passage in pairs:
                    query_inputs = bge_tokenizer(f'A: {query}',
                                                 return_tensors=None,
                                                 add_special_tokens=False,
                                                 max_length=1024 * 3 // 4,
                                                 truncation=True)
                    passage_inputs = bge_tokenizer(f'B: {passage}',
                                                   return_tensors=None,
                                                   add_special_tokens=False,
                                                   max_length=1024,
                                                   truncation=True)
                    item = bge_tokenizer.prepare_for_model(
                        [bge_tokenizer.bos_token_id] + query_inputs['input_ids'],
                        sep_inputs + passage_inputs['input_ids'],
                        truncation='only_second',
                        max_length=1024,
                        padding=False,
                        return_attention_mask=False,
                        return_token_type_ids=False,
                        add_special_tokens=False)
                    item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
                    item['attention_mask'] = [1] * len(item['input_ids'])
                    inputs.append(item)

                inputs_after_pad = bge_tokenizer.pad(inputs,
                                                     padding=True,
                                                     max_length=1024 + len(sep_inputs) +
                                                                len(prompt_inputs),
                                                     pad_to_multiple_of=8,
                                                     return_tensors='pt')

                inputs_after_pad = inputs_after_pad.to(bge_model.device)
                all_scores = bge_model(**inputs_after_pad,
                                       return_dict=True,
                                       cutoff_layers=[28])
                scores = [
                    scores[:, -1].view(-1, ).float()
                    for scores in all_scores[0]
                ]
                scores = scores[0].cpu()

        logger.debug(f"scores: {scores}")
        return {"scores": scores}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/rerank")
async def v1_rerank(request: RerankerQueryRequest):
    try:
        pairs = [[request.query, doc] for doc in request.documents]

        if model_type == 'bce':
            scores = bce_model.compute_score(pairs)
        else:
            import torch
            with torch.no_grad():
                prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
                sep = '\n'
                prompt_inputs = bge_tokenizer(prompt,
                                              return_tensors=None,
                                              add_special_tokens=False)['input_ids']
                sep_inputs = bge_tokenizer(sep,
                                           return_tensors=None,
                                           add_special_tokens=False)['input_ids']
                inputs = []
                for query, passage in pairs:
                    query_inputs = bge_tokenizer(f'A: {query}',
                                                 return_tensors=None,
                                                 add_special_tokens=False,
                                                 max_length=1024 * 3 // 4,
                                                 truncation=True)
                    passage_inputs = bge_tokenizer(f'B: {passage}',
                                                   return_tensors=None,
                                                   add_special_tokens=False,
                                                   max_length=1024,
                                                   truncation=True)
                    item = bge_tokenizer.prepare_for_model(
                        [bge_tokenizer.bos_token_id] + query_inputs['input_ids'],
                        sep_inputs + passage_inputs['input_ids'],
                        truncation='only_second',
                        max_length=1024,
                        padding=False,
                        return_attention_mask=False,
                        return_token_type_ids=False,
                        add_special_tokens=False)
                    item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
                    item['attention_mask'] = [1] * len(item['input_ids'])
                    inputs.append(item)

                inputs_after_pad = bge_tokenizer.pad(inputs,
                                                     padding=True,
                                                     max_length=1024 + len(sep_inputs) +
                                                                len(prompt_inputs),
                                                     pad_to_multiple_of=8,
                                                     return_tensors='pt')

                inputs_after_pad = inputs_after_pad.to(bge_model.device)
                all_scores = bge_model(**inputs_after_pad,
                                       return_dict=True,
                                       cutoff_layers=[28])
                scores = [
                    scores[:, -1].view(-1, ).float()
                    for scores in all_scores[0]
                ]
                scores = scores[0].cpu()

        results = []
        for i, score in enumerate(scores):
            data = {'index': i,
                    'relevance_score': float(score)}
            results.append(data)

        results.sort(key=lambda x: x['relevance_score'], reverse=True)

        return results

    except Exception as e:
        logger.error(f"Reranking error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_path',
        default='./config.ini')
    parser.add_argument(
        '--gpu_id',
        default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    reranker_serve(args)


if __name__ == "__main__":
    main()