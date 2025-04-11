from __future__ import annotations
import argparse
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from loguru import logger
from langchain.docstore.document import Document
from tools import calc_time
from tqdm.auto import tqdm


def _default_text_mapping() -> Dict:
    return {
        'properties': {
            'text': {'type': 'text'},
            'updated_at': {'type': 'keyword'}
        }
    }


class ElasticKeywordsSearch():

    def __init__(
            self,
            elasticsearch_url: str,
            index_name: str,
            drop_old: Optional[bool] = False,
            *,
            ssl_verify: Optional[Dict[str, Any]] = None,
    ):
        try:
            import elasticsearch
        except ImportError:
            logger.error('Could not import elasticsearch python package. '
                         'Please install it with `pip install elasticsearch`.')
            return
        self.index_name = index_name
        self.drop_old = drop_old
        _ssl_verify = ssl_verify or {}
        self.elasticsearch_url = elasticsearch_url
        self.ssl_verify = _ssl_verify
        try:
            self.client = elasticsearch.Elasticsearch(elasticsearch_url, **_ssl_verify)
        except ValueError as e:
            logger.error(f'Your elasticsearch client string is mis-formatted. Got error: {e}')
            return

        index_exists = self.client.indices.exists(index=index_name)

        if drop_old and index_exists:
            try:
                self.client.indices.delete(index=index_name)
                logger.info(f"Deleted existing index '{index_name}'")
                index_exists = False
            except Exception as e:
                logger.error(f"Error deleting index '{index_name}': {e}")
                raise

        if not index_exists:
            try:
                mapping = _default_text_mapping()
                self.create_index(self.client, index_name, mapping)
                logger.info(f"Created new index '{index_name}' with mapping")
            except Exception as e:
                logger.error(f"Error creating index '{index_name}': {e}")
                raise

        logger.debug(f"ElasticKeywordsSearch initialized with URL: {elasticsearch_url} and index: {index_name}")

    def add_meta_info_in_kb(self, kb_id) -> None:
        current_time = calc_time()
        kb_info = {
            "dataset_id": kb_id,
            "type": "kb_info",
            "content": "",
            "collection_id": "",
            "embedding_id": "",
            "updated_at": current_time
        }
        try:
            self.client.update(
                index=self.index_name,
                id=f"{kb_id}_info",
                body={"doc": kb_info, "doc_as_upsert": True}
            )
        except Exception as e:
            logger.error(f"Failed to add knowledge base meta info: {str(e)}")
            raise

    def add_texts(
            self,
            knowledge_base_id: str,
            texts: List[str],
            file_id_list: List[str],
            chunks_ids_list: List[str],
            file_name_list: List[str],
            refresh_indices: bool = True,
            **kwargs: Any,
    ) -> Tuple[List[str], List[str]]:

        try:
            from elasticsearch.helpers import bulk
        except ImportError:
            raise ImportError('Could not import elasticsearch python package. '
                              'Please install it with `pip install elasticsearch`.')

        requests = []
        ids = []
        current_time = calc_time()

        estimated_size = 0
        MAX_BATCH_SIZE = 100 * 1024 * 1024  # 50MB
        current_batch = []
        all_failures = []

        pbar = tqdm(zip(texts, file_id_list, chunks_ids_list, file_name_list), total=len(texts))
        for text, file_id, chunk_id, file_name in pbar:
            pbar.set_description(f"Adding chunks to ES [{file_name}|{file_id}]")

            request = {
                '_op_type': 'update',
                '_index': self.index_name,
                '_id': f"{knowledge_base_id}_{file_id}_{chunk_id}",
                'doc': {
                    'dataset_id': knowledge_base_id,
                    'content': text,
                    'collection_id': file_id,
                    'embedding_id': chunk_id,
                    'file_name': file_name,
                    'updated_at': current_time
                },
                'doc_as_upsert': True
            }

            current_size = len(str(request).encode('utf-8'))
            estimated_size += current_size

            current_batch.append(request)
            ids.append(f"{knowledge_base_id}_{file_id}_{chunk_id}")

            if estimated_size >= MAX_BATCH_SIZE:
                try:
                    success, errors = bulk(
                        self.client,
                        current_batch,
                        stats_only=False,
                        chunk_size=1000,
                        request_timeout=120,
                        raise_on_error=False,
                        max_retries=3,
                        initial_backoff=2,
                        max_backoff=600
                    )
                    if errors:
                        all_failures.extend(errors)
                except Exception as e:
                    logger.error(f"Batch bulk operation failed: {str(e)}")
                    all_failures.extend([{"update": {"_id": req['_id']}} for req in current_batch])

                current_batch = []
                estimated_size = 0

        if current_batch:
            try:
                success, errors = bulk(
                    self.client,
                    current_batch,
                    stats_only=False,
                    chunk_size=500,
                    request_timeout=60,
                    raise_on_error=False
                )
                if errors:
                    all_failures.extend(errors)
            except Exception as e:
                logger.error(f"Final batch bulk operation failed: {str(e)}")
                all_failures.extend([{"update": {"_id": req['_id']}} for req in current_batch])

        if not all_failures:
            failed_files = []
        else:
            failed_files = {error['update']['doc']['collection_id'] for error in all_failures}

        if refresh_indices:
            try:
                self.client.indices.refresh(index=self.index_name)
            except Exception as e:
                logger.error(f"Refresh index failed: {str(e)}")

        return ids, failed_files

    def update_chunk(
            self,
            knowledge_base_id: str,
            file_id: str,
            chunk_id: str,
            text: str = None,
            **kwargs: Any,
    ) -> str:

        try:
            id = f"{knowledge_base_id}_{file_id}_{chunk_id}"

            if not self.client.exists(index=self.index_name, id=id):
                logger.error(f"Not found chunk with id in elasticsearch: {id}")
                return f"Not found chunk with id in elasticsearch: {id}"
            current_time = calc_time()
            update_body = {
                "doc": {
                    "dataset_id": knowledge_base_id,
                    "collection_id": file_id,
                    "embedding_id": chunk_id,
                    "updated_at": current_time
                }
            }

            if text is not None:
                update_body["doc"]["content"] = text

            response = self.client.update(
                index=self.index_name,
                id=id,
                body=update_body,
                refresh=kwargs.get('refresh', True)
            )

            success = response.get('result') in ['updated', 'noop']
            if success:
                logger.info(f"Successfully updated chunk with id: {id} in elasticsearch")
                return "success"

            else:
                logger.error(f"Failed to update chunk with id: {id} in elasticsearch")
                return f"Failed to update chunk with id: {id} in elasticsearch"

        except Exception as e:
            logger.error(f"Update chunk failed in elasticsearch: {str(e)}")
            return f"Update chunk failed in elasticsearch: {str(e)}"

    def similarity_search(self,
                          query: str,
                          k: int = 4,
                          query_strategy: str = 'match_phrase',
                          must_or_should: str = 'should',
                          **kwargs: Any) -> List[Document]:
        if k == 0:
            # pm need to control
            return []
        docs_and_scores = self.similarity_search_with_score(query,
                                                            k=k,
                                                            query_strategy=query_strategy,
                                                            must_or_should=must_or_should,
                                                            **kwargs)
        documents = [d[0] for d in docs_and_scores]
        return documents

    @staticmethod
    def _relevance_score_fn(distance: float) -> float:
        """Normalize the distance to a score on a scale [0, 1]."""
        # Todo: normalize the es score on a scale [0, 1]
        return distance

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return self._relevance_score_fn

    def similarity_search_with_score(
            self,
            query: str,
            knowledge_base_id: str,
            file_ids: List[str],
            keywords: list[str],
            k: int = 10,
            query_strategy: str = 'match_phrase',
            must_or_should: str = 'should',
            **kwargs: Any
    ) -> List[Tuple[Document, float]]:

        if not query or not query.strip():
            return []

        if k == 0:
            return []

        if not file_ids:
            logger.warning(f"No file_ids provided for knowledge_base_id: {knowledge_base_id}")
            return []

        assert must_or_should in ['must', 'should'], 'only support must and should.'

        match_query = {
            'bool': {
                'must': [
                    {'term': {'dataset_id': knowledge_base_id}},
                    {'terms': {'collection_id': file_ids}}
                ]
            }
        }

        keyword_queries = []
        for key in keywords:
            keyword_queries.append({query_strategy: {'content': key}})

        if keyword_queries:
            match_query['bool'][must_or_should] = keyword_queries
            if must_or_should == 'should':
                match_query['bool']['minimum_should_match'] = 1

        response = self.client_search(
            self.client,
            self.index_name,
            match_query,
            size=k
        )
        hits = response['hits']['hits']

        docs_and_scores = []
        for hit in hits:
            doc = Document(
                page_content=hit['_source']['content'],
                metadata={
                    'knowledge_base_id': hit['_source']['dataset_id'],
                    'file_id': hit['_source']['collection_id'],
                    'chunk_id': hit['_source']['embedding_id'],
                    'relevance_score': hit['_score'],
                    'file_name': hit['_source']['file_name']
                }
            )
            docs_and_scores.append(doc)

        return docs_and_scores

    def create_index(self, client: Any, index_name: str, mapping: Dict) -> None:
        version_num = client.info()['version']['number'][0]
        version_num = int(version_num)
        if version_num >= 8:
            client.indices.create(index=index_name, mappings=mapping)
        else:
            client.indices.create(index=index_name, body={'mappings': mapping})

    def client_search(self, client: Any, index_name: str, script_query: Dict, size: int) -> Any:
        version_num = client.info()['version']['number'][0]
        version_num = int(version_num)
        if version_num >= 8:
            response = client.search(index=index_name, query=script_query, size=size, timeout='5s')
        else:
            response = client.search(index=index_name, body={'query': script_query, 'size': size}, timeout='5s')
        return response

    def list_knowledge_database_id(self) -> List[str]:
        """
        Get all unique knowledge_base_id (dataset_id) from the index

        Returns:
            List[str]: List of unique knowledge_base_ids, returns empty list if exception occurs
        """
        try:
            query = {
                "size": 0,
                "aggs": {
                    "unique_dataset_ids": {
                        "terms": {
                            "field": "dataset_id.keyword",
                            "size": 10000
                        }
                    }
                }
            }

            response = self.client.search(
                index=self.index_name,
                body=query
            )

            dataset_ids = [bucket['key'] for bucket in response['aggregations']['unique_dataset_ids']['buckets']]

            if not dataset_ids:
                logger.warning(f"No dataset_ids found in index: {self.index_name}")

            return dataset_ids

        except Exception as e:
            logger.warning(f"Error fetching unique dataset_ids from index {self.index_name}: {str(e)}")
            return []

    def delete(
            self,
            knowledge_base_id: Optional[str] = None,
            file_id: Optional[str] = None,
            chunk_id: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """
        Method to delete documents

        Args:
            knowledge_base_id: Knowledge base ID (required)
            file_id: File ID (optional)
            chunk_id: Chunk ID (optional, must be provided together with file_id if used)

        Raises:
            ValueError: Raises exception when knowledge_base_id is not provided
        """
        try:
            from elasticsearch.exceptions import NotFoundError
        except ImportError:
            raise ImportError('Could not import elasticsearch python package. '
                              'Please install it with `pip install elasticsearch`.')

        try:

            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"dataset_id": knowledge_base_id}}
                        ]
                    }
                }
            }

            if file_id is not None:
                query["query"]["bool"]["must"].append(
                    {"term": {"collection_id": file_id}}
                )

                if chunk_id is not None:
                    query["query"]["bool"]["must"].append(
                        {"term": {"embedding_id": chunk_id}}
                    )
            elif chunk_id is not None:
                raise ValueError("file_id must be provided when chunk_id is specified")

            result = self.client.delete_by_query(
                index=self.index_name,
                body=query,
                refresh=True
            )

            # delete meta info
            if file_id is None and chunk_id is None:
                try:
                    self.client.delete(
                        index=self.index_name,
                        id=f"{knowledge_base_id}_info",
                        refresh=True
                    )
                except NotFoundError:
                    pass

            logger.info(f"Deleted {result['deleted']} documents")

            return result

        except NotFoundError:
            logger.warning(f"Index '{self.index_name}' not found")
        except ValueError as ve:
            logger.error(f"Validation error: {str(ve)}")
            raise
        except Exception as e:
            logger.error(f"Error deleting from index '{self.index_name}': {e}")
            raise

    def list_file_ids(self, knowledge_base_id: str):
        try:

            search_body = {
                "size": 0,
                "query": {
                    "term": {
                        "dataset_id.keyword": knowledge_base_id
                    }
                },
                "aggs": {
                    "unique_file_ids": {
                        "terms": {
                            "field": "collection_id.keyword",
                            "size": 10000
                        }
                    }
                }
            }

            response = self.client.search(
                index=self.index_name,
                body=search_body
            )

            buckets = response.get('aggregations', {}).get('unique_file_ids', {}).get('buckets', [])
            file_ids = [bucket['key'] for bucket in buckets]

            # logger.info(f"Retrieved {len(file_ids)} unique file IDs for knowledge_base_id '{knowledge_base_id}'.")
            return file_ids

        except Exception as e:
            logger.error(f"Failed to retrieve file IDs for knowledge_base_id '{knowledge_base_id}': {e}")
            raise

    def list_chunks_info(
            self,
            knowledge_base_id: str,
            file_id: str,
            chunk_id: Optional[str] = None,
            page_size: Optional[int] = None,
            page_num: Optional[int] = None,
            use_embedding_id_as_id: Optional[bool] = True,
            key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List chunks information for the specified knowledge base and file, with pagination support

        Args:
            knowledge_base_id (str): Knowledge base ID
            file_id (str): File ID
            chunk_id (Optional[str]): Chunk ID
            page_size (Optional[int]): Number of items per page
            page_num (Optional[int]): Page number, starting from 1
            use_embedding_id_as_id (bool): If True, use embedding_id as the id field in response

        Returns:
            Dict[str, Any]: Unified return format {"total": int, "chunks": List[Dict]}

        Raises:
            Exception: Raises exception when query fails
        """
        try:
            from elasticsearch import NotFoundError
        except ImportError:
            raise ImportError('Could not import elasticsearch python package. '
                              'Please install it with `pip install elasticsearch`.')

        try:
            if chunk_id:
                doc_id = f"{knowledge_base_id}_{file_id}_{chunk_id}"
                try:
                    response = self.client.get(
                        index=self.index_name,
                        id=doc_id
                    )
                    chunk_info = {
                        'id': response['_source']['embedding_id'] if use_embedding_id_as_id else response['_id'],
                        **response['_source']
                    }
                    return {
                        "total": 1,
                        "chunks": [chunk_info]
                    }
                except NotFoundError:
                    logger.warning(f"Document not found with id: {doc_id}")
                    return {
                        "total": 0,
                        "chunks": []
                    }

            query = {
                "bool": {
                    "must": [
                        {"term": {"dataset_id": knowledge_base_id}}
                    ]
                }
            }

            if file_id:
                query["bool"]["must"].append({"term": {"collection_id": file_id}})

            if key:
                query["bool"]["must"].append({
                    "match_phrase": {
                        "content": key
                    }
                })

            if page_size is not None and page_num is not None:
                if page_size <= 0 or page_num <= 0:
                    raise ValueError("page_size and page_num must be positive integers")

                from_index = (page_num - 1) * page_size
                response = self.client.search(
                    index=self.index_name,
                    query=query,
                    from_=from_index,
                    size=page_size,
                    sort=[{"_score": "desc"}] if key else None
                )

                total_hits = response['hits']['total']['value']
                chunks = [{
                    'id': hit['_source']['embedding_id'] if use_embedding_id_as_id else hit['_id'],
                    **hit['_source']
                } for hit in response['hits']['hits']]

                return {
                    "total": total_hits,
                    "chunks": chunks
                }

            scroll_size = 1000
            scroll_time = '2m'

            response = self.client.search(
                index=self.index_name,
                query=query,
                scroll=scroll_time,
                size=scroll_size,
                sort=[{"_score": "desc"}] if key else None
            )

            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']
            total_hits = response['hits']['total']['value']
            chunks = []

            for hit in hits:
                chunk_info = {
                    'id': hit['_source']['embedding_id'] if use_embedding_id_as_id else hit['_id'],
                    'score': hit['_score'] if key else None,
                    **hit['_source']
                }
                chunks.append(chunk_info)

            while len(hits) > 0:
                response = self.client.scroll(
                    scroll_id=scroll_id,
                    scroll=scroll_time
                )

                scroll_id = response['_scroll_id']
                hits = response['hits']['hits']

                for hit in hits:
                    chunk_info = {
                        'id': hit['_source']['embedding_id'] if use_embedding_id_as_id else hit['_id'],
                        'score': hit['_score'] if key else None,
                        **hit['_source']
                    }
                    chunks.append(chunk_info)

            try:
                self.client.clear_scroll(scroll_id=scroll_id)
            except Exception as e:
                logger.warning(f"Error clearing scroll: {str(e)}")

            return {
                "total": total_hits,
                "chunks": chunks
            }

        except Exception as e:
            logger.error(f"Error listing chunks info for knowledge base {knowledge_base_id} "
                         f"and file {file_id}: {str(e)}")
            raise

    def check_index_content(self) -> None:

        try:
            query = {
                "query": {
                    "match_all": {}
                },
                "size": 1
            }
            response = self.client.search(index=self.index_name, body=query)
            total_docs = response['hits']['total']['value']
            logger.info(f"Total documents in index: {total_docs}")
            if response['hits']['hits']:
                logger.info(f"Sample document: {response['hits']['hits'][0]['_source']}")
            else:
                logger.info("No documents found in the index")
        except Exception as e:
            logger.info(f"Error checking index content: {str(e)}")


def read_text(filepath):
    with open(filepath) as f:
        txt = f.read()
    return txt


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Feature store for processing directories.')
    parser.add_argument(
        '--elasticsearch_url',
        type=str,
        default=None)
    parser.add_argument(
        '--index_name',
        type=str,
        default=None)
    parser.add_argument(
        '--query',
        type=str,
        default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    elastic_search = ElasticKeywordsSearch(
        elasticsearch_url=args.elasticsearch_url,
        index_name=args.index_name)