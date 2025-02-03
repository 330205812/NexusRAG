import os
import shutil
import hashlib
import numpy as np
import jieba.analyse
from loguru import logger
from typing import List, Optional, Any
from pymilvus import connections, utility, Collection
from pymilvus.exceptions import MilvusException
from utils.document_processor import DocumentProcessor, FeatureDataBase
from utils.tools import handle_status_file
from store.milvus import Milvus
from store.elastic_search import ElasticSearch
from clients.emdeddings.client import EmbedderClient
from clients.reranker.client import RerankerClient
from clients.pdf.client import PdfClient
from models.models import SearchMode, SearchResult


class KnowledgeBaseService:

    def __init__(self, config):

        """
        knowledge_base_service.py
        work_dir/
        ├── uploads/
        │   └── knowledge_id_dir_uploads/
        │       ├── sanitize_filename(file1)/
        │       │   └── file1.pdf
        │       ├── sanitize_filename(file2)/
        │       │   └── file2.docx
        │       └── sanitize_filename(file3)/
        │           └── file3.txt
        └── processed/
            └── knowledge_id_dir_processed/
                ├── sanitize_filename(file1)/
                │   ├── images/
                │   ├── file1_layout.pdf
                │   ├── file1_origin.pdf
                │   └── sanitize_filename(file1).txt
                ├── sanitize_filename(file2)/
                │   └── file2-md5.txt
                └── sanitize_filename(file3)/
                    └── file3-md5.txt
        """
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.work_dir = os.path.join(self.base_dir, "work_dir")
        self.uploads_dir = os.path.join(self.work_dir, "uploads")
        self.processed_dir = os.path.join(self.work_dir, "processed")

        self._init_directories()

        # --------------------------------------------------------------------
        # --------------connect es, drop_old means build new one--------------
        # --------------------------------------------------------------------

        es_url = config.get('rag', 'es_url')
        index_name = config.get('rag', 'index_name')
        self.milvus_address = config.get('rag', 'milvus_url')

        self.elastic_search = ElasticSearch(
            elasticsearch_url=es_url,
            index_name=index_name)

        # --------------------------------------------------------------------
        # ------------------------connect pdf_server--------------------------
        # --------------------------------------------------------------------

        pdf_server_bind_port = config.get('ocr', 'pdf_server_bind_port')
        pdf_server_url = f"http://127.0.0.1:{pdf_server_bind_port}"
        self.pdf_server = PdfClient(pdf_server_url)
        self.pdf_status = self.pdf_server.check_health()
        if not self.pdf_status:
            logger.warning("Health check failed. The PDF-ocr server at"
                           " '{}' is not responding as expected.".format(pdf_server_url))
        else:
            logger.debug("Health check succeeded. The PDF-ocr server at"
                         " '{}' is responding as expected.".format(pdf_server_url))

        self.file_processor = DocumentProcessor()
        self.feature_db = FeatureDataBase()

    def _init_directories(self):

        directories = [
            self.work_dir,
            self.uploads_dir,
            self.processed_dir
        ]

        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
            else:
                logger.debug(f"Directory already exists: {directory}")

    def _get_knowledge_dirs(self, knowledge_id: str,
                            mode: str) -> tuple:
        if mode not in ["create", "query"]:
            raise ValueError("mode must be either 'create' or 'query'")

        knowledge_id_dir_uploads = os.path.join(self.uploads_dir, knowledge_id)
        knowledge_id_dir_processed = os.path.join(self.processed_dir, knowledge_id)

        if mode == 'create':
            os.makedirs(knowledge_id_dir_uploads, exist_ok=True)
            os.makedirs(knowledge_id_dir_processed, exist_ok=True)

        return knowledge_id_dir_uploads, knowledge_id_dir_processed

    def _ensure_collections(self):

        if not self.milvus.collection_exists(self.milvus.collection_name):
            self.milvus.create_collection(self.milvus.collection_name)
            logger.info(f"Milvus collection '{self.milvus.collection_name}' created.")

        if not self.elastic_search.index_exists(self.elastic_search.index_name):
            self.elastic_search.create_index(self.elastic_search.index_name)
            logger.info(f"ElasticSearch index '{self.elastic_search.index_name}' created.")

    def create_knowledge_base(self, user_id: str,
                              name: str,
                              vector_model_url: Optional[str] = None,
                              vector_model_name: Optional[str] = None) -> str:

        logger.info(f"Using user id:{user_id} and knowledgebase name:{name} to create a knowledge base.")
        user_id_hashi = hashlib.sha256(f"{user_id}".encode()).hexdigest()[0:16]
        kb_id = "kb_" + f"{user_id_hashi}" + "_" + hashlib.sha256(f"{name}".encode()).hexdigest()[0:16]

        milvus_is_new = None
        if vector_model_url and vector_model_name:
            try:
                embedding_server = EmbedderClient(vector_model_url)
                res = embedding_server.v1_embeddings(
                    model=vector_model_name,
                    input=["test"]
                )

                logger.debug(
                    f"Embedding server at '{vector_model_url}' is responding as expected. Creating knowledge base with id:{kb_id} in milvus.")

                collection_name = kb_id

                connection_args = {
                    "address": self.milvus_address
                }

                dim = np.array(res.get("data", [])[0].get("embedding", [])).shape[0]

                self.milvus = Milvus(
                    embedding_server=embedding_server,
                    dim=dim,
                    collection_name=collection_name,
                    connection_args=connection_args,
                    consistency_level='Strong',
                    index_params=None,
                    search_params=None,
                    drop_old=False,
                    auto_id=False)

                if self.milvus.is_new_collection:
                    logger.info(f"Knowledge base created with id: {collection_name} in milvus.")
                    milvus_is_new = True
                else:
                    logger.info(f"Knowledge base already exists with id: {collection_name} in milvus.")
                    milvus_is_new = False

            except Exception as e:
                error_msg = f"Failed to connect to embedding server at '{vector_model_url}': {str(e)}"
                logger.error(error_msg)
                raise Exception(error_msg)

        try:
            logger.debug(f"Creating knowledge base with id:{kb_id} in elasticsearch.")
            es_knowledge_database_ids = self.elastic_search.list_knowledge_database_id()

            if kb_id not in es_knowledge_database_ids:
                logger.info(f"Knowledge base created with id: {kb_id} in elasticsearch.")
                es_is_new = True
                self.elastic_search.add_meta_info_in_kb(kb_id)

            else:
                logger.info(f"Knowledge base already exists with id: {kb_id} in elasticsearch.")
                es_is_new = False

        except Exception as e:
            error_msg = f"Failed to create knowledge base in elasticsearch: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

        logger.info(f"Creating knowledge base directories for knowledge_base_id: {kb_id}")
        knowledge_id_dir_uploads, knowledge_id_dir_processed = self._get_knowledge_dirs(
            knowledge_id=kb_id,
            mode='create')

        file_status_dir = os.path.join(knowledge_id_dir_uploads, 'file_status')

        os.makedirs(knowledge_id_dir_uploads, exist_ok=True)
        os.makedirs(knowledge_id_dir_processed, exist_ok=True)
        os.makedirs(file_status_dir, exist_ok=True)

        return kb_id, milvus_is_new, es_is_new

    def add_files(self,
                  files,
                  work_dir: str,
                  knowledge_base_id: str,
                  is_md_splitter: Optional[bool] = False,
                  vector_model_url: Optional[str] = None,
                  vector_model_name: Optional[str] = None,
                  chunk_size: Optional[int] = None
                  ):
        milvus_store = None
        if vector_model_url and vector_model_name:
            try:
                embedding_server = EmbedderClient(vector_model_url)
                res = embedding_server.v1_embeddings(
                    model=vector_model_name,
                    input=["test"]
                )

                connection_args = {
                    "address": self.milvus_address
                }

                milvus_store = Milvus(
                    embedding_server=embedding_server,
                    collection_name=knowledge_base_id,
                    connection_args=connection_args
                )

            except Exception as e:
                logger.error(f"Failed to initialize Milvus connection: {str(e)}")
                raise

        if not files:
            logger.warning("No files to insert.")
            return [], []

        if not milvus_store:
            logger.debug(f"No Milvus store initialized, proceeding without vector storage.")

        else:
            if self.pdf_status:
                self.feature_db.preprocess(
                    files=files,
                    work_dir=work_dir,
                    file_opr=self.file_processor,
                    is_md_splitter=is_md_splitter,
                    server=self.pdf_server)
            else:
                self.feature_db.preprocess(
                    files=files,
                    work_dir=work_dir,
                    file_opr=self.file_processor)
            logger.debug(f"Preprocess finish, beginning to insert text data into the database!")

            chunk_size_overlap = None if not chunk_size else 0.2 * chunk_size
            successful_files, failed_files = self.feature_db.build_database(
                files=files,
                file_opr=self.file_processor,
                knowledge_base_id=knowledge_base_id,
                is_md_splitter=is_md_splitter,
                elastic_search=self.elastic_search,
                milvus=milvus_store,
                vector_model_name=vector_model_name,
                chunk_size=chunk_size,
                chunk_size_overlap=chunk_size_overlap
            )

            file_ids = [f"{file.file_id}" for file in files if file.status]

            logger.info(f"Inserted files into Knowledge Base {knowledge_base_id}: {file_ids}")
            return successful_files, failed_files

    def update_chunk(
            self,
            knowledge_base_id: str,
            file_id: str,
            chunk_id: str,
            text: str,
            vector_model_url: str = None,
            vector_model_name: str = None,
            **kwargs: Any,
    ) -> bool:
        """
        Update document content, attempting to update both Milvus and ES simultaneously.
        If Milvus update conditions are not met, only ES will be updated.

        Args:
            knowledge_base_id: Knowledge base ID
            file_id: File ID
            chunk_id: Chunk ID
            text: New text content
            vector_model_url: Vector model service URL
            vector_model_name: Vector model name
            **kwargs: Additional parameters

        Returns:
            bool: Whether the update was successful
        """
        milvus_msg = None
        es_msg = None

        if vector_model_url and vector_model_name:
            try:
                embedding_server = EmbedderClient(vector_model_url)
                connection_args = {
                    "address": self.milvus_address
                }

                milvus_store = Milvus(
                    embedding_server=embedding_server,
                    collection_name=knowledge_base_id,
                    connection_args=connection_args
                )

                milvus_msg = milvus_store.update_chunk(
                    knowledge_base_id=knowledge_base_id,
                    file_id=file_id,
                    chunk_id=chunk_id,
                    text=text,
                    vector_model_name=vector_model_name,
                    **kwargs
                )
            except Exception as e:
                logger.warning(f"Failed to update chunk in milvus : {str(e)}")
                milvus_msg = f"Failed to update chunk in milvus : {str(e)}"
        else:
            logger.warning("Vector model URL or name not provided, skipping milvus update")
            milvus_msg = "Vector model URL or name not provided, skipping milvus update"

        try:
            es_msg = self.elastic_search.update_chunk(
                knowledge_base_id=knowledge_base_id,
                file_id=file_id,
                chunk_id=chunk_id,
                text=text,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to update chunk in elasticsearch: {str(e)}")
            es_msg = f"Failed to update chunk in elasticsearch: {str(e)}"

        return es_msg, milvus_msg

    def delete(
            self,
            knowledge_base_id: str,
            file_id: Optional[str] = None,
            chunk_id: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """
        Args:
            knowledge_base_id: Knowledge base ID (required)
            file_id: File ID (optional)
            chunk_id: Chunk ID (optional, must be provided together with file_id if used)

        Raises:
            ValueError: Raises exception when knowledge_base_id is not provided
        """
        if not knowledge_base_id:
            error_msg = "knowledge_base_id must be provided"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if chunk_id and not file_id:
            error_msg = "file_id must be provided when chunk_id is specified"
            logger.error(error_msg)
            raise ValueError(error_msg)

        delete_info = f'Deleting knowledge base content, knowledgebase_id: 【{knowledge_base_id}】'
        if file_id:
            delete_info += f', file_id: 【{file_id}】'
        if chunk_id:
            delete_info += f', chunk_id: 【{chunk_id}】'
        logger.info(delete_info)

        drop_knowledge_base = False
        try:
            es_knowledge_database_ids = self.elastic_search.list_knowledge_database_id()
            if knowledge_base_id in es_knowledge_database_ids:
                es_res = self.elastic_search.delete(knowledge_base_id, file_id, chunk_id)
                if not file_id and not chunk_id:
                    drop_knowledge_base = True
                    es_message = f"knowledge base id: {knowledge_base_id} has been dropped in es"
                    logger.info(f"knowledge base id: {knowledge_base_id} has been dropped in es")
                else:
                    es_message = f"Es deleted {es_res['deleted']} documents in {knowledge_base_id}"
                    logger.info(f"Es deleted {es_res['deleted']} documents in {knowledge_base_id}")
            else:
                logger.warning(f"knowledge base id: {knowledge_base_id} not found in es")
                es_message = f"knowledge base id: {knowledge_base_id} not found in es"

        except Exception as e:
            logger.error(f"Error during ES deletion: {str(e)}")
            es_message = f"Failed to delete from ES: {str(e)}"

        try:
            kb_exits = False
            milvus_host, milvus_port = self.milvus_address.split(":")
            connections.connect(
                alias="default",
                host=milvus_host,
                port=milvus_port
            )
            if utility.has_collection(knowledge_base_id, using="default"):
                kb_exits = True
                connection_args = {"address": self.milvus_address}
                milvus_temp = Milvus(
                    collection_name=knowledge_base_id,
                    connection_args=connection_args
                )
            # Case 1: Only knowledge_base_id - drop the entire collection
            if knowledge_base_id and not file_id and not chunk_id:
                if kb_exits:
                    collection = Collection(knowledge_base_id, using="default")
                    collection.drop()
                    logger.info(f"knowledge base id: {knowledge_base_id} has been dropped in milvus")
                    milvus_message = f"knowledge base id: {knowledge_base_id} has been dropped in milvus"
                else:
                    logger.warning(f"knowledge base id: {knowledge_base_id} not found in milvus")
                    milvus_message = f"knowledge base id: {knowledge_base_id} not found in milvus"

            # Case 2 & 3
            elif knowledge_base_id and file_id:
                if kb_exits:
                    milvus_res = milvus_temp.delete(
                        knowledge_base_id=knowledge_base_id,
                        file_id=file_id,
                        chunk_id=chunk_id
                    )
                    milvus_message = f"Milvus deleted {milvus_res.delete_count} documents in {knowledge_base_id}"
                else:
                    logger.warning(f"knowledge base id: {knowledge_base_id} not found in milvus")
                    milvus_message = f"knowledge base id: {knowledge_base_id} not found in milvus"

        except Exception as e:
            logger.error(f"Error during Milvus deletion: {str(e)}")
            milvus_message = f"Failed to delete from Milvus: {str(e)}"

        # delete the Knowledge base directories
        if drop_knowledge_base:
            knowledge_id_dir_uploads, knowledge_id_dir_processed = self._get_knowledge_dirs(
                knowledge_id=knowledge_base_id,
                mode='query')
            if os.path.exists(knowledge_id_dir_uploads):
                logger.info(f"Removing uploads directory for knowledge_base_id: {knowledge_base_id}")
                shutil.rmtree(knowledge_id_dir_uploads)

            if os.path.exists(knowledge_id_dir_processed):
                logger.info(f"Removing processed directory for knowledge_base_id: {knowledge_base_id}")
                shutil.rmtree(knowledge_id_dir_processed)

        return {
            "es_message": es_message,
            "milvus_message": milvus_message
        }

    def weighted_reciprocal_rank(self, es_docs, vector_docs, weights: List[float] = None):
        # Create a union of all unique documents in the input doc_lists
        if weights:
            if not isinstance(weights, list) or not all(isinstance(w, float) for w in weights):
                weights = [0.5, 0.5]
            if len(weights) != 2:
                weights = [0.5, 0.5]
            if sum(weights) != 1.0:
                weights = [0.5, 0.5]
        else:
            weights = [0.5, 0.5]
        all_documents = set()

        for vector_doc in vector_docs:
            all_documents.add(vector_doc.page_content)
        for es_doc in es_docs:
            all_documents.add(es_doc.page_content)

        rrf_score_dic = {doc: 0.0 for doc in all_documents}

        for rank, vector_doc in enumerate(vector_docs, start=1):
            rrf_score = weights[1] * (1 / (rank + 60))
            rrf_score_dic[vector_doc.page_content] += rrf_score
        for rank, es_doc in enumerate(es_docs, start=1):
            rrf_score = weights[0] * (1 / (rank + 60))
            rrf_score_dic[es_doc.page_content] += rrf_score

        sorted_documents = sorted(rrf_score_dic.keys(), key=lambda x: rrf_score_dic[x], reverse=True)

        # Map the sorted page_content back to the original document objects
        page_content_to_doc_map = {}
        for doc in es_docs:
            page_content_to_doc_map[doc.page_content] = doc
        for doc in vector_docs:
            page_content_to_doc_map[doc.page_content] = doc

        sorted_docs = [page_content_to_doc_map[page_content] for page_content in sorted_documents]

        return sorted_docs

    def remove_duplicates(self, sorted_docs):

        seen = set()
        unique_docs = []

        for document in sorted_docs:
            identifier = (
                document.metadata.get('knowledge_base_id', ''),
                document.metadata.get('file_id', ''),
                document.metadata.get('chunk_id', '')
            )
            if identifier not in seen:
                seen.add(identifier)
                unique_docs.append(document)

        return unique_docs

    def search(self,
               knowledge_base_id: List[str],
               query: str,
               file_ids: List[str],
               vector_model_name: Optional[str] = None,
               vector_model_url: Optional[str] = None,
               search_mode: SearchMode = SearchMode.HYBRID,
               using_rerank: bool = False,
               reranker_model_name: str = None,
               reranker_model_url: Optional[str] = None,
               top_k: int = 10,
               threshold: float = None,
               weights: List[float] = None) -> List[SearchResult]:

        """
        Knowledge base search method

        Args:
            knowledge_base_id: Knowledge base ID
            query: Search query text
            vector_model_name: Vector model name
            vector_model_url: Vector model service URL
            search_mode: Search mode
            using_rerank: Whether to use reranking
            reranker_model: Reranker model URL
            top_k: Number of results to return
            threshold: Minimum similarity score threshold for filtering results.
                Results with scores below this value will be filtered out. Defaults to None.
        """
        if not query or not query.strip():
            logger.warning("Empty query provided, returning empty results")
            return [], [], "Empty query provided."

        if not knowledge_base_id or not file_ids:
            logger.warning("Missing required parameters: knowledge_base_id or file_ids")
            return [], [], "Missing required parameters."

        # --------------------------------------------------------------------
        # ----------------------Connect embedding_server----------------------
        # --------------------------------------------------------------------
        embedding_server_connectivity = False
        if vector_model_name and vector_model_url:
            embedding_server = EmbedderClient(vector_model_url)
            try:
                embedding_res = embedding_server.v1_embeddings(model=vector_model_name, input=[query])

                if embedding_res:
                    data = embedding_res.get("data", [])
                    if data:
                        for i, embedding_item in enumerate(data):
                            query_embedding = embedding_item.get("embedding", [])

                logger.debug(
                    f"'{vector_model_name}' embedding server at '{vector_model_url}' is responding as expected.")
                embedding_server_connectivity = True
            except Exception as e:
                logger.error(f"Embedding server is not responding: {str(e)}")

        # --------------------------------------------------------------------
        # ----------------------Connect reranker_server-----------------------
        # --------------------------------------------------------------------
        reranker_server_connectivity = False
        if using_rerank and reranker_model_url:
            try:
                reranker = RerankerClient(reranker_model_url)
                reranker_res = reranker.v1_rerank(
                    query='What is the capital of France?',
                    documents=[
                        "Paris",
                        "London",
                        "Berlin"
                    ],
                    model=reranker_model_name
                )
                logger.debug(
                    f"'{reranker_model_name}' reranker server at '{reranker_model_url}' is responding as expected.")
                reranker_server_connectivity = True
            except Exception as e:
                logger.error(f"Reranker server server is not responding: {str(e)}")

        # --------------------------------------------------------------------
        # --------------------------Get Search Results------------------------
        # --------------------------------------------------------------------

        def get_milvus_results():
            if not embedding_server_connectivity:
                logger.error("Cannot perform vector search: Embedding server is not responding")
                return []
            try:
                connection_args = {
                    "address": self.milvus_address
                }
                vector_res = []
                for kb_id in knowledge_base_id:
                    milvus = Milvus(
                        embedding_server=embedding_server,
                        collection_name=kb_id,
                        connection_args=connection_args,
                        consistency_level='Strong'
                    )
                    vector_results = milvus.search(query_embedding=query_embedding,
                                                   knowledge_base_id=kb_id,
                                                   file_ids=file_ids,
                                                   vector_model_name=vector_model_name,
                                                   k=top_k,
                                                   threshold=threshold)
                    vector_res += vector_results

                sorted_results = sorted(vector_res, key=lambda x: x.metadata['relevance_score'], reverse=True)[:top_k]
                return sorted_results

            except Exception as e:
                logger.error(f"Error in vector search: {str(e)}")
                return []

        def get_es_results():
            try:
                keywords = jieba.analyse.extract_tags(query, topK=10, withWeight=False)
                logger.info('jieba search keywords:{}'.format(keywords))

                es_res = []
                for kb_id in knowledge_base_id:
                    es_results = self.elastic_search.similarity_search_with_score(
                        query=query,
                        knowledge_base_id=kb_id,
                        file_ids=file_ids,
                        keywords=keywords,
                        k=top_k
                    )
                    es_res += es_results

                es_res = sorted(es_res, key=lambda x: x.metadata['relevance_score'], reverse=True)[:top_k]
                return es_res, keywords

            except Exception as e:
                logger.error(f"Error in fulltext search: {str(e)}")
                return [], []

        # --------------------------------------------------------------------
        # ------------------------------Rerank--------------------------------
        # --------------------------------------------------------------------

        def apply_rerank(results: List[SearchResult]) -> List[SearchResult]:
            if len(results) <= 1:
                logger.debug(f"Skip reranking since results count ({len(results)}) ")
                return results
            try:
                documents = [doc.page_content for doc in results]
                reranker_outputs = reranker.v1_rerank(
                    query=query,
                    documents=documents,
                    model=reranker_model_name
                )

                reranker_results = []
                for reranker_output in reranker_outputs:
                    index = reranker_output['index']
                    reranker_results.append(results[index])

                return reranker_results
            except Exception as e:
                logger.error(f"Error in reranking: {str(e)}")
                return results

        # --------------------------------------------------------------------
        # -----------------------------Workflow-------------------------------
        # --------------------------------------------------------------------
        try:
            if len(knowledge_base_id) == 1:
                logger.debug(
                    f"Searching in knowledge database 【{knowledge_base_id}】 with Query: 【{query}】"
                )
            else:
                logger.debug(
                    f"Searching in multiple knowledge databases 【{' | '.join(knowledge_base_id)}】 with Query: 【{query}】"
                )

            keywords = []
            results = []
            if search_mode == SearchMode.SEMANTIC:
                logger.debug(
                    f"Using semantic search mode"
                )
                results = get_milvus_results()

            elif search_mode == SearchMode.FULLTEXT:
                logger.debug(
                    f"Using fulltext search mode."
                )
                results, keywords = get_es_results()

            elif search_mode == SearchMode.HYBRID:
                logger.debug(
                    f"Using hybird search mode"
                )
                try:
                    from concurrent.futures import ThreadPoolExecutor

                    with ThreadPoolExecutor(max_workers=2) as executor:
                        future_vector = executor.submit(get_milvus_results)
                        future_es = executor.submit(get_es_results)

                    try:
                        vector_results = future_vector.result(timeout=10)
                    except TimeoutError:
                        logger.error("Vector search timeout")
                        vector_results = []

                    try:
                        es_results, keywords = future_es.result(timeout=10)
                    except TimeoutError:
                        logger.error("ES search timeout")
                        es_results = []

                    if vector_results and es_results:
                        results = self.remove_duplicates(self.weighted_reciprocal_rank(es_results,
                                                                                       vector_results,
                                                                                       weights))
                    elif vector_results:
                        results = vector_results
                        logger.warning("Fulltext search returned empty results. Using vector search results.")
                    elif es_results:
                        results = es_results
                        logger.warning("Vector search returned empty results. Using fulltext search results.")
                    else:
                        results = []
                        logger.warning("Both vector and fulltext search returned empty results.")

                except Exception as e:
                    logger.error(f"Error in hybrid search with ThreadPoolExecutor: {str(e)}")
                    try:
                        vector_results = get_milvus_results()
                        es_results, keywords = get_es_results()
                        if vector_results and es_results:
                            results = self.remove_duplicates(self.weighted_reciprocal_rank(es_results,
                                                                                           vector_results,
                                                                                           weights))
                        elif vector_results:
                            results = vector_results
                            logger.warning("Fulltext search returned empty results. Using vector search results.")
                        elif es_results:
                            results = es_results
                            logger.warning("Vector search returned empty results. Using fulltext search results.")
                        else:
                            results = []
                            logger.warning("Both vector and fulltext search returned empty results in fallback.")
                    except Exception as e:
                        logger.error(f"Error in fallback hybrid search: {str(e)}")
                        results = []

            else:
                raise ValueError(f"Unsupported search mode: {search_mode}")

            if using_rerank and reranker_server_connectivity and results:
                logger.debug(f"Using rerank")
                results = apply_rerank(results)

            if results:
                results = results[:top_k]

            return results, keywords, None

        except Exception as e:
            logger.error(f"Error in knowledge base search: {str(e)}")
            return results, keywords, str(e)

    def obtain_knowledge_base_info(self, user_id: str, is_admin: bool = False):

        milvus_kb_ids = []
        es_kb_ids = []
        user_id_hashi = hashlib.sha256(f"{user_id}".encode()).hexdigest()[0:16]
        try:
            uri = f"http://{self.milvus_address}"
            connections.connect(alias="default", uri=uri)

            all_collections = utility.list_collections()

            if is_admin:
                milvus_kb_ids = all_collections
            else:
                milvus_kb_ids = [kb_id for kb_id in all_collections
                                 if kb_id.startswith(f"kb_{user_id_hashi}")]

            logger.debug(f"Retrieved knowledge_base_ids from Milvus: {milvus_kb_ids}")

        except MilvusException as me:
            logger.error(f"Failed to connect or operate with Milvus: {me}")
            raise ConnectionError(f"Failed to connect or operate with Milvus: {me}")
        except Exception as e:
            logger.error(f"An unknown error occurred while connecting to Milvus: {e}")
            raise ConnectionError(f"An unknown error occurred while connecting to Milvus: {e}")
        finally:
            try:
                connections.disconnect(alias="default")
            except Exception as e:
                logger.warning(f"Error occurred while disconnecting from Milvus: {e}")

        es_knowledge_database_ids = self.elastic_search.list_knowledge_database_id()

        if is_admin:
            es_kb_ids = es_knowledge_database_ids
        else:
            es_kb_ids = [kb_id for kb_id in es_knowledge_database_ids
                         if kb_id.startswith(f"kb_{user_id_hashi}")]

        logger.debug(f"Retrieved knowledge_base_ids from ES: {es_kb_ids}")

        return {
            "elasticsearch_knowledge_bases_ids": es_kb_ids,
            "milvus_knowledge_bases_ids": milvus_kb_ids,
        }

    def list_files_with_status(self, knowledge_base_id: str, is_admin: bool = False):

        es_knowledge_database_ids = self.elastic_search.list_knowledge_database_id()
        if knowledge_base_id in es_knowledge_database_ids:
            status_dict = {}

            knowledge_base_dir = os.path.join(self.uploads_dir, knowledge_base_id)

            if not os.path.exists(knowledge_base_dir):
                logger.error(
                    f"Knowledge base not found: {knowledge_base_id}, a knowledge base will be created automatically after file upload")
                raise FileNotFoundError(
                    f"Knowledge base not found: {knowledge_base_id}, a knowledge base will be created automatically after file upload")

            file_status_dir = os.path.join(knowledge_base_dir, 'file_status')

            if not os.path.exists(file_status_dir):
                logger.error(f"Status directory not found in Knowledge base id: {knowledge_base_dir}")
                raise FileNotFoundError(f"Status directory not found in Knowledge base id: {knowledge_base_dir}")

            for filename in os.listdir(file_status_dir):
                if filename.endswith('.status'):
                    file_path = os.path.join(file_status_dir, filename)
                    try:
                        with open(file_path, 'r') as f:
                            status_content = f.read().strip()

                            file_id = filename.split('_')[-1].split('.')[0]
                            status_dict[file_id] = status_content
                    except Exception as e:
                        logger.error(f"Error reading status file {filename}: {e}")
                        continue
            return status_dict
        else:
            raise FileNotFoundError(f"Knowledge base not found: {knowledge_base_id}")

    def list_chunks_info(self, knowledge_base_id: str,
                         file_id: str,
                         chunk_id: str = None,
                         size: int = None,
                         offset: int = None,
                         use_embedding_id_as_id: Optional[bool] = True):
        chunks_info = self.elastic_search.list_chunks_info(knowledge_base_id=knowledge_base_id,
                                                           file_id=file_id,
                                                           chunk_id=chunk_id,
                                                           page_size=size,
                                                           page_num=offset,
                                                           use_embedding_id_as_id=use_embedding_id_as_id)
        return chunks_info

    def create_chunk(self, knowledge_base_id: str,
                     text: str,
                     file_id: str = None,
                     vector_model_url: str = None,
                     vector_model_name: str = None):
        if not isinstance(text, str) or not text.strip():
            return {
                "file_id": None,
                "chunk_id": None,
                "es_msg": "Invalid text input",
                "milvus_msg": "Invalid text input"
            }

        if not isinstance(knowledge_base_id, str) or not knowledge_base_id.strip():
            return {
                "file_id": None,
                "chunk_id": None,
                "es_msg": "Invalid knowledge base ID",
                "milvus_msg": "Invalid knowledge base ID"
            }

        logger.info(f"Starting chunk creation for knowledge base: {knowledge_base_id}")
        milvus_msg = None
        es_msg = None

        hash_value = hashlib.sha256(text.encode('utf-8')).hexdigest()
        chunk_id = hash_value[:16]

        if file_id is None:
            file_id = hash_value[16:32]

        logger.debug(f"Generated file_id: {file_id}, chunk_id: {chunk_id}")

        try:
            es_knowledge_database_ids = self.elastic_search.list_knowledge_database_id()
            if knowledge_base_id in es_knowledge_database_ids:
                logger.info(
                    f"Found knowledge base {knowledge_base_id} in ES, starting to create chunk and status file!")

                status_dir = os.path.join(self.uploads_dir, knowledge_base_id, 'file_status')
                handle_status_file(status_dir,
                                   f"{knowledge_base_id}_{file_id}",
                                   operation='write',
                                   status='pending')
                try:
                    ids, failed_files = self.elastic_search.add_texts(
                        knowledge_base_id=knowledge_base_id,
                        texts=[text],
                        file_id_list=[file_id],
                        chunks_ids_list=[chunk_id]
                    )
                    if failed_files:
                        es_msg = f"Partial failure in ES: Some files failed to be created: {failed_files}"
                        handle_status_file(status_dir,
                                           f"{knowledge_base_id}_{file_id}",
                                           operation='write',
                                           status='error')
                    else:
                        es_msg = "success"
                        handle_status_file(status_dir,
                                           f"{knowledge_base_id}_{file_id}",
                                           operation='write',
                                           status='success')
                except Exception as e:
                    logger.error(f"Error in ES create_chunk: {str(e)}")
                    es_msg = f"Failed to create chunk in ES: {str(e)}"
                    handle_status_file(status_dir,
                                       f"{knowledge_base_id}_{file_id}",
                                       operation='write',
                                       status='error')
            else:
                logger.warning(f"Knowledge base id: {knowledge_base_id} not found in ES!")
                es_msg = f"Knowledge base id: {knowledge_base_id} not found in ES!"
        except Exception as e:
            logger.error(f"Error checking knowledge base in ES: {str(e)}")
            es_msg = f"Error checking knowledge base in ES: {str(e)}"

        if vector_model_url and vector_model_name:
            try:
                uri = f"http://{self.milvus_address}"
                connections.connect(alias="default", uri=uri)

                collections = utility.list_collections()
                if knowledge_base_id in collections:
                    logger.info(f"Found knowledge base {knowledge_base_id} in Milvus, starting to create chunk")
                    try:
                        embedding_server = EmbedderClient(vector_model_url)
                        connection_args = {
                            "address": self.milvus_address
                        }

                        milvus_store = Milvus(
                            embedding_server=embedding_server,
                            collection_name=knowledge_base_id,
                            connection_args=connection_args
                        )

                        success_ids, all_ids = milvus_store.add_texts(
                            knowledge_base_id=knowledge_base_id,
                            vector_model_name=vector_model_name,
                            texts=[text],
                            file_ids=[file_id],
                            chunk_ids=[chunk_id],
                        )

                        if not success_ids:
                            milvus_msg = "Failed to add texts: no documents were successfully inserted"
                        else:
                            milvus_msg = "success"

                    except Exception as e:
                        logger.error(f"Failed to operate with Milvus: {str(e)}")
                        milvus_msg = f"Failed to operate with Milvus: {str(e)}"
                else:
                    logger.warning(f"Knowledge base id: {knowledge_base_id} not found in Milvus!")
                    milvus_msg = f"Knowledge base id: {knowledge_base_id} not found in Milvus!"

            except MilvusException as me:
                logger.error(f"Failed to connect or operate with Milvus: {me}")
                milvus_msg = f"Failed to connect or operate with Milvus: {str(me)}"
            except Exception as e:
                logger.error(f"An unknown error occurred while connecting to Milvus: {e}")
                milvus_msg = f"An unknown error occurred while connecting to Milvus: {str(e)}"
            finally:
                try:
                    connections.disconnect(alias="default")
                except Exception as e:
                    logger.warning(f"Error occurred while disconnecting from Milvus: {e}")
        else:
            milvus_msg = "No vector model or url provided"

        logger.info(f"Chunk creation completed. Results: ES={es_msg}, Milvus={milvus_msg}")
        return {
            "file_id": file_id,
            "chunk_id": chunk_id,
            "es_msg": es_msg or "not processed",
            "milvus_msg": milvus_msg or "not processed"
        }