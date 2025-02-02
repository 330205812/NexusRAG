from __future__ import annotations
import numpy as np
from loguru import logger
from typing import Any, List, Optional, Tuple
from uuid import uuid4
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

try:
    from pymilvus import Collection, utility, MilvusException, connections, CollectionSchema, DataType, FieldSchema
except ImportError:
    raise ValueError(
        "Could not import pymilvus python package. "
        "Please install it with `pip install pymilvus`."
    )
from langchain_community.vectorstores.utils import maximal_marginal_relevance

DEFAULT_MILVUS_CONNECTION = {
    "host": "localhost",
    "port": "19530",
    "user": "",
    "password": "",
    "secure": False,
}


class Milvus():

    def __init__(
            self,
            embedding_server=None,
            embedding_url: str = None,
            dim: int = None,
            collection_name: str = "LangChainCollection",
            collection_description: str = "",
            collection_properties: Optional[dict[str, Any]] = None,
            connection_args: Optional[dict[str, Any]] = None,
            consistency_level: str = "Session",
            index_params: Optional[dict] = None,
            search_params: Optional[dict] = None,
            drop_old: Optional[bool] = False,
            auto_id: bool = False,
            *,
            primary_field: str = "id",
            text_field: str = "text",
            vector_field: str = "vector",
            metadata_field: Optional[str] = None,
            partition_key_field: Optional[str] = None,
            partition_names: Optional[list] = None,
            replica_number: int = 1,
            timeout: Optional[float] = None,
    ):
        """
            Initialize the Milvus vector store.
        """

        # Default search params when one is not provided.
        self.default_search_params = {
            "IVF_FLAT": {"metric_type": "L2", "params": {"nprobe": 10}},
            "IVF_SQ8": {"metric_type": "L2", "params": {"nprobe": 10}},
            "IVF_PQ": {"metric_type": "L2", "params": {"nprobe": 10}},
            "HNSW": {"metric_type": "L2", "params": {"ef": 10}},
            "RHNSW_FLAT": {"metric_type": "L2", "params": {"ef": 10}},
            "RHNSW_SQ": {"metric_type": "L2", "params": {"ef": 10}},
            "RHNSW_PQ": {"metric_type": "L2", "params": {"ef": 10}},
            "IVF_HNSW": {"metric_type": "L2", "params": {"nprobe": 10, "ef": 10}},
            "ANNOY": {"metric_type": "L2", "params": {"search_k": 10}},
            "SCANN": {"metric_type": "L2", "params": {"search_k": 10}},
            "AUTOINDEX": {"metric_type": "L2", "params": {}},
            "GPU_CAGRA": {
                "metric_type": "L2",
                "params": {
                    "itopk_size": 128,
                    "search_width": 4,
                    "min_iterations": 0,
                    "max_iterations": 0,
                    "team_size": 0,
                },
            },
            "GPU_IVF_FLAT": {"metric_type": "L2", "params": {"nprobe": 10}},
            "GPU_IVF_PQ": {"metric_type": "L2", "params": {"nprobe": 10}},
        }
        self.embedding_func = None
        self.dim = None
        if embedding_server:
            self.embedding_func = embedding_server
            self.dim = dim

        self.collection_name = collection_name
        self.collection_description = collection_description
        self.collection_properties = collection_properties
        self.index_params = index_params
        self.search_params = search_params
        self.consistency_level = consistency_level
        self.auto_id = auto_id

        # In order for a collection to be compatible, pk needs to be varchar
        self._primary_field = primary_field
        # In order for compatibility, the text field will need to be called "text"
        self._text_field = text_field
        # In order for compatibility, the vector field needs to be called "vector"
        self._vector_field = vector_field
        self._metadata_field = metadata_field
        self._partition_key_field = partition_key_field
        self.fields: list[str] = []
        self.partition_names = partition_names
        self.replica_number = replica_number
        self.timeout = timeout
        self.is_new_collection = True

        # Create the connection to the server
        if connection_args is None:
            connection_args = DEFAULT_MILVUS_CONNECTION
        self.alias, address = self._create_connection_alias(connection_args)
        self.col: Optional[Collection] = None

        # Grab the existing collection if it exists
        if utility.has_collection(self.collection_name, using=self.alias):
            self.col = Collection(
                self.collection_name,
                using=self.alias,
            )
            if self.collection_properties is not None:
                self.col.set_properties(self.collection_properties)
            self.is_new_collection = False
        # If need to drop old, drop it
        if drop_old and isinstance(self.col, Collection):
            self.col.drop()
            self.col = None

        # Initialize the vector store
        self._init(
            partition_names=partition_names,
            replica_number=replica_number,
            timeout=timeout,
        )
        if self.dim:
            logger.debug(
                f"Milvus initialized with URL:{address} and collection: {self.collection_name}, embedding dimension: {self.dim}")
        else:
            logger.debug(f"Milvus initialized with URL:{address} and collection: {self.collection_name}")

    def __del__(self):

        try:
            if hasattr(self, 'alias'):
                connections.disconnect(alias=self.alias)
        except:
            pass

    def _init(
            self,
            partition_names: Optional[list] = None,
            replica_number: int = 1,
            timeout: Optional[float] = None,
    ) -> None:
        if self.col is None:
            self._create_collection_schema()
        self._extract_fields()
        self._create_index()
        self._create_search_params()
        self._load(
            partition_names=partition_names,
            replica_number=replica_number,
            timeout=timeout,
        )

    def _create_collection_schema(self):
        """Create new collection with schema but without data"""

        schema = CollectionSchema(
            fields=[
                FieldSchema(name=self._primary_field, dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema(name=self._text_field, dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name=self._vector_field, dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="knowledge_base_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="file_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100),
            ],
            description="Collection for storing text and its embeddings"
        )

        self.col = Collection(
            name=self.collection_name,
            schema=schema,
            consistency_level=self.consistency_level,
            using=self.alias,
        )

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_func

    def _create_connection_alias(self, connection_args: dict) -> str:
        """Create the connection to the Milvus server."""

        # Grab the connection arguments that are used for checking existing connection
        address: str = connection_args.get("address", None)
        user = connection_args.get("user", None)

        if address is not None:
            given_address = address
        else:
            given_address = None
            logger.debug("Missing standard address type for reuse attempt")

        # User defaults to empty string when getting connection info
        if user is not None:
            tmp_user = user
        else:
            tmp_user = ""

        # If a valid address was given, then check if a connection exists
        if given_address is not None:
            for con in connections.list_connections():
                addr = connections.get_connection_addr(con[0])
                if (
                        con[1]
                        and ("address" in addr)
                        and (addr["address"] == given_address)
                        and ("user" in addr)
                        and (addr["user"] == tmp_user)
                ):
                    logger.debug(f"Using previous connection: {con[0]}")
                    return con[0], given_address

        # Generate a new connection if one doesn't exist
        alias = uuid4().hex
        try:
            connections.connect(alias=alias, **connection_args)
            logger.debug(f"Created new connection using: {alias}")
            return alias, address
        except MilvusException as e:
            logger.error(f"Failed to create new connection using: {alias}")
            raise e

    def _extract_fields(self) -> None:
        """Grab the existing fields from the Collection"""

        if isinstance(self.col, Collection):
            schema = self.col.schema
            for x in schema.fields:
                self.fields.append(x.name)

    def _get_index(self) -> Optional[dict[str, Any]]:
        """Return the vector index information if it exists"""

        if isinstance(self.col, Collection):
            for x in self.col.indexes:
                if x.field_name == self._vector_field:
                    return x.to_dict()
        return None

    def _create_index(self) -> None:
        """Create a index on the collection"""

        if isinstance(self.col, Collection) and self._get_index() is None:
            try:
                # If no index params, use a default HNSW based one
                if self.index_params is None:
                    self.index_params = {
                        "metric_type": "L2",
                        "index_type": "HNSW",
                        "params": {"M": 8, "efConstruction": 64},
                    }

                try:
                    self.col.create_index(
                        self._vector_field,
                        index_params=self.index_params,
                        using=self.alias,
                    )

                # If default did not work, most likely on Zilliz Cloud
                except MilvusException:
                    # Use AUTOINDEX based index
                    self.index_params = {
                        "metric_type": "L2",
                        "index_type": "AUTOINDEX",
                        "params": {},
                    }
                    self.col.create_index(
                        self._vector_field,
                        index_params=self.index_params,
                        using=self.alias,
                    )
                logger.debug(f"Successfully created an index on collection: {self.collection_name}")

            except MilvusException as e:
                logger.error(
                    f"Failed to create an index on collection: {self.collection_name}"
                )
                raise e

    def _create_search_params(self) -> None:
        """Generate search params based on the current index type"""

        if isinstance(self.col, Collection) and self.search_params is None:
            index = self._get_index()
            if index is not None:
                index_type: str = index["index_param"]["index_type"]
                metric_type: str = index["index_param"]["metric_type"]
                self.search_params = self.default_search_params[index_type]
                self.search_params["metric_type"] = metric_type

    def _load(
            self,
            partition_names: Optional[list] = None,
            replica_number: int = 1,
            timeout: Optional[float] = None,
    ) -> None:
        """Load the collection if available."""
        from pymilvus.client.types import LoadState

        timeout = self.timeout or timeout
        if (
                isinstance(self.col, Collection)
                and self._get_index() is not None
                and utility.load_state(self.collection_name, using=self.alias)
                == LoadState.NotLoad
        ):
            self.col.load(
                partition_names=partition_names,
                replica_number=replica_number,
                timeout=timeout,
            )

    def add_texts(
            self,
            knowledge_base_id: str,
            vector_model_name: str,
            texts: List[str],
            file_ids: List[str],
            chunk_ids: List[str],
            **kwargs: Any,
    ) -> List[str]:
        """
        upsert
        """
        try:
            # embeddings = self.embedding_func.embed_documents(texts)
            embeddings = []

            res = self.embedding_func.v1_embeddings(
                model=vector_model_name,
                input=texts
            )

            if res:
                data = res.get("data", [])
                if data:
                    for i, embedding_item in enumerate(data):
                        embedding = embedding_item.get("embedding", [])
                        embeddings.append(embedding)

            entities = []
            for text, embedding, file_id, chunk_id in zip(texts, embeddings, file_ids, chunk_ids):
                pk_id = f"{knowledge_base_id}_{file_id}_{chunk_id}"
                entity = {
                    "id": pk_id,
                    "vector": embedding,
                    "knowledge_base_id": knowledge_base_id,
                    "file_id": file_id,
                    "chunk_id": chunk_id,
                    "text": text
                }
                entities.append(entity)

            result = self.col.upsert(
                entities,
                partition_name=kwargs.get('partition_name'),
                timeout=kwargs.get('timeout')
            )

            all_ids = [entity["id"] for entity in entities]
            success_ids = result.primary_keys

            return success_ids, all_ids

        except Exception as e:
            logger.error(f"Upsert documents failed: {str(e)}")
            all_ids = [entity["id"] for entity in entities]
            return [], all_ids

    def update_chunk(
            self,
            knowledge_base_id: str,
            file_id: str,
            chunk_id: str,
            text: str = None,
            vector_model_name: str = None,
            **kwargs: Any,
    ) -> str:

        try:
            id = f"{knowledge_base_id}_{file_id}_{chunk_id}"
            expr = f'id == "{id}"'
            results = self.col.query(
                expr=expr,
                output_fields=["id", "vector", "knowledge_base_id", "file_id", "chunk_id", "text"],
                timeout=kwargs.get('timeout')
            )

            if not results:
                logger.error(f"Not found chunk with id in milvus: {id}")
                return f"Not found chunk with id in milvus: {id}"

            entity = {
                "id": id,
                "knowledge_base_id": knowledge_base_id,
                "file_id": file_id,
                "chunk_id": chunk_id
            }

            res = self.embedding_func.v1_embeddings(
                model=vector_model_name,
                input=[text]
            )

            if res and res.get("data"):
                embedding = res["data"][0].get("embedding", [])
                entity["vector"] = embedding
                entity["text"] = text

            result = self.col.upsert(
                [entity],
                partition_name=kwargs.get('partition_name'),
                timeout=kwargs.get('timeout')
            )

            success = len(result.primary_keys) > 0
            if success:
                logger.info(f"Successfully updated chunk with id: {id}")
                return "success"
            else:
                return f"Update chunk failed, please check log."

        except Exception as e:
            logger.error(f"Update chunk failed: {str(e)}")
            return f"Update chunk failed: {str(e)}"

    def search(self,
               query_embedding: List[float],
               knowledge_base_id: str,
               file_ids: List[str],
               k: int = 10,
               threshold: float = None,
               **kwargs: Any,
               ) -> List[Tuple[Document, float]]:
        """
            Search within specified file_ids under a given knowledge_base_id
            Args:
                query: Query text
                knowledge_base_id: Knowledge base ID
                file_ids: List of file IDs
                k: Number of matches to return
            Returns:
                List[Tuple[Document, float]]: List of tuples containing Document objects and similarity scores
        """

        file_conditions = [f'file_id == "{file_id}"' for file_id in file_ids]
        expr = f'knowledge_base_id == "{knowledge_base_id}" && ({" || ".join(file_conditions)})'

        results = self.similarity_search_with_score_by_vector(
            embedding=query_embedding,
            k=k,
            threshold=threshold,
            expr=expr,
            **kwargs
        )

        return results

    def similarity_search(
            self,
            query: str,
            k: int = 4,
            param: Optional[dict] = None,
            expr: Optional[str] = None,
            timeout: Optional[float] = None,
            **kwargs: Any,
    ) -> List[Document]:
        """Perform a similarity search against the query string.

        Args:
            query (str): The text to search.
            k (int, optional): How many results to return. Defaults to 4.
            param (dict, optional): The search params for the index type.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (int, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[Document]: Document results for search.
        """
        if self.col is None:
            logger.debug("No existing collection to search.")
            return []
        timeout = self.timeout or timeout
        res = self.similarity_search_with_score(
            query=query, k=k, param=param, expr=expr, timeout=timeout, **kwargs
        )
        return [doc for doc, _ in res]

    def similarity_search_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            param: Optional[dict] = None,
            expr: Optional[str] = None,
            timeout: Optional[float] = None,
            **kwargs: Any,
    ) -> List[Document]:
        """Perform a similarity search against the query string.

        Args:
            embedding (List[float]): The embedding vector to search.
            k (int, optional): How many results to return. Defaults to 4.
            param (dict, optional): The search params for the index type.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (int, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[Document]: Document results for search.
        """
        if self.col is None:
            logger.debug("No existing collection to search.")
            return []
        timeout = self.timeout or timeout
        res = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, param=param, expr=expr, timeout=timeout, **kwargs
        )
        return [doc for doc, _ in res]

    def similarity_search_with_score(
            self,
            query: str,
            k: int = 4,
            param: Optional[dict] = None,
            expr: Optional[str] = None,
            timeout: Optional[float] = None,
            **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a search on a query string and return results with score.

        For more information about the search parameters, take a look at the pymilvus
        documentation found here:
        https://milvus.io/api-reference/pymilvus/v2.2.6/Collection/search().md

        Args:
            query (str): The text being searched.
            k (int, optional): The amount of results to return. Defaults to 4.
            param (dict): The search params for the specified index.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (float, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[float], List[Tuple[Document, any, any]]:
        """
        if self.col is None:
            logger.debug("No existing collection to search.")
            return []

        # Embed the query text.
        embedding = self.embedding_func.embed_query(query)
        timeout = self.timeout or timeout
        res = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, param=param, expr=expr, timeout=timeout, **kwargs
        )
        return res

    def similarity_search_with_score_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            threshold: Optional[float] = None,
            param: Optional[dict] = None,
            expr: Optional[str] = None,
            timeout: Optional[float] = None,
            **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a search on a query string and return results with score.

        For more information about the search parameters, take a look at the pymilvus
        documentation found here:
        https://milvus.io/api-reference/pymilvus/v2.2.6/Collection/search().md

        Args:
            embedding (List[float]): The embedding vector being searched.
            k (int, optional): The amount of results to return. Defaults to 4.
            threshold (float, optional): The minimum score threshold for results.
                Results with scores below this value will be filtered out. Defaults to None.
            param (dict): The search params for the specified index.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (float, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[Tuple[Document, float]]: Result doc and score.
        """
        if self.col is None:
            logger.debug("No existing collection to search.")
            return []

        if param is None:
            param = self.search_params

        # Determine result metadata fields with PK.
        output_fields = self.fields[:]
        output_fields.remove(self._vector_field)
        timeout = self.timeout or timeout
        # Perform the search.
        res = self.col.search(
            data=[embedding],
            anns_field=self._vector_field,
            param=param,
            limit=k,
            expr=expr,
            output_fields=output_fields,
            timeout=timeout,
            **kwargs,
        )
        # Organize results.
        ret = []
        for result in res[0]:
            if threshold is not None and result.score < threshold:
                continue

            data = {x: result.entity.get(x) for x in output_fields}
            doc = self._parse_document(data, result.score)
            ret.append(doc)

        return ret

    def max_marginal_relevance_search(
            self,
            query: str,
            k: int = 4,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            param: Optional[dict] = None,
            expr: Optional[str] = None,
            timeout: Optional[float] = None,
            **kwargs: Any,
    ) -> List[Document]:
        """Perform a search and return results that are reordered by MMR.

        Args:
            query (str): The text being searched.
            k (int, optional): How many results to give. Defaults to 4.
            fetch_k (int, optional): Total results to select k from.
                Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5
            param (dict, optional): The search params for the specified index.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (float, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.


        Returns:
            List[Document]: Document results for search.
        """
        if self.col is None:
            logger.debug("No existing collection to search.")
            return []

        embedding = self.embedding_func.embed_query(query)
        timeout = self.timeout or timeout
        return self.max_marginal_relevance_search_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            param=param,
            expr=expr,
            timeout=timeout,
            **kwargs,
        )

    def max_marginal_relevance_search_by_vector(
            self,
            embedding: list[float],
            k: int = 4,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            param: Optional[dict] = None,
            expr: Optional[str] = None,
            timeout: Optional[float] = None,
            **kwargs: Any,
    ) -> List[Document]:
        """Perform a search and return results that are reordered by MMR.

        Args:
            embedding (str): The embedding vector being searched.
            k (int, optional): How many results to give. Defaults to 4.
            fetch_k (int, optional): Total results to select k from.
                Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5
            param (dict, optional): The search params for the specified index.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (float, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[Document]: Document results for search.
        """
        if self.col is None:
            logger.debug("No existing collection to search.")
            return []

        if param is None:
            param = self.search_params

        # Determine result metadata fields.
        output_fields = self.fields[:]
        output_fields.remove(self._vector_field)
        timeout = self.timeout or timeout
        # Perform the search.
        res = self.col.search(
            data=[embedding],
            anns_field=self._vector_field,
            param=param,
            limit=fetch_k,
            expr=expr,
            output_fields=output_fields,
            timeout=timeout,
            **kwargs,
        )
        # Organize results.
        ids = []
        documents = []
        scores = []
        for result in res[0]:
            data = {x: result.entity.get(x) for x in output_fields}
            doc = self._parse_document(data)
            documents.append(doc)
            scores.append(result.score)
            ids.append(result.id)

        vectors = self.col.query(
            expr=f"{self._primary_field} in {ids}",
            output_fields=[self._primary_field, self._vector_field],
            timeout=timeout,
        )
        # Reorganize the results from query to match search order.
        vectors = {x[self._primary_field]: x[self._vector_field] for x in vectors}

        ordered_result_embeddings = [vectors[x] for x in ids]

        # Get the new order of results.
        new_ordering = maximal_marginal_relevance(
            np.array(embedding), ordered_result_embeddings, k=k, lambda_mult=lambda_mult
        )

        # Reorder the values and return.
        ret = []
        for x in new_ordering:
            # Function can return -1 index
            if x == -1:
                break
            else:
                ret.append(documents[x])
        return ret

    def delete(
            self,
            knowledge_base_id: str,
            file_id: Optional[str] = None,
            chunk_id: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Delete documents or collection based on knowledge_base_id, file_id, and chunk_id.

        Args:
            knowledge_base_id: The ID of knowledge database. If only this is provided,
                                the entire collection with this name will be dropped.
            file_id: The ID of file
            chunk_id: The ID of chunk
            kwargs: Other parameters in Milvus delete api

        Raises:
            ValueError: If knowledge_base_id is not provided
        """

        try:
            # Case 2: knowledge_base_id and file_id
            if knowledge_base_id and file_id and not chunk_id:
                prefix = f"{knowledge_base_id}_{file_id}_"
                expr = f"id like '{prefix}%'"

            # Case 3: knowledge_base_id, file_id and chunk_id
            elif knowledge_base_id and file_id and chunk_id:
                pk_id = f"{knowledge_base_id}_{file_id}_{chunk_id}"
                expr = f"id == '{pk_id}'"

            result = self.col.delete(expr=expr, **kwargs)
            logger.info(f"Milvus deleted {result.delete_count} documents")
            return result

        except Exception as e:
            logger.error(f"Delete operation failed: {str(e)}")
            raise e

    def _parse_document(self, data: dict, score) -> Document:
        metadata = data.pop(self._metadata_field) if self._metadata_field else data
        metadata['relevance_score'] = score

        return Document(
            page_content=data.pop(self._text_field),
            metadata=metadata
        )

    def get_pks(self, expr: str, **kwargs: Any) -> List[int] | None:
        """Get primary keys with expression

        Args:
            expr: Expression - E.g: "id in [1, 2]", or "title LIKE 'Abc%'"

        Returns:
            List[int]: List of IDs (Primary Keys)
        """

        if self.col is None:
            logger.debug("No existing collection to get pk.")
            return None

        try:
            query_result = self.col.query(
                expr=expr, output_fields=[self._primary_field]
            )
        except MilvusException as exc:
            logger.error("Failed to get ids: %s error: %s", self.collection_name, exc)
            raise exc
        pks = [item.get(self._primary_field) for item in query_result]
        return pks

    def upsert(
            self,
            ids: Optional[List[str]] = None,
            documents: List[Document] | None = None,
            **kwargs: Any,
    ) -> List[str] | None:
        """Update/Insert documents to the vectorstore.

        Args:
            ids: IDs to update - Let's call get_pks to get ids with expression \n
            documents (List[Document]): Documents to add to the vectorstore.

        Returns:
            List[str]: IDs of the added texts.
        """
        if documents is None or len(documents) == 0:
            logger.debug("No documents to upsert.")
            return None

        if ids is not None and len(ids):
            try:
                self.delete(ids=ids)
            except MilvusException:
                pass
        try:
            return self.add_documents(documents=documents, **kwargs)
        except MilvusException as exc:
            logger.error(
                "Failed to upsert entities: %s error: %s", self.collection_name, exc
            )
            raise exc

    def list_files(self) -> Tuple[int, List[str]]:
        """List all unique file IDs in current collection using distinct query."""
        try:
            limit = 10000
            offset = 0
            all_file_ids = []
            while True:
                results = self.col.query(
                    expr="",
                    output_fields=["file_id"],
                    consistency_level=self.consistency_level,
                    distinct_field="file_id",
                    limit=limit,
                    offset=offset
                )

                file_ids_list = [result['file_id'] for result in results]
                all_file_ids.extend(file_ids_list)

                if len(file_ids_list) < limit:
                    break

                offset += limit

            unique_file_ids = list(set(all_file_ids))
            return unique_file_ids

        except Exception as e:
            logger.error(f"Failed to list files for collection {self.collection_name}: {str(e)}")
            raise