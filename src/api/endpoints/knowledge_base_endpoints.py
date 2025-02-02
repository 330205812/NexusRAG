import os
import asyncio
import threading
from loguru import logger
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from ...services.service_manager import ServiceManager
from ...utils.tools import handle_status_file, sanitize_filename
from ...models.models import (
    CreateKnowledgeBaseRequest, CreateKnowledgeBaseResponse,
    AddFileRequest, AddFileResponse,
    SearchRequest, SearchResponse, SearchResult,
    DeleteRequest, DeleteResponse,
    UpdateChunkRequest, UpdateChunkResponse,
    ObtainRequest, ObtainResponse,
    ListFileRequest, ListFileResponse,
    ListChunkRequest, ListChunkResponse,
    CreateChunkRequest, CreateChunkResponse,
    FileStatusRequest, BatchFileStatusResponse,
    FileStatusItem
)

try:
    from pymilvus import connections, utility, exceptions
except ImportError:
    raise ValueError(
        "Could not import pymilvus python package. "
        "Please install it with `pip install pymilvus`."
    )


router = APIRouter()


@router.post("/create_knowledge_base", response_model=CreateKnowledgeBaseResponse)
async def create_knowledge_base(request: CreateKnowledgeBaseRequest):
    """
    Create a new knowledge base with specified parameters.

    This endpoint creates a new knowledge base instance with associated vector store
    and elasticsearch index partitioning.

    Parameters:
    ----------
    request : CreateKnowledgeBaseRequest
        Request object containing:
        - user_id: ID of the user creating the knowledge base
        - name: Name of the knowledge base
        - vector_model_url: Optional URL for custom vector model
        - vector_model_name: Optional name of custom vector model

    Returns:
    -------
    CreateKnowledgeBaseResponse
        Response containing:
        - ID: Unique identifier for the created knowledge base
        - milvus_is_new: Boolean indicating if new Milvus collection was created
        - es_is_new: Boolean indicating if new logical partition was created in ES index

    Notes:
    -----
    - Creates necessary storage directories for the knowledge base
    - If vector_model_url and vector_model_name are provided and connection is successful,
      creates a new Milvus collection using the knowledge base ID
    - Creates a logical partition within the existing Elasticsearch index using the knowledge base ID
      to isolate different users' knowledge bases
    - Supports custom vector models for embeddings

    Raises:
    ------
    HTTPException
        500 status code if knowledge base creation fails for any reason
    """
    knowledgebase_service = ServiceManager.get_knowledge_base_service()
    try:
        kb_id, milvus_is_new, es_is_new = knowledgebase_service.create_knowledge_base(
            user_id=request.user_id,
            name=request.name,
            vector_model_url=request.vector_model_url,
            vector_model_name=request.vector_model_name
        )
        return CreateKnowledgeBaseResponse(ID=kb_id, milvus_is_new=milvus_is_new, es_is_new=es_is_new)
    except Exception as e:
        logger.error(f"Error creating knowledge base: {e}")
        raise HTTPException(status_code=500, detail="Error creating knowledge base")


@router.post("/add_files", response_model=AddFileResponse)
async def add_files(file: UploadFile = File(...),
                    knowledge_base_id: str = Form(...),
                    is_md_splitter: Optional[bool] = Form(default=False),
                    vector_model_url: Optional[str] = Form(default=None),
                    vector_model_name: Optional[str] = Form(default=None),
                    chunk_size: Optional[int] = Form(default=None)
                    ):
    """
    Add files to a knowledge base and process them asynchronously.

    This endpoint handles file upload, saves the file, and initiates background processing
    for knowledge base integration.

    Parameters:
    ----------
    file : UploadFile
        The file to be uploaded and processed
    knowledge_base_id : str
        ID of the target knowledge base
    is_md_splitter : bool, optional
        Whether to use markdown splitter for text processing (default: False)
    vector_model_url : str, optional
        URL of the vector model service if using custom embedding
    vector_model_name : str, optional
        Name of the vector model if using custom embedding
    chunk_size : int, optional
        Size of text chunks for processing (default: None)

    Returns:
    -------
    AddFileResponse
        Response containing:
        - code: HTTP status code
        - statustext: Processing status ('pending')
        - file_id: ID assigned to the processed file

    Raises:
    ------
    HTTPException
        - 404 if knowledge base doesn't exist
        - 500 if:
            - Failed to connect to vector database
            - Failed to save uploaded file
            - Failed to generate file ID
            - Error during background processing
    ValueError
        If knowledge base directories don't exist

    Notes:
    -----
    - Files are processed asynchronously in background threads
    - Progress can be monitored via status files in the file_status directory
    - Supports custom vector models for embedding if URL and name provided
    """
    knowledgebase_service = ServiceManager.get_knowledge_base_service()
    if vector_model_url and vector_model_name:
        milvus_host, milvus_port = knowledgebase_service.milvus_address.split(":")
        collection_name = knowledge_base_id

        try:
            connections.connect(
                alias="default",
                host=milvus_host,
                port=milvus_port
            )

            collections = utility.list_collections()
            if collection_name not in collections:
                raise HTTPException(
                    status_code=404,
                    detail="Knowledge base does not exist. Please create it first"
                )

        except exceptions.ConnectionConfigException as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise HTTPException(status_code=500, detail=f"Unable to connect to vector database: {e}")

        except HTTPException as e:
            raise

        except Exception as e:
            logger.error(f"Error occurred while checking knowledge base: {e}")
            raise HTTPException(status_code=500, detail=f"Error occurred while checking knowledge base: {e}")

        finally:
            connections.disconnect("default")
    # checkout the path
    knowledge_id_dir_uploads = os.path.join(knowledgebase_service.uploads_dir, knowledge_base_id)
    knowledge_id_dir_processed = os.path.join(knowledgebase_service.processed_dir, knowledge_base_id)
    file_status_dir = os.path.join(knowledge_id_dir_uploads, 'file_status')
    if not all(
            os.path.exists(path) for path in [knowledge_id_dir_uploads, knowledge_id_dir_processed, file_status_dir]):
        raise ValueError(
            "Knowledge base directories does not exist, please check if the knowledge base ID is correct or recreate the knowledge base.")

    logger.info(f"Starting to receive file: {file.filename}")
    # logger.debug(f'file.filename:{file.filename}')
    file_name_without_ext = sanitize_filename(os.path.splitext(file.filename)[0])
    # logger.debug(f'file_name_without_ext:{file_name_without_ext}')
    file_specific_dir_before_process = os.path.join(knowledge_id_dir_uploads, file_name_without_ext)
    file_specific_dir_after_process = os.path.join(knowledge_id_dir_processed, file_name_without_ext)

    os.makedirs(file_specific_dir_before_process, exist_ok=True)
    os.makedirs(file_specific_dir_after_process, exist_ok=True)

    file_path = os.path.join(file_specific_dir_before_process, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            while content := await file.read(16 * 1024 * 1024):
                buffer.write(content)
        logger.info(f"File saved successfully: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    files = knowledgebase_service.file_processor.scan_directory(file_specific_dir_before_process)
    if not files:
        raise HTTPException(status_code=500, detail="Failed to generate file ID")

    file_ids = []
    for file_obj in files:
        file_id = file_obj.file_id
        file_ids.append(file_id)

        handle_status_file(file_status_dir,
                           file_id,
                           operation='write',
                           status='pending')
    try:
        thread = threading.Thread(
            target=run_async_function,
            args=(process_and_update_status(
                file_ids,
                files,
                file_specific_dir_after_process,
                knowledge_base_id,
                file_status_dir,
                is_md_splitter,
                vector_model_url,
                vector_model_name,
                chunk_size
            ),)
        )
        thread.start()

        logger.info(f"File upload completed, starting background processing: {file_ids}")

        if len(file_ids) == 1:
            return AddFileResponse(code=200,
                                   statustext='pending',
                                   file_id=file_ids[0])
        else:
            # TODO
            pass

    except Exception as e:
        logger.error(f"Error occurred during background process: {e}")

        for file_id in file_ids:
            handle_status_file(file_status_dir, file_id, operation='write', status='error')
        raise HTTPException(status_code=500, detail=f"Error occurred during background process: {e}")


def run_async_function(coroutine):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coroutine)
    except Exception as e:
        logger.error(f"Async task execution failed: {e}")
        raise
    finally:
        loop.close()


async def process_and_update_status(file_ids: List[str],
                                    files: List,
                                    work_dir: str,
                                    knowledge_base_id: str,
                                    path: str,
                                    is_md_splitter: Optional[bool] = False,
                                    vector_model_url: Optional[str] = None,
                                    vector_model_name: Optional[str] = None,
                                    chunk_size: Optional[int] = None):

    knowledgebase_service = ServiceManager.get_knowledge_base_service()
    try:
        successful_files, failed_files = knowledgebase_service.add_files(
            files=files,
            work_dir=work_dir,
            knowledge_base_id=knowledge_base_id,
            is_md_splitter=is_md_splitter,
            vector_model_url=vector_model_url,
            vector_model_name=vector_model_name,
            chunk_size=chunk_size
        )

        for file_id in successful_files:
            handle_status_file(path,
                               file_id,
                               operation='write',
                               status='success')

        for file_id in failed_files:
            handle_status_file(path,
                               file_id,
                               operation='write',
                               status='error')

        logger.info(f"File processing completed: {successful_files} succeeded, {failed_files} failed")

    except Exception as e:
        logger.error(f"Error occurred while processing file: {e}")
        for file_id in file_ids:
            try:
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        handle_status_file(path,
                                           file_id,
                                           operation='write',
                                           status='error')
                        break
                    except Exception as retry_error:
                        if attempt == max_retries - 1:
                            logger.error(f"Failed to update status file after {max_retries} retries: {retry_error}")
                            raise
                        else:
                            import time
                            time.sleep(1)
            except Exception as lock_error:
                logger.error(f"Failed to update status file: {lock_error}")
        raise


@router.post("/batch_file_status", response_model=BatchFileStatusResponse)
async def batch_file_parse_result(requests: List[FileStatusRequest]):
    """
    Batch retrieve file processing status for multiple files.

    This endpoint checks the processing status of multiple files across knowledge bases
    by reading their status files.

    Parameters:
    ----------
    requests : List[FileStatusRequest]
        List of requests, each containing:
        - knowledge_base_id: ID of the knowledge base
        - file_id: ID of the file to check status for

    Returns:
    -------
    BatchFileStatusResponse
        Response containing a list of FileStatusItem objects, each with:
        - knowledge_base_id: ID of the knowledge base
        - file_id: ID of the file
        - status: Current processing status of the file. Possible values:
            - "pending": File is waiting to be processed
            - "success": File processing completed successfully
            - "error": Error occurred during processing
            - "": Status unknown or can not reading status file

    Notes:
    -----
    - Checks status files in the 'file_status' directory of each knowledge base
    - Returns empty status string if:
        - Status file doesn't exist
    - Continues processing remaining files even if some fail
    """

    knowledgebase_service = ServiceManager.get_knowledge_base_service()
    results = []
    for request in requests:
        try:
            knowledge_id_dir_origin, _ = knowledgebase_service._get_knowledge_dirs(
                knowledge_id=request.knowledge_base_id,
                mode='query'
            )

            file_status_dir = os.path.join(knowledge_id_dir_origin, 'file_status')
            status_file = os.path.join(file_status_dir, f"{request.file_id}.status")

            status = ""
            if os.path.exists(status_file):
                try:
                    status = handle_status_file(
                        file_status_dir,
                        request.file_id,
                        operation='read'
                    )
                except Exception as e:
                    logger.error(f"Error reading status for file {request.file_id}: {e}")
                    status = ""

            results.append(FileStatusItem(
                knowledge_base_id=request.knowledge_base_id,
                file_id=request.file_id,
                status=status
            ))

        except Exception as e:
            logger.error(f"Error processing request for file {request.file_id}: {e}")
            results.append(FileStatusItem(
                knowledge_base_id=request.knowledge_base_id,
                file_id=request.file_id,
                status=""
            ))

    return BatchFileStatusResponse(results=results)


@router.post("/delete", response_model=DeleteResponse)
async def delete(request: DeleteRequest):
    """
    Delete content from knowledge base at different granularity levels.

    This endpoint provides three levels of deletion functionality based on the parameters provided:
    1. Delete entire knowledge base
    2. Delete all chunks of a specific file
    3. Delete a specific chunk from a file

    Parameters:
    ----------
    request : DeleteRequest
        Request object containing:
        - knowledge_base_id: ID of the knowledge base (required)
        - file_id: ID of the file (optional)
        - chunk_id: ID of the specific chunk (optional)

    Returns:
    -------
    DeleteResponse
        Response containing deletion operation result

    Behavior:
    --------
    - If only knowledge_base_id provided:
        Deletes the entire knowledge base including all files and chunks
    - If knowledge_base_id and file_id provided:
        Deletes all chunks associated with the specified file
    - If knowledge_base_id, file_id, and chunk_id provided:
        Deletes only the specified chunk

    Raises:
    ------
    HTTPException
        - 400 status code for validation errors
        - 500 status code for server-side errors
    """

    knowledgebase_service = ServiceManager.get_knowledge_base_service()
    try:
        chunk_id = request.chunk_id
        if chunk_id == "":
            chunk_id = None

        res = knowledgebase_service.delete(
            knowledge_base_id=request.knowledge_base_id,
            file_id=request.file_id,
            chunk_id=chunk_id
        )

        return res

    except ValueError as e:
        logger.error(f"Failed to delete knowledge base: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Retrieve relevant documents from specified knowledge bases based on user query.

    This API allows searching through one or multiple knowledge bases using different search strategies:
    semantic similarity, keyword matching, or a combination of both, with optional result reranking.

    Parameters:
    ----------
    request : SearchRequest
        Request object containing:
        - knowledge_base_id: List of knowledge base IDs to search in
        - query: Search query text
        - file_ids: List of file IDs to restrict search to
        - vector_model_name: Name of vector embedding model (optional)
        - vector_model_url: URL of vector embedding service (optional)
        - search_mode: Search mode (semantic, fulltext, or hybrid)
        - using_rerank: Whether to apply reranking on results
        - reranker_model_name: Name of reranker model (optional)
        - reranker_model_url: URL of reranker service (optional)
        - top_k: Number of results to return (default: 10)
        - threshold: Minimum similarity score threshold (optional)
        - weights: Weights for hybrid search result combination [es_weight, vector_weight]

    Returns:
    -------
    SearchResponse
        Response containing:
        - code: Status code (200 for success, 500 for error)
        - data: List of SearchResult objects containing matched documents and metadata
        - keywords: Keywords extracted from query for fulltext search
        - error_msg: Error message if any

    Notes:
    -----
    - Supports three search modes:
        - semantic: Pure vector similarity search using Milvus
        - fulltext: Keyword-based search using Elasticsearch
        - hybrid: Combines results from both semantic and fulltext search
    - Returns empty results with error message for empty queries
    - Handles connection errors to embedding and reranking services gracefully
    """
    knowledgebase_service = ServiceManager.get_knowledge_base_service()
    search_results = []
    keywords = []

    try:
        normalized_search_mode = request.search_mode.normalize()
        results, keywords, error_message = knowledgebase_service.search(
            knowledge_base_id=request.knowledge_base_id,
            query=request.query,
            file_ids=request.file_ids,
            vector_model_name=request.vector_model_name,
            vector_model_url=request.vector_model_url,
            search_mode=normalized_search_mode,
            using_rerank=request.using_rerank,
            reranker_model_name=request.reranker_model_name,
            reranker_model_url=request.reranker_model_url,
            top_k=request.top_k,
            threshold=request.threshold,
            weights=request.weights
        )

        logger.debug(results)

        for doc in results:
            search_result = SearchResult(
                text=doc.page_content,
                knowledge_base_id=doc.metadata.get("knowledge_base_id", ""),
                file_id=doc.metadata.get("file_id", ""),
                chunk_id=doc.metadata.get("chunk_id", ""),
                score=doc.metadata.get("relevance_score", 0.0)
            )
            search_results.append(search_result)

        if error_message:
            return SearchResponse(code=500, data=search_results, keywords=keywords, error_msg=error_message)
        else:
            return SearchResponse(data=search_results, keywords=keywords)

    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        return SearchResponse(code=500, data=search_results, keywords=keywords, error_msg=str(e))


@router.post("/update_chunk", response_model=UpdateChunkResponse)
async def update_chunk(request: UpdateChunkRequest):
    """
    Update the content of a specific chunk in a knowledge base.

    This endpoint updates both the text content in Elasticsearch and the vector embedding
    in Milvus for a specified chunk within a file.

    Parameters:
    ----------
    request : UpdateChunkRequest
        Request object containing:
        - knowledge_base_id: ID of the knowledge base
        - file_id: ID of the file
        - chunk_id: ID of the chunk
        - text: New text content for the chunk
        - vector_model_url: URL of the vector embedding service (optional)
        - vector_model_name: Name of the vector model to use (optional)

    Returns:
    -------
    UpdateChunkResponse
        Response containing:
        - file_id: ID of the updated file
        - chunk_id: ID of the updated chunk
        - es_msg: Message indicating Elasticsearch update status
        - milvus_msg: Message indicating Milvus update status

    Notes:
    -----
    - Requires all of file_id, chunk_id, and text to be non-empty
    - Updates both text content and vector embedding if vector model details provided
    - Updates only text content if vector model details not provided
    - Both storage backends (ES and Milvus) must be updated successfully

    Raises:
    ------
    HTTPException
        - 400 status code if required fields are missing or empty
        - 500 status code if update operation fails
    """
    knowledgebase_service = ServiceManager.get_knowledge_base_service()
    try:

        if not request.file_id or not request.chunk_id or not request.text:
            raise HTTPException(
                status_code=400,
                detail="Failed to create chunk due to invalid input"
            )

        es_msg, milvus_msg = knowledgebase_service.update_chunk(
            knowledge_base_id=request.knowledge_base_id,
            file_id=request.file_id,
            chunk_id=request.chunk_id,
            text=request.text,
            vector_model_url=request.vector_model_url,
            vector_model_name=request.vector_model_name
        )

        return UpdateChunkResponse(
            file_id=request.file_id,
            chunk_id=request.chunk_id,
            es_msg=es_msg,
            milvus_msg=milvus_msg,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/obtain_knowledge_base_info", response_model=ObtainResponse)
async def obtain_knowledge_base_info(request: ObtainRequest):
    """
    Retrieve information about knowledge bases accessible to a user.

    This endpoint returns a list of knowledge bases and their details based on user permissions.
    Admin users can see all knowledge bases, while regular users only see knowledge bases they have access to.

    Parameters:
    ----------
    request : ObtainRequest
        Request object containing:
        - user_id: ID of the user requesting knowledge base information
        - is_admin: Boolean flag indicating whether the user has admin privileges

    Returns:
    -------
    ObtainResponse
    Response containing:
        - elasticsearch_knowledge_bases_ids: List of knowledge base IDs present in Elasticsearch
        - milvus_knowledge_bases_ids: List of knowledge base IDs present in Milvus vector storage

    Notes:
    -----
    - Admin users see all knowledge bases in the system
    - Regular users only see knowledge bases they are authorized to access
    - Lists may differ between Elasticsearch and Milvus
    - Returns empty list if user has no accessible knowledge bases

    Raises:
    ------
    HTTPException
        - 500 status code for server-side errors during retrieval
    """
    knowledgebase_service = ServiceManager.get_knowledge_base_service()
    try:
        result = knowledgebase_service.obtain_knowledge_base_info(request.user_id,
                                                                  request.is_admin)

        return ObtainResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/list_files_with_status", response_model=ListFileResponse)
async def list_files_with_status(request: ListFileRequest):
    """
    List status of all files in a knowledge base by reading their status files.

    This endpoint retrieves the processing status of files in a specified knowledge base by reading
    the corresponding .status files from the file system.

    Parameters:
    ----------
    request : ListFileRequest
        Request object containing:
        - knowledge_base_id: ID of the knowledge base to list file statuses from
        - is_admin: Boolean flag indicating whether the request is from an admin user

    Returns:
    -------
    ListFileResponse
        Response containing:
        - status_dict: Dictionary mapping file IDs to their status strings
            {
                "file_id": "status_content",  # Status content from .status file
                ...
            }

    Notes:
    -----
    - Verifies knowledge base exists in Elasticsearch before checking files
    - Reads status from individual .status files in the file_status directory
    - Skips files that can't be read due to permissions or corruption
    - Status content is read as plain text from status files

    Raises:
    ------
    HTTPException
        - 500 status code for server-side errors
    FileNotFoundError
        - When knowledge base ID doesn't exist in Elasticsearch
        - When knowledge base directory doesn't exist
        - When file_status directory doesn't exist in knowledge base
    """
    knowledgebase_service = ServiceManager.get_knowledge_base_service()
    try:
        result = knowledgebase_service.list_files_with_status(request.knowledge_base_id,
                                                              request.is_admin)
        return ListFileResponse(status_dict=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/list_chunks_info", response_model=ListChunkResponse)
async def list_chunks_info(request: ListChunkRequest):
    """
    List information about chunks in a knowledge base, with support for single chunk lookup and pagination.

    This endpoint queries Elasticsearch to retrieve chunk information, supporting three modes:
    1. Single chunk lookup by ID
    2. Paginated list of chunks
    3. Complete list of all chunks (using scroll API)

    Parameters:
    ----------
    request : ListChunkRequest
        Request object containing:
        - knowledge_base_id: ID of the knowledge base
        - file_id: ID of the file to list chunks from
        - chunk_id: Optional; specific chunk ID to retrieve
        - size: Optional; number of items per page (pagination)
        - offset: Optional; page number, starting from 1 (pagination)
        - use_embedding_id_as_id: Optional; if True, use embedding_id as the id field in response

    Returns:
    -------
    ListChunkResponse
        Response containing:
        - total: Total number of chunks matching the query
        - chunks: List of chunk information objects, each containing:
            - id: Chunk ID (Elasticsearch document ID)
            - dataset_id: Knowledge base ID
            - collection_id: File ID
            - embedding_id: chunk ID
            - text: Chunk content
            - metadata: Additional chunk metadata

    Query Behavior:
    -------------
    1. Single Chunk Query (when chunk_id provided):
       - Returns exact match or empty result
       - Document ID format: "{knowledge_base_id}_{file_id}_{chunk_id}"

    2. Paginated Query (when size and offset provided):
       - Returns specified page of results
       - Validates positive values for pagination parameters

    3. Complete List Query (when no pagination parameters):
       - Retrieves all chunks in batches of 1000
       - Automatically cleans up scroll context

    Notes:
    -----
    - Handles missing documents gracefully
    - Automatically cleans up scroll contexts to prevent resource leaks
    - Returns empty result if no matches found

    Raises:
    ------
    HTTPException
        - 500 status code for server-side errors
    ValueError
        - When pagination parameters are invalid (non-positive)
    ImportError
        - When Elasticsearch package is not installed
    """
    knowledgebase_service = ServiceManager.get_knowledge_base_service()
    try:
        result = knowledgebase_service.list_chunks_info(request.knowledge_base_id,
                                                        request.file_id,
                                                        request.chunk_id,
                                                        request.size,
                                                        request.offset,
                                                        request.use_embedding_id_as_id)
        return ListChunkResponse(total=result["total"], chunks=result["chunks"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create_chunk", response_model=CreateChunkResponse)
async def create_chunk(request: CreateChunkRequest):
    """
    Create a new text chunk in both Elasticsearch and Milvus (optional) storage systems.

    This endpoint creates a new chunk by storing the text in Elasticsearch and optionally creating
    vector embeddings in Milvus. The chunk is associated with a knowledge base and can be linked
    to an existing or new file.

    Parameters:
    ----------
    request : CreateChunkRequest
        Request object containing:
        - knowledge_base_id: ID of the knowledge base to store the chunk in
        - text: Content of the chunk to be stored
        - file_id: Optional; ID of the file to associate the chunk with (auto-generated if not provided)
        - vector_model_url: Optional; URL of the embedding model service
        - vector_model_name: Optional; Name of the vector embedding model to use

    Returns:
    -------
    CreateChunkResponse
        Response containing:
        - file_id: ID of the file (provided or auto-generated)
        - chunk_id: Generated ID for the chunk (first 16 chars of SHA-256 hash of text)
        - es_msg: Status message from Elasticsearch operation
        - milvus_msg: Status message from Milvus operation (if vector storage attempted)

    Notes:
    -----
    - Chunk ID is deterministic based on text content
    - File ID is either provided or generated from text hash
    - Operations in Elasticsearch and Milvus are independent
    - Status files track the state of document creation
    - Both storages support failure recovery
    - Milvus connection is properly closed after operation

    Raises:
    ------
    HTTPException
        - 400 status code for invalid input parameters
        - 500 status code for internal server errors during processing
    """
    knowledgebase_service = ServiceManager.get_knowledge_base_service()
    try:

        result = knowledgebase_service.create_chunk(
            knowledge_base_id=request.knowledge_base_id,
            text=request.text,
            file_id=request.file_id,
            vector_model_url=request.vector_model_url,
            vector_model_name=request.vector_model_name
        )

        if result["file_id"] is None and result["chunk_id"] is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to create chunk due to invalid input"
            )

        return CreateChunkResponse(**result)

    except Exception as e:
        logger.error(f"Error in create_chunk_endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )