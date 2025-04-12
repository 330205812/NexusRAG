from typing import Dict
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, PositiveInt
from dataclasses import dataclass

__all__ = ['KnowledgeBaseType', 'CreateKnowledgeBaseRequest', 'CreateKnowledgeBaseResponse',
           'AddFileRequest', 'AddFileResponse', 'FileStatusRequest', 'FileStatusItem',
           'BatchFileStatusResponse', 'DeleteRequest', 'DeleteResponse', 'SearchMode',
           'SearchResult', 'SearchRequest', 'SearchResponse', 'UpdateChunkRequest',
           'UpdateChunkResponse', 'ObtainRequest', 'ObtainResponse', 'ListFileResponse',
           'ListFileRequest', 'ListChunkRequest', 'ListChunkResponse', 'CreateChunkRequest',
           'CreateChunkResponse']


class KnowledgeBaseType(str, Enum):
    public = "public"
    private = "private"


class CreateKnowledgeBaseRequest(BaseModel):
    user_id: str
    name: str
    vector_model_url: Optional[str] = None
    vector_model_name: Optional[str] = None


class CreateKnowledgeBaseResponse(BaseModel):
    ID: str
    milvus_is_new: bool | None
    es_is_new: bool


@dataclass
class AddFileRequest:
    file: str
    knowledge_base_id: str
    vector_model_url: Optional[str] = None
    vector_model_name: Optional[str] = None
    chunk_size: Optional[int] = None


class AddFileResponse(BaseModel):
    code: int
    statustext: str
    file_id: str


class FileStatusRequest(BaseModel):
    knowledge_base_id: str
    file_id: str


class FileStatusItem(BaseModel):
    knowledge_base_id: str
    file_id: str
    status: str
    error_msg: str


class BatchFileStatusResponse(BaseModel):
    results: List[FileStatusItem]


class DeleteRequest(BaseModel):
    knowledge_base_id: Optional[str] = None
    file_id: Optional[str] = None
    chunk_id: Optional[str] = None


class DeleteResponse(BaseModel):
    es_message: str
    milvus_message: str


class SearchMode(str, Enum):
    SEMANTIC = "semantic"
    FULLTEXT = "fulltext"
    HYBRID = "hybrid"
    EMBEDDING = "embedding"
    FULLTEXT_RECALL = "fullTextRecall"
    MIXED_RECALL = "mixedRecall"

    def normalize(self) -> 'SearchMode':
        mode_mapping = {
            self.EMBEDDING: self.SEMANTIC,
            self.FULLTEXT_RECALL: self.FULLTEXT,
            self.MIXED_RECALL: self.HYBRID,
            self.SEMANTIC: self.SEMANTIC,
            self.FULLTEXT: self.FULLTEXT,
            self.HYBRID: self.HYBRID
        }
        return mode_mapping[self]


class SearchResult(BaseModel):
    text: str
    knowledge_base_id: str
    file_id: str
    chunk_id: str
    score: float
    title: str


class SearchResponse(BaseModel):
    code: int = 200
    data: List[SearchResult] = []
    keywords: List[str] = []
    error_msg: Optional[str] = ""


class SearchRequest(BaseModel):
    knowledge_base_id: Optional[List[str]] = None
    query: Optional[str] = None
    file_ids: Optional[List[str]] = None
    vector_model_name: Optional[str] = None
    vector_model_url: Optional[str] = None
    search_mode: SearchMode = SearchMode.HYBRID
    using_rerank: bool = False
    reranker_model_name: Optional[str] = None
    reranker_model_url: Optional[str] = None
    top_k: Optional[PositiveInt] = 10
    threshold: Optional[float] = None
    weights: Optional[List[float]] = None


class UpdateChunkRequest(BaseModel):
    knowledge_base_id: str
    file_id: str
    chunk_id: str
    text: str
    vector_model_url: Optional[str] = None
    vector_model_name: Optional[str] = None


class UpdateChunkResponse(BaseModel):
    file_id: Optional[str] = Field(None, description="File identifier")
    chunk_id: Optional[str] = Field(None, description="Chunk identifier")
    es_msg: str = Field(..., description="Elasticsearch operation message")
    milvus_msg: str = Field(..., description="Milvus operation message")


class ObtainRequest(BaseModel):
    user_id: Optional[str] = None
    is_admin: Optional[bool] = False


class ObtainResponse(BaseModel):
    elasticsearch_knowledge_bases_ids: List[str] = Field(default_factory=list)
    milvus_knowledge_bases_ids: List[str] = Field(default_factory=list)


class ListFileResponse(BaseModel):
    status_dict: Dict[str, str]


class ListFileRequest(BaseModel):
    knowledge_base_id: Optional[str] = None
    is_admin: Optional[bool] = False


class ListChunkRequest(BaseModel):
    knowledge_base_id: str
    file_id: str
    chunk_id: Optional[str] = None
    size: Optional[int] = None
    offset: Optional[int] = None
    use_embedding_id_as_id: Optional[bool] = True
    key: Optional[str] = None


class ListChunkResponse(BaseModel):
    total: int
    chunks: List[Dict[str, str]]


class CreateChunkRequest(BaseModel):
    knowledge_base_id: str = Field(..., description="Knowledge base identifier")
    text: str = Field(..., description="Text content to be chunked")
    file_id: Optional[str] = Field(None, description="Optional file identifier")
    title: str = Field(..., description="Title of content")
    vector_model_url: Optional[str] = Field(None, description="URL for the vector model")
    vector_model_name: Optional[str] = Field(None, description="Name of the vector model")


class CreateChunkResponse(BaseModel):
    file_id: Optional[str] = Field(None, description="File identifier")
    chunk_id: Optional[str] = Field(None, description="Chunk identifier")
    es_msg: str = Field(..., description="Elasticsearch operation message")
    milvus_msg: str = Field(..., description="Milvus operation message")