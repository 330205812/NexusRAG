# NexusRAG
NexusRAG is an enterprise-level knowledge base backend system designed for LLM implementation. The system provides ready-to-use modules for document processing and RAG, enabling rapid deployment of large-scale knowledge retrieval systems through integrated full-text search, semantic retrieval, and knowledge graph querying capabilities. The system allows users to:
- Create and manage personal knowledge bases
- Automatically upload and process documents in personal knowledge bases (supports multimodal OCR)
- Employ multiple search methods across knowledge bases (full-text, vector, or hybrid search)
- Perform fine-grained management of document segments within knowledge bases
- Track document processing status in real-time

## Quick start

### 1. Prepare Environment and 下载对应模型文件

Create volume directories under the `docker` directory：
```bash
cd docker
mkdir -p volumes/elasticsearch
mkdir -p volumes/etcd
mkdir -p volumes/milvus
mkdir -p volumes/minio
```
Start Services
```bash
docker-compose up -d
```
This command will download and start the following containers:

- elasticsearch: For full-text search and document indexing
- milvus: For vector similarity search
- minio: For object storage
- etcd: For distributed key-value storage
- kibana: For elasticsearch visualization

All images will be downloaded to Docker's default image storage location (/var/lib/docker/). Total size ~2GB, may take 5-10 minutes depending on your network speed.

If the following words are displayed, it indicates that the download is complete.
```
[+] Running 6/6
⠿ Network docker_default       Created
⠿ Container elasticsearch      Started
⠿ Container milvus-etcd        Started
⠿ Container milvus-minio       Started
⠿ Container milvus-standalone  Started
⠿ Container kibana             Started
```
Check the service operation status
```bash
docker-compose ps
```
When you see output similar to this, all services have been successfully started, 
elasticsearch runs on port 9200, and milvus runs on port 19530:
```
NAME                COMMAND                  SERVICE             STATUS              PORTS
elasticsearch       "/bin/tini -- /usr/l…"   elasticsearch       running             0.0.0.0:9200->9200/tcp, 0.0.0.0:9300->9300/tcp, :::9200->9200/tcp, :::9300->9300/tcp
kibana              "/bin/tini -- /usr/l…"   kibana              running             0.0.0.0:5601->5601/tcp, :::5601->5601/tcp
milvus-etcd         "etcd -advertise-cli…"   etcd                running             2379-2380/tcp
milvus-minio        "/usr/bin/docker-ent…"   minio               running (healthy)   9000/tcp
milvus-standalone   "/tini -- milvus run…"   standalone          running             0.0.0.0:9091->9091/tcp, 0.0.0.0:19530->19530/tcp, :::9091->9091/tcp, :::19530->19530/tcp
```
### 2. Modify the config.ini file.

```
[rag]
knowledgebase_log_dir = /path/to/your/knowledgebase_server_log
knowledgebase_bind_port = 1001

embedding_model_path = /path/to/your/text2vec-large-chinese
embedding_bind_port = 5003

reranker_model_path = /path/to/your/bce-reranker-base_v1
reranker_bind_port = 5001

es_url = http://localhost:9200
index_name = test
milvus_url = localhost:19530
```








## 🔧 FAQ
If Elasticsearch fails to start successfully, enter the following in the docker directory within the container:
```bash
sudo chown -R 1000:0 ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/elasticsearch
```
