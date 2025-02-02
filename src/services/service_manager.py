from knowledge_base_service import KnowledgeBaseService

class ServiceManager:
    _instance = None
    knowledge_base_service = None

    @classmethod
    def initialize(cls, config):
        if not cls._instance:
            cls._instance = cls()
            cls.knowledge_base_service = KnowledgeBaseService(config)
        return cls._instance

    @classmethod
    def get_knowledge_base_service(cls):
        if not cls.knowledge_base_service:
            raise RuntimeError("KnowledgeBaseService not initialized")
        return cls.knowledge_base_service