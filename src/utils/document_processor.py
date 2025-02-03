import fitz
import re
import os
import time
import pandas as pd
import hashlib
import textract
import shutil
import multiprocessing
from multiprocessing import Pool
from loguru import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from .markdown_text_splitter import MarkdownParser
from .tools import get_partial_sha256_hash


class DocumentName:

    def __init__(self, directory: str, name: str, category: str):
        self.directory = directory
        self.prefix = name.replace('/', '_')
        self.basename = os.path.basename(name)
        self.origin_path = os.path.join(directory, name)
        self.copy_path = ''
        self.file_id = get_partial_sha256_hash(os.path.join(directory, name))
        self._category = category
        self.status = True
        self.message = ''

    def __str__(self):
        return '{},{},{},{}\n'.format(self.basename, self.copy_path, self.status,
                                      self.message)


class DocumentProcessor:

    def __init__(self):
        self.image_suffix = ['.jpg', '.jpeg', '.png', '.bmp']
        self.md_suffix = '.md'
        self.text_suffix = ['.txt', '.text']
        self.excel_suffix = ['.xlsx', '.xls', '.csv']
        self.pdf_suffix = '.pdf'
        self.ppt_suffix = '.pptx'
        self.html_suffix = ['.html', '.htm', '.shtml', '.xhtml']
        self.word_suffix = ['.docx', '.doc']
        self.json_suffix = '.json'

    def read_file_type(self, filepath: str):
        filepath = filepath.lower()
        if filepath.endswith(self.pdf_suffix):
            return 'pdf'

        if filepath.endswith(self.md_suffix):
            return 'md'

        if filepath.endswith(self.ppt_suffix):
            return 'ppt'

        if filepath.endswith(self.json_suffix):
            return 'json'

        for suffix in self.image_suffix:
            if filepath.endswith(suffix):
                return 'image'

        for suffix in self.text_suffix:
            if filepath.endswith(suffix):
                return 'text'

        for suffix in self.word_suffix:
            if filepath.endswith(suffix):
                return 'word'

        for suffix in self.excel_suffix:
            if filepath.endswith(suffix):
                return 'excel'

        for suffix in self.html_suffix:
            if filepath.endswith(suffix):
                return 'html'

        return None

    def scan_directory(self, repo_dir: str):
        documents = []
        for directory, _, names in os.walk(repo_dir):
            for name in names:
                category = self.read_file_type(name)
                if category is not None:
                    documents.append(
                        DocumentName(directory=directory,
                                     name=name,
                                     category=category))
        return documents

    def read(self, filepath: str):

        file_type = self.read_file_type(filepath)

        text = ''
        if not os.path.exists(filepath):
            return text

        try:
            if file_type == 'md' or file_type == 'text':
                text = []
                with open(filepath) as f:
                    txt = f.read()
                cleaned_txt = re.sub(r'\n\s*\n', '\n\n', txt)
                text.append(cleaned_txt)

            elif file_type == 'pdf':
                text += self.read_pdf(filepath)
                text = re.sub(r'\n\s*\n', '\n\n', text)

            elif file_type == 'excel':
                text += self.read_excel(filepath)

            elif file_type == 'word' or file_type == 'ppt':
                # https://stackoverflow.com/questions/36001482/read-doc-file-with-python
                # https://textract.readthedocs.io/en/latest/installation.html
                text = textract.process(filepath).decode('utf8')
                text = re.sub(r'\n\s*\n', '\n\n', text)

            elif file_type == 'html':
                with open(filepath) as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                    text += soup.text

            elif file_type == 'json':
                with open(filepath, 'r', encoding='utf-8') as file:
                    text = file.read()

        except Exception as e:
            logger.error((filepath, str(e)))
            return '', e

        return text, None

    def read_excel(self, filepath: str):
        table = None
        if filepath.endswith('.csv'):
            table = pd.read_csv(filepath)
        else:
            table = pd.read_excel(filepath)
        if table is None:
            return ''
        json_text = table.dropna(axis=1).to_json(force_ascii=False)
        return json_text

    def read_pdf(self, filepath: str):

        text = ''
        with fitz.open(filepath) as pages:
            for page in pages:
                text += page.get_text()
                tables = page.find_tables()
                for table in tables:
                    tablename = '_'.join(
                        filter(lambda x: x is not None and 'Col' not in x,
                               table.header.names))
                    pan = table.to_pandas()
                    json_text = pan.dropna(axis=1).to_json(force_ascii=False)
                    text += tablename
                    text += '\n'
                    text += json_text
                    text += '\n'
        return text


def read_and_save(file: DocumentName, file_opr: DocumentProcessor):
    try:
        if os.path.exists(file.copy_path):
            # already exists, return
            logger.info('{} already processed, output file: {}, skip load'
                        .format(file.origin_path, file.copy_path))
            return

        logger.info('reading {}, would save to {}'.format(file.origin_path,
                                                          file.copy_path))
        content, error = file_opr.read(file.origin_path)
        if error is not None:
            logger.error('{} load error: {}'.format(file.origin_path, str(error)))
            return

        if content is None or len(content) < 1:
            logger.warning('{} empty, skip save'.format(file.origin_path))
            return

        cleaned_content = re.sub(r'\n\s*\n', '\n\n', content)
        with open(file.copy_path, 'w') as f:
            f.write(os.path.splitext(file.basename)[0] + '\n')
            f.write(cleaned_content)

    except Exception as e:
        logger.error(f"Error in read_and_save: {e}")


def read_pdf_from_server(file: DocumentName,
                         server=None,
                         ocr_process_dir=None):
    try:
        if os.path.exists(file.copy_path):
            logger.info('{} already processed, output file: {}, skip load'
                        .format(file.origin_path, file.copy_path))
            return True

        logger.info('reading {} with ocr server'.format(file.origin_path))
        output_file_path = server.ocr_pdf_client(path=file.origin_path, output_dir=ocr_process_dir)

        if not output_file_path:
            error_msg = '{} reading error'.format(file.origin_path)
            logger.error(error_msg)
            raise Exception(error_msg)

        logger.info(f'File processed successfully, output_file_path:{output_file_path}')
        return True

    except Exception as e:
        error_msg = f"Error in process {file.origin_path} with read_pdf_from_server: {str(e)}"
        logger.error(error_msg)
        raise


class FeatureDataBase:

    def __init__(self) -> None:
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1068, chunk_overlap=32)
        self.md_splitter = MarkdownParser(max_chunk_size=1500)

    def get_split_texts(self, text):
        # if len(text) <= 1:
        #     return []
        chunks = self.splitter.create_documents(text)
        split_texts = []
        chunks_hashes = []
        for chunk in chunks:
            split_texts.append(chunk.page_content)
            hash_value = hashlib.sha256(chunk.page_content.encode('utf-8')).hexdigest()[:16]
            chunks_hashes.append(hash_value)
        return split_texts, chunks_hashes

    def build_database(self, files: list,
                       file_opr: DocumentProcessor,
                       knowledge_base_id=str,
                       is_md_splitter: bool = False,
                       elastic_search=None,
                       milvus=None,
                       vector_model_name: str = None,
                       chunk_size: int = None,
                       chunk_size_overlap: int = None
                       ):

        if chunk_size and chunk_size_overlap:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_size_overlap)
            self.md_splitter = MarkdownParser(max_chunk_size=chunk_size)
        if is_md_splitter:
            self.splitter = self.md_splitter
        else:
            self.splitter = self.text_splitter

        texts = []
        file_id_list = []
        chunks_ids_list = []
        unique_file_id_list = []

        for i, file in enumerate(files):
            if not file.status:
                continue
            text, error = file_opr.read(file.copy_path)

            if error is not None:
                file.status = False
                file.message = str(error)
                continue
            file.message = str(text[0])
            unique_file_id_list.append(f"{file.file_id}")

            split_texts, chunks_ids = self.get_split_texts(text)
            file_id_list_temp = [file.file_id for _ in range(len(split_texts))]

            texts += split_texts
            file_id_list += file_id_list_temp
            chunks_ids_list += chunks_ids

            logger.debug('Pipeline {}/{}.. register 《{}》 and split {} documents'
                         .format(i + 1, len(files), file.basename, len(split_texts)))

        if milvus is not None:
            logger.debug('Milvus pipeline: registering {} files (total {} chunks) to knowledge base {}'.format(
                len(unique_file_id_list),
                len(texts),
                knowledge_base_id
            ))
            milvus_time_before_register = time.time()
            success_ids, all_ids = milvus.add_texts(
                knowledge_base_id=knowledge_base_id,
                vector_model_name=vector_model_name,
                texts=texts,
                file_ids=file_id_list,
                chunk_ids=chunks_ids_list,
            )
            milvus_time_after_register = time.time()
            logger.debug(
                'Milvus pipeline take time: {} '.format(milvus_time_after_register - milvus_time_before_register))

            success_file_ids = list(dict.fromkeys(id.split("_", 2)[1] for id in success_ids))
            all_file_ids = list(dict.fromkeys(id.split("_", 2)[1] for id in all_ids))
            failed_file_ids = list(set(all_file_ids) - set(success_file_ids))
            logger.debug(
                'Milvus pipeline successfully inserted files count: {}, file ids: {}'.format(len(success_file_ids),
                                                                                             success_file_ids))
            logger.debug(
                'Milvus pipeline failed inserted files count: {}, file ids: {}'.format(len(failed_file_ids),
                                                                                       failed_file_ids))

        if elastic_search is not None:
            logger.debug('ES pipeline: registering {} files (total {} chunks) to knowledge base {}'.format(
                len(unique_file_id_list),
                len(texts),
                knowledge_base_id
            ))
            es_time_before_register = time.time()
            ids, failed_files = elastic_search.add_texts(knowledge_base_id=knowledge_base_id,
                                                         texts=texts,
                                                         file_id_list=file_id_list,
                                                         chunks_ids_list=chunks_ids_list)
            es_time_after_register = time.time()
            logger.debug('ES pipeline take time: {} '.format(es_time_after_register - es_time_before_register))

            successful_files = [file_id for file_id in unique_file_id_list if file_id not in failed_files]
            logger.debug('ES pipeline successfully inserted files count: {}, file ids: {}'.format(len(successful_files),
                                                                                                  successful_files))
            logger.debug(
                'ES pipeline failed inserted files count: {}, file ids: {}'.format(len(failed_files), failed_files))

        return successful_files, failed_files

    def preprocess(self, files: list, work_dir: str, file_opr: DocumentProcessor, is_md_splitter: bool = False,
                   server=None):

        """
        Input work_dir is file_specific_dir_after_process (a folder named after the file).
        - OCR results for PDF documents are stored in work_dir/ocr_process/
        - Other document processing results are stored in work_dir/preprocess/
        Both directories are created if they don't exist.
        """
        for idx, file in enumerate(files):
            if not os.path.exists(file.origin_path):
                file.status = False
                file.message = 'skip not exist'
                continue

            if file._category == 'image':
                file.status = False
                file.message = 'skip image'

            elif file._category == 'pdf':
                if server:
                    processed_knowledge_id_dir, file_name_without_ext = os.path.split(work_dir)
                    file.copy_path = os.path.join(work_dir,
                                                  file_name_without_ext + '.txt')
                    if is_md_splitter:
                        file.copy_path = os.path.join(work_dir,
                                                      file_name_without_ext + '.md')
                    read_pdf_from_server(file, server, processed_knowledge_id_dir)
                else:
                    file.copy_path = os.path.join(work_dir,
                                                  '{}.text'.format(file.file_id))
                    read_and_save(file, file_opr)

            elif file._category in ['word', 'ppt', 'html', 'excel', 'json']:
                # read pdf/word/excel file and save to text format
                file.copy_path = os.path.join(work_dir,
                                              '{}.text'.format(file.file_id))
                read_and_save(file, file_opr)

            elif file._category in ['md', 'text']:
                # rename text files to new dir
                file.copy_path = os.path.join(work_dir,
                                              '{}.text'.format(file.file_id))

                if os.path.exists(file.copy_path):
                    # already exists, return
                    logger.info('{} already processed, output file: {}, skip'
                                .format(file.origin_path, file.copy_path))
                    file.status = True
                    file.message = 'preprocessed'
                else:
                    try:
                        shutil.copy2(file.origin_path, file.copy_path)
                        file.status = True
                        file.message = 'preprocessed'
                    except Exception as e:
                        file.status = False
                        file.message = str(e)
            else:
                file.status = False
                file.message = 'skip unknown format'

        for file in files:
            if file._category in ['pdf', 'word', 'excel', 'json']:
                if os.path.exists(file.copy_path):
                    file.status = True
                    file.message = 'preprocessed'
                else:
                    file.status = False
                    file.message = 'read error'

    def async_preprocess(self, files: list, work_dir: str, file_opr: DocumentProcessor,
                         is_md_splitter: bool = False, server=None, processes=16, timeout=3600):

        """
        Asynchronously preprocess files using a process pool.

        Args:
            files: List of files to be processed
            work_dir: Working directory for storing processed files
            file_opr: Document processor instance for handling file operations
            is_md_splitter: Whether to use markdown splitter
            server: PDF server instance for processing PDF files
            processes: Number of processes in the pool (default: 16)
            timeout: Timeout in seconds for each task (default: 3600)

        Notes:
            - For PDF files, OCR results are stored in work_dir/ocr_process/
            - Other document processing results are stored in work_dir/preprocess/
            - Both directories are created if they don't exist
            - Supports file types: PDF, Word, PPT, HTML, Excel, JSON, Markdown, Text
            - Skips image files and unknown formats
        """
        async_results = []
        pool = Pool(processes=processes)

        for idx, file in enumerate(files):
            if not os.path.exists(file.origin_path):
                file.status = False
                file.message = 'skip not exist'
                continue

            if file._category == 'image':
                file.status = False
                file.message = 'skip image'

            elif file._category == 'pdf':
                if server:
                    processed_knowledge_id_dir, file_name_without_ext = os.path.split(work_dir)
                    file.copy_path = os.path.join(work_dir,
                                                  file_name_without_ext + '.txt')
                    if is_md_splitter:
                        file.copy_path = os.path.join(work_dir,
                                                      file_name_without_ext + '.md')
                    result = pool.apply_async(read_pdf_from_server,
                                              args=(file, server, processed_knowledge_id_dir))
                    async_results.append((file, result))
                else:
                    file.copy_path = os.path.join(work_dir,
                                                  '{}.text'.format(file.file_id))
                    result = pool.apply_async(read_and_save, args=(file, file_opr))
                    async_results.append((file, result))

            elif file._category in ['word', 'ppt', 'html', 'excel', 'json']:
                file.copy_path = os.path.join(work_dir,
                                              '{}.text'.format(file.file_id))
                result = pool.apply_async(read_and_save, args=(file, file_opr))
                async_results.append((file, result))

            elif file._category in ['md', 'text']:
                file.copy_path = os.path.join(work_dir,
                                              '{}.text'.format(file.file_id))

                if os.path.exists(file.copy_path):
                    logger.info('{} already processed, output file: {}, skip'
                                .format(file.origin_path, file.copy_path))
                    file.status = True
                    file.message = 'preprocessed'
                else:
                    try:
                        shutil.copy2(file.origin_path, file.copy_path)
                        file.status = True
                        file.message = 'preprocessed'
                    except Exception as e:
                        file.status = False
                        file.message = str(e)
            else:
                file.status = False
                file.message = 'skip unknown format'

        pool.close()
        logger.debug('waiting for preprocess read finish..')

        for file, result in async_results:
            try:
                result.get(timeout=timeout)
                if os.path.exists(file.copy_path):
                    file.status = True
                    file.message = 'preprocessed'
            except multiprocessing.TimeoutError:
                logger.error(f"Task timeout for file {file.origin_path}")
                file.status = False
                file.message = 'Task timeout'
            except Exception as e:
                logger.error(f"Error processing file {file.origin_path}: {str(e)}")
                file.status = False
                file.message = str(e)

        pool.join()

        # check process result
        for file in files:
            if file._category in ['pdf', 'word', 'excel', 'json']:
                if os.path.exists(file.copy_path):
                    file.status = True
                    file.message = 'preprocessed'
                else:
                    file.status = False
                    file.message = 'read error'