
import fitz
import re
import os
import time
import pandas as pd
import textract
import shutil
import subprocess
import asyncio
from loguru import logger
from bs4 import BeautifulSoup
from recursive_character_text_splitter import RecursiveCharacterTextSplitter
from markdown_text_splitter import MarkdownParser
from tools import get_partial_sha256_hash


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
                logger.info(category)
                if category is not None:
                    documents.append(
                        DocumentName(directory=directory,
                                     name=name,
                                     category=category))
        return documents

    def read(self, filepath: str):

        text = ''
        if not os.path.exists(filepath):
            error_message = f"Error: File '{filepath}' does not exist."
            return text, error_message

        file_type = self.read_file_type(filepath)

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
                text += self.read_word(filepath)

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

    def read_word(self, filepath: str):
        errors = []

        try:
            result = subprocess.run(['antiword', filepath],
                                    capture_output=True,
                                    text=True)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout
            errors.append("Antiword failed to extract content")
        except Exception as e:
            errors.append(f"Antiword error: {str(e)}")

        try:
            content = textract.process(filepath)
            try:
                text = content.decode('utf8')
            except UnicodeDecodeError:
                text = content.decode('gbk')
            if text.strip():
                return text
            errors.append("Textract failed to extract content")
        except Exception as e:
            errors.append(f"Textract error: {str(e)}")

        try:
            process = subprocess.Popen(['catdoc', filepath],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            for encoding in ['utf-8', 'gbk', 'gb18030']:
                try:
                    text = stdout.decode(encoding)
                    if text.strip():
                        return text
                except UnicodeDecodeError:
                    continue
            errors.append("Catdoc failed to extract content")
        except Exception as e:
            errors.append(f"Catdoc error: {str(e)}")

        error_msg = "\n".join(errors)
        raise Exception(f"Failed to read word file using all methods:\n{error_msg}")

    def read_excel(self, filepath: str):
        try:
            if filepath.endswith('.csv'):
                table = pd.read_csv(filepath)
            elif filepath.endswith('.xls'):
                table = pd.read_excel(filepath, engine='xlrd')
            elif filepath.endswith(('.xlsx', '.xlsm')):
                table = pd.read_excel(filepath, engine='openpyxl')
            else:
                raise ValueError("Unsupported file format")
            json_text = table.dropna(axis=1).to_json(force_ascii=False)
            return json_text
        except Exception as e:
            logger.error(f"Error reading {filepath}: {str(e)}")
            return ''

    def read_pdf(self, filepath: str):
        # load pdf and serialize table

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
                             ocr_process_dir=None
                             ):
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
            chunk_size=1068, chunk_overlap=100)
        self.md_splitter = MarkdownParser(max_chunk_size=1200)

    async def build_database(self, files: list,
                             file_opr: DocumentProcessor,
                             knowledge_base_id: str,
                             is_md_splitter: bool = False,
                             elastic_search=None,
                             milvus=None,
                             vector_model_name: str = None,
                             chunk_size: int = None,
                             chunk_size_overlap: int = None
                             ):
        error_msg = ""

        if chunk_size and chunk_size_overlap:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_size_overlap)
            self.md_splitter = MarkdownParser(max_chunk_size=chunk_size)

        if is_md_splitter:
            self.splitter = self.md_splitter
            splitter_type = "MarkdownParser"
        else:
            self.splitter = self.text_splitter
            splitter_type = "RecursiveCharacterTextSplitter"

        content_list = []
        file_id_list = []
        unique_file_id_list = []
        file_name_list = []

        es_success_files = []
        es_failed_files = []
        milvus_success_files = []
        milvus_failed_files = []

        for i, file in enumerate(files):
            try:
                if not file.status:
                    continue

                start_read = time.time()
                text, error = file_opr.read(file.copy_path)
                read_time = time.time() - start_read
                logger.info(
                    f"File {file.basename} read time: {read_time:.2f}s, size: {os.path.getsize(file.copy_path) / 1024 / 1024:.2f}MB")

                if error is not None:
                    file.status = False
                    file.message = str(error)
                    error_msg = f"File read error for {file.basename}: {str(error)}"
                    continue

                file.message = str(text[0])
                unique_file_id_list.append(f"{file.file_id}")

                logger.info(f"Starting create documents using {splitter_type} splitter!")
                splitter_time_before = time.time()
                try:

                    loop = asyncio.get_running_loop()
                    final_chunks, content_list_temp, hash_list_temp = await loop.run_in_executor(
                        None,
                        lambda: self.splitter.create_documents(text)
                    )
                except Exception as e:
                    error_msg = f"Document splitting error for {file.basename}: {str(e)}"
                    logger.error(f"Splitting error for file {file.basename}: {str(e)}")
                    continue

                splitter_time_after = time.time()
                logger.debug(
                    'Splitter take time: {} '.format(splitter_time_after - splitter_time_before))

                content_list.extend(content_list_temp)
                file_id_list.extend([file.file_id for _ in range(len(final_chunks))])
                hash_list = hash_list_temp if not 'hash_list' in locals() else hash_list + hash_list_temp

                logger.debug('Pipeline {}/{}.. register 《{}》 and split {} chunks'
                             .format(i + 1, len(files), file.basename, len(final_chunks)))

            except Exception as e:
                error_msg = f"Processing error for {file.basename}: {str(e)}"
                logger.error(f"Error processing file {file.basename}: {str(e)}")
                return [], list(set(f.file_id for f in files)), error_msg

        file_name_list = [f"{file.basename}" for _ in range(len(final_chunks))]

        if not content_list:
            error_msg = "No valid files to process - all files were skipped due to missing copy paths"
            logger.warning(error_msg)
            return [], list(set(f.file_id for f in files)), error_msg

        if milvus is not None:
            try:
                logger.debug('Milvus pipeline: registering {} files (total {} chunks) to knowledge base {}'.format(
                    len(unique_file_id_list),
                    len(content_list),
                    knowledge_base_id
                ))
                milvus_time_before_register = time.time()
                try:
                    success_ids = await milvus.add_texts(
                        knowledge_base_id=knowledge_base_id,
                        vector_model_name=vector_model_name,
                        texts=content_list,
                        file_ids=file_id_list,
                        chunk_ids=hash_list,
                        file_name_list=file_name_list
                    )
                finally:
                    await milvus.close()
                milvus_time_after_register = time.time()
                logger.debug(
                    'Milvus pipeline take time: {} '.format(milvus_time_after_register - milvus_time_before_register))

                success_file_ids = list(dict.fromkeys("_".join(id.split("_")[5:8]) for id in success_ids))
                milvus_failed_files = list(set(unique_file_id_list) - set(success_file_ids))
                milvus_success_files = success_file_ids

                logger.debug(
                    'Milvus pipeline successfully inserted files count: {}, file ids: {}'.format(
                        len(milvus_success_files),
                        milvus_success_files))
                logger.debug(
                    'Milvus pipeline failed inserted files count: {}, file ids: {}'.format(len(milvus_failed_files),
                                                                                           milvus_failed_files))
            except Exception as e:
                error_msg = f"Milvus insertion error: {str(e)}"
                logger.error(f"Milvus add_texts error: {str(e)}")
                return [], list(set(f.file_id for f in files)), error_msg

        if elastic_search is not None:
            try:
                logger.debug('ES pipeline: registering {} files (total {} chunks) to knowledge base {}'.format(
                    len(unique_file_id_list),
                    len(content_list),
                    knowledge_base_id
                ))
                es_time_before_register = time.time()
                ids, es_failed_files = elastic_search.add_texts(knowledge_base_id=knowledge_base_id,
                                                                texts=content_list,
                                                                file_id_list=file_id_list,
                                                                chunks_ids_list=hash_list,
                                                                file_name_list=file_name_list)
                es_time_after_register = time.time()
                logger.debug('ES pipeline take time: {} '.format(es_time_after_register - es_time_before_register))

                es_success_files = [file_id for file_id in unique_file_id_list if file_id not in es_failed_files]

                logger.debug(
                    'ES pipeline successfully inserted files count: {}, file ids: {}'.format(len(es_success_files),
                                                                                             es_success_files))
                logger.debug(
                    'ES pipeline failed inserted files count: {}, file ids: {}'.format(len(es_failed_files),
                                                                                       es_failed_files))
            except Exception as e:
                error_msg = f"Elasticsearch insertion error: {str(e)}"
                logger.error(f"ES pipeline error: {str(e)}")
                return [], list(set(f.file_id for f in files)), error_msg

        successful_files = list(set(milvus_success_files) & set(es_success_files)) if (
                milvus is not None and elastic_search is not None) else (
            milvus_success_files if milvus is not None else es_success_files)

        failed_files = list(set(milvus_failed_files) | set(es_failed_files)) if (
                milvus is not None and elastic_search is not None) else (
            milvus_failed_files if milvus is not None else es_failed_files)

        return successful_files, failed_files, error_msg

    def preprocess(self, files: list,
                   work_dir: str,
                   file_opr: DocumentProcessor,
                   is_md_splitter: bool = False,
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
                    try:
                        processed_knowledge_id_dir, file_name_with_ext = os.path.split(work_dir)
                        file.copy_path = os.path.join(work_dir,
                                                      file_name_with_ext + '.txt')
                        if is_md_splitter:
                            file.copy_path = os.path.join(work_dir,
                                                          file_name_with_ext + '.md')

                        read_pdf_from_server(file, server, processed_knowledge_id_dir)
                    except Exception as e:
                        file.status = False
                        file.message = f'pdf processing error: {str(e)}'
                        raise Exception(f"Failed to process PDF file {file.file_id}: {str(e)}")
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

    def initialize(self,
                   files: list,
                   work_dir: str,
                   file_opr: DocumentProcessor,
                   is_md_splitter: bool,
                   knowledge_base_id: str,
                   elastic_search=None,
                   milvus=None,
                   server=None):

        self.preprocess(files=files,
                        work_dir=work_dir,
                        file_opr=file_opr,
                        server=server)
        self.build_database(files=files,
                            file_opr=file_opr,
                            knowledge_base_id=knowledge_base_id,
                            is_md_splitter=is_md_splitter,
                            elastic_search=elastic_search,
                            milvus=milvus
                            )