import typing
from typing import Any, Dict, List, Tuple, TypedDict, Optional
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from abc import ABC, abstractmethod
from tqdm.auto import tqdm
import hashlib


class LineType(TypedDict):
    """Line type as typed dict."""
    metadata: Dict[str, str]
    content: str


class HeaderType(TypedDict):
    """Header type as typed dict."""
    level: int
    name: str
    data: str


class MarkdownTextSplitter(RecursiveCharacterTextSplitter):
    """Attempts to split the text along Markdown-formatted headings."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a MarkdownTextSplitter."""
        separators = self.get_separators_for_language("markdown")
        super().__init__(separators=separators, **kwargs)


class MarkdownHeaderTextSplitter:
    """Splitting markdown files based on specified headers."""

    def __init__(
            self,
            headers_to_split_on: List[Tuple[str, str]],
            return_each_line: bool = False,
            strip_headers: bool = True,
    ):
        """Create a new MarkdownHeaderTextSplitter.

        Args:
            headers_to_split_on: Headers we want to track
            return_each_line: Return each line w/ associated headers
            strip_headers: Strip split headers from the content of the chunk
        """
        # Output line-by-line or aggregated into chunks w/ common headers
        self.return_each_line = return_each_line
        # Given the headers we want to split on,
        # (e.g., "#, ##, etc") order by length
        self.headers_to_split_on = sorted(
            headers_to_split_on, key=lambda split: len(split[0]), reverse=True
        )
        # Strip headers split headers from the content of the chunk
        self.strip_headers = strip_headers

    def aggregate_lines_to_chunks(self, lines: List[LineType]) -> List[Document]:
        """Combine lines with common metadata into chunks
        Args:
            lines: Line of text / associated header metadata
        """
        aggregated_chunks: List[LineType] = []

        for line in lines:
            if (
                    aggregated_chunks
                    and aggregated_chunks[-1]["metadata"] == line["metadata"]
            ):
                # If the last line in the aggregated list
                # has the same metadata as the current line,
                # append the current content to the last lines's content
                aggregated_chunks[-1]["content"] += "  \n" + line["content"]
            elif (
                    aggregated_chunks
                    and aggregated_chunks[-1]["metadata"] != line["metadata"]
                    # may be issues if other metadata is present
                    and len(aggregated_chunks[-1]["metadata"]) < len(line["metadata"])
                    and aggregated_chunks[-1]["content"].split("\n")[-1][0] == "#"
                    and not self.strip_headers
            ):
                # If the last line in the aggregated list
                # has different metadata as the current line,
                # and has shallower header level than the current line,
                # and the last line is a header,
                # and we are not stripping headers,
                # append the current content to the last line's content
                aggregated_chunks[-1]["content"] += "  \n" + line["content"]
                # and update the last line's metadata
                aggregated_chunks[-1]["metadata"] = line["metadata"]
            else:
                # Otherwise, append the current line to the aggregated list
                aggregated_chunks.append(line)

        return [
            Document(page_content=chunk["content"], metadata=chunk["metadata"])
            for chunk in aggregated_chunks
        ]

    def split_text(self, text: str) -> List[Document]:
        """Split markdown file
        Args:
            text: Markdown file"""

        # Split the input text by newline character ("\n").
        lines = text.split("\n")
        # Final output
        lines_with_metadata: List[LineType] = []
        # Content and metadata of the chunk currently being processed
        current_content: List[str] = []
        current_metadata: Dict[str, str] = {}
        # Keep track of the nested header structure
        # header_stack: List[Dict[str, Union[int, str]]] = []
        header_stack: List[HeaderType] = []
        initial_metadata: Dict[str, str] = {}

        in_code_block = False
        opening_fence = ""

        for line in lines:
            stripped_line = line.strip()
            # Remove all non-printable characters from the string, keeping only visible
            # text.
            stripped_line = "".join(filter(str.isprintable, stripped_line))
            if not in_code_block:
                # Exclude inline code spans
                if stripped_line.startswith("```") and stripped_line.count("```") == 1:
                    in_code_block = True
                    opening_fence = "```"
                elif stripped_line.startswith("~~~"):
                    in_code_block = True
                    opening_fence = "~~~"
            else:
                if stripped_line.startswith(opening_fence):
                    in_code_block = False
                    opening_fence = ""

            if in_code_block:
                current_content.append(stripped_line)
                continue

            # Check each line against each of the header types (e.g., #, ##)
            for sep, name in self.headers_to_split_on:
                # Check if line starts with a header that we intend to split on
                if stripped_line.startswith(sep) and (
                        # Header with no text OR header is followed by space
                        # Both are valid conditions that sep is being used a header
                        len(stripped_line) == len(sep) or stripped_line[len(sep)] == " "
                ):
                    # Ensure we are tracking the header as metadata
                    if name is not None:
                        # Get the current header level
                        current_header_level = sep.count("#")

                        # Pop out headers of lower or same level from the stack
                        while (
                                header_stack
                                and header_stack[-1]["level"] >= current_header_level
                        ):
                            # We have encountered a new header
                            # at the same or higher level
                            popped_header = header_stack.pop()
                            # Clear the metadata for the
                            # popped header in initial_metadata
                            if popped_header["name"] in initial_metadata:
                                initial_metadata.pop(popped_header["name"])

                        # Push the current header to the stack
                        header: HeaderType = {
                            "level": current_header_level,
                            "name": name,
                            "data": stripped_line[len(sep):].strip(),
                        }
                        header_stack.append(header)
                        # Update initial_metadata with the current header
                        initial_metadata[name] = header["data"]

                    # Add the previous line to the lines_with_metadata
                    # only if current_content is not empty
                    if current_content:
                        lines_with_metadata.append(
                            {
                                "content": "\n".join(current_content),
                                "metadata": current_metadata.copy(),
                            }
                        )
                        current_content.clear()

                    if not self.strip_headers:
                        current_content.append(stripped_line)

                    break
            else:
                if stripped_line:
                    current_content.append(stripped_line)
                elif current_content:
                    lines_with_metadata.append(
                        {
                            "content": "\n".join(current_content),
                            "metadata": current_metadata.copy(),
                        }
                    )
                    current_content.clear()

            current_metadata = initial_metadata.copy()

        if current_content:
            lines_with_metadata.append(
                {"content": "\n".join(current_content), "metadata": current_metadata}
            )

        # lines_with_metadata has each line with associated header metadata
        # aggregate these into chunks based on common metadata
        if not self.return_each_line:
            return self.aggregate_lines_to_chunks(lines_with_metadata)
        else:
            return [
                Document(page_content=chunk["content"], metadata=chunk["metadata"])
                for chunk in lines_with_metadata
            ]


class BaseParser(ABC):
    """
    BaseParser is an Abstract Base Class (ABC) that serves as a template for all parser objects.
    It contains the common attributes and methods that each parser should implement.
    """

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    async def get_chunks(
            self,
            filepath: str,
            metadata: Optional[dict],
            *args,
            **kwargs,
    ) -> typing.List[Document]:
        """
        Abstract method. This should asynchronously read a file and return its content in chunks.

        Parameters:
            loaded_data_point (LoadedDataPoint): Loaded Document to read and parse.

        Returns:
            typing.List[Document]: A list of Document objects, each representing a chunk of the file.
        """
        pass


def contains_text(text):
    # Check if the token contains at least one alphanumeric character
    return any(char.isalnum() for char in text)


class MarkdownParser(BaseParser):
    """
    Custom Markdown parser for extracting chunks from Markdown files.
    """

    max_chunk_size: int
    supported_file_extensions = [".md"]

    def __init__(self, max_chunk_size: int = 512, *args, **kwargs):
        """
        Initializes the MarkdownParser object.
        """
        self.max_chunk_size = max_chunk_size

    def create_documents(
            self,
            content,
    ) -> typing.List[Document]:
        chunks = []
        if type(content) == list:
            for one_content in content:
                chunks.extend(self.get_chunks(one_content))
            return chunks
        else:
            return self.get_chunks(content)

    def get_chunks(
            self,
            content: str,
            *args,
            **kwargs,
    ) -> typing.Tuple[typing.List[Document], typing.List[str], typing.List[str]]:
        """
        Extracts chunks of content from a given Markdown file.
        Returns:
            - final_chunks: List[Document] - All Document
            - content_list: List[str] - All page_content
            - hash_list: List[str] - All hash_value
        """

        headers_to_split_on = [
            ("#", "Header1"),
            ("##", "Header2"),
            ("###", "Header3"),
            ("####", "Header4"),
        ]
        chunks_arr = self._recurse_split(
            content, {}, 0, headers_to_split_on, self.max_chunk_size
        )
        final_chunks = []
        content_list = []
        hash_list = []
        lastAddedChunkSize = self.max_chunk_size + 1

        for chunk in tqdm(chunks_arr, desc="Processing markdown splitter"):
            page_content = self._include_headers_in_content(
                content=chunk.page_content,
                metadata=chunk.metadata,
            )
            chunk_length = len(page_content)

            if chunk_length + lastAddedChunkSize <= self.max_chunk_size:
                lastAddedChunk: Document = final_chunks.pop()
                content_list.pop()
                hash_list.pop()

                lastAddedChunk.page_content = (
                    f"{lastAddedChunk.page_content}\n{page_content}"
                )
                hash_value = hashlib.sha256(
                    lastAddedChunk.page_content.encode('utf-8')
                ).hexdigest()[:16]
                lastAddedChunk.metadata['hash_value'] = hash_value

                final_chunks.append(lastAddedChunk)
                content_list.append(lastAddedChunk.page_content)
                hash_list.append(hash_value)
                lastAddedChunkSize = chunk_length + lastAddedChunkSize
                continue

            lastAddedChunkSize = chunk_length
            if contains_text(page_content):
                hash_value = hashlib.sha256(
                    page_content.encode('utf-8')
                ).hexdigest()[:16]
                chunk.metadata['hash_value'] = hash_value

                new_doc = Document(
                    page_content=page_content,
                    metadata=chunk.metadata,
                )
                final_chunks.append(new_doc)
                content_list.append(page_content)
                hash_list.append(hash_value)

        return final_chunks, content_list, hash_list

    def _include_headers_in_content(self, content: str, metadata: dict):

        if "Header4" in metadata and metadata["Header4"] not in content:
            content = "#### " + metadata["Header4"] + "\n" + content
        elif "Header3" in metadata and metadata["Header3"] not in content:
            content = "### " + metadata["Header3"] + "\n" + content
        elif "Header2" in metadata and metadata["Header2"] not in content:
            content = "## " + metadata["Header2"] + "\n" + content
        elif "Header1" in metadata and metadata["Header1"] not in content:
            content = "# " + metadata["Header1"] + "\n" + content

        return content

    def _recurse_split(self, content, metadata, i, headers_to_split_on, max_chunk_size):
        headers_len = len(headers_to_split_on)
        if i >= len(headers_to_split_on):
            # Use Markdown Text Splitter in this case
            text_splitter = MarkdownTextSplitter(
                chunk_size=max_chunk_size,
                chunk_overlap=0,
                length_function=len,
            )
            texts = text_splitter.split_text(content)
            chunks_arr = [
                Document(
                    page_content=text,
                    metadata=metadata,
                )
                for text in texts
                if contains_text(text)
            ]
            return chunks_arr
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on[i:i + 1],
            strip_headers=False
        )
        md_header_splits = markdown_splitter.split_text(content)
        chunks_arr = []
        for document in md_header_splits:
            document.metadata.update(metadata)
            chunk_length = len(document.page_content)
            if chunk_length <= max_chunk_size and contains_text(document.page_content):
                chunks_arr.append(
                    Document(
                        page_content=document.page_content,
                        metadata=document.metadata,
                    )
                )
                continue
            chunks_arr.extend(
                self._recurse_split(
                    document.page_content,
                    document.metadata,
                    i + 1,
                    headers_to_split_on,
                    max_chunk_size,
                )
            )
        return chunks_arr