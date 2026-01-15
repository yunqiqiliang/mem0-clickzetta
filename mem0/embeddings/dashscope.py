import os
from typing import Literal, Optional

from openai import OpenAI

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase


class DashScopeEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        # Set default model and dimensions for DashScope
        self.config.model = self.config.model or "text-embedding-v1"
        self.config.embedding_dims = self.config.embedding_dims or 1536

        # Get API key from config or environment
        api_key = self.config.api_key or os.getenv("DASHSCOPE_API_KEY")

        # Set base URL for DashScope API
        base_url = (
            self.config.dashscope_base_url
            or os.getenv("DASHSCOPE_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        # Initialize OpenAI client with DashScope endpoint
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        Get the embedding for the given text using DashScope.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """
        text = text.replace("\n", " ")

        # Ensure model and dimensions are not None
        model = self.config.model or "text-embedding-v1"
        dimensions = self.config.embedding_dims or 1536

        return (
            self.client.embeddings.create(
                input=[text],
                model=model,
                dimensions=dimensions
            )
            .data[0]
            .embedding
        )