from typing import Any, ClassVar, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ClickZettaConfig(BaseModel):
    """Configuration for ClickZetta vector store."""

    collection_name: str = Field("mem0", description="Name of the collection (table)")
    embedding_model_dims: Optional[int] = Field(1536, description="Dimensions of the embedding model")

    # ClickZetta connection parameters
    service: str = Field(..., description="ClickZetta service name")
    instance: str = Field(..., description="ClickZetta instance name")
    workspace: str = Field(..., description="ClickZetta workspace name")
    database_schema: str = Field(..., description="ClickZetta schema name", alias="schema")
    username: str = Field(..., description="Username for authentication")
    password: str = Field(..., description="Password for authentication")
    vcluster: str = Field(..., description="Virtual cluster name")

    # Connection settings
    connection_timeout: int = Field(30, description="Connection timeout in seconds")
    query_timeout: int = Field(300, description="Query timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retry attempts for failed operations")
    retry_delay: float = Field(1.0, description="Initial delay between retries in seconds")
    retry_backoff: float = Field(2.0, description="Backoff multiplier for retry delay")

    # Table column names
    embedding_column: str = Field("embedding", description="Name of the embedding column")
    content_column: str = Field("content", description="Name of the content column")
    metadata_column: str = Field("metadata", description="Name of the metadata column")
    id_column: str = Field("id", description="Name of the ID column")

    # Vector settings
    distance_metric: str = Field("cosine", description="Distance metric (cosine, euclidean, manhattan)")
    vector_element_type: str = Field("float", description="Vector element type (float, int, tinyint)")

    # Performance settings
    batch_size: int = Field(1000, description="Batch size for bulk operations")
    enable_vector_index: bool = Field(True, description="Whether to create vector index automatically")
    index_build_timeout: int = Field(600, description="Timeout for index building in seconds")

    # Logging and debugging
    enable_query_logging: bool = Field(False, description="Enable detailed query logging")
    log_level: str = Field("INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)")

    # Optional existing engine instance
    engine: Optional[Any] = Field(None, description="Existing ClickZettaEngine instance")

    @model_validator(mode="before")
    @classmethod
    def validate_required_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that all required connection parameters are provided."""
        required_fields = ["service", "instance", "workspace", "username", "password", "vcluster"]

        # Handle both 'schema' and 'database_schema' field names
        if not values.get("schema") and not values.get("database_schema"):
            raise ValueError("'schema' (or 'database_schema') is required for ClickZetta connection")

        for field in required_fields:
            if not values.get(field):
                raise ValueError(f"'{field}' is required for ClickZetta connection")

        return values

    @model_validator(mode="before")
    @classmethod
    def validate_distance_metric(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate distance metric."""
        distance_metric = values.get("distance_metric", "cosine")
        valid_metrics = ["cosine", "euclidean", "l2", "manhattan"]

        if distance_metric not in valid_metrics:
            raise ValueError(f"distance_metric must be one of {valid_metrics}, got '{distance_metric}'")

        return values

    @model_validator(mode="before")
    @classmethod
    def validate_vector_element_type(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate vector element type."""
        vector_element_type = values.get("vector_element_type", "float")
        valid_types = ["float", "int", "tinyint"]

        if vector_element_type not in valid_types:
            raise ValueError(f"vector_element_type must be one of {valid_types}, got '{vector_element_type}'")

        return values

    @model_validator(mode="before")
    @classmethod
    def validate_performance_settings(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate performance settings."""
        batch_size = values.get("batch_size", 1000)
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        max_retries = values.get("max_retries", 3)
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        retry_delay = values.get("retry_delay", 1.0)
        if retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")

        retry_backoff = values.get("retry_backoff", 2.0)
        if retry_backoff < 1.0:
            raise ValueError("retry_backoff must be >= 1.0")

        return values

    @model_validator(mode="before")
    @classmethod
    def validate_log_level(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate log level."""
        log_level = values.get("log_level", "INFO")
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        if log_level not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}, got '{log_level}'")

        return values

    @model_validator(mode="before")
    @classmethod
    def validate_extra_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that no extra fields are provided."""
        allowed_fields = set(cls.model_fields.keys())
        input_fields = set(values.keys())
        extra_fields = input_fields - allowed_fields

        if extra_fields:
            raise ValueError(
                f"Extra fields not allowed: {', '.join(extra_fields)}. "
                f"Please input only the following fields: {', '.join(allowed_fields)}"
            )

        return values

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)