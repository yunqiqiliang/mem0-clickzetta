import hashlib
import json
import logging
import re
import time
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

try:
    import clickzetta
    from clickzetta.zettapark.session import Session
except ImportError:
    raise ImportError(
        "The 'clickzetta-connector-python' and 'clickzetta-zettapark-python' libraries are required. "
        "Please install them using 'pip install clickzetta-connector-python clickzetta-zettapark-python'."
    )

from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class ClickZettaConnectionError(Exception):
    """Raised when connection to ClickZetta fails."""
    pass


class ClickZettaQueryError(Exception):
    """Raised when query execution fails."""
    pass


class ClickZettaConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


def retry_on_connection_error(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry operations on connection errors.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for delay
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (ClickZettaConnectionError, ConnectionError) as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Connection error on attempt {attempt + 1}/{max_retries + 1}: {e}")
                        logger.info(f"Retrying in {current_delay} seconds...")
                        time.sleep(current_delay)
                        current_delay *= backoff

                        # Try to reconnect if it's a ClickZetta instance
                        if hasattr(args[0], '_reconnect'):
                            try:
                                args[0]._reconnect()
                            except Exception as reconnect_error:
                                logger.warning(f"Reconnection failed: {reconnect_error}")
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed")
                        break

            raise last_exception

        return wrapper
    return decorator


class OutputData(BaseModel):
    id: Optional[str] = None
    score: Optional[float] = None
    payload: Optional[Dict] = None


class ClickZettaEngine:
    """ClickZetta database engine for managing connections and queries."""

    def __init__(
        self,
        service: str,
        instance: str,
        workspace: str,
        schema: str,
        username: str,
        password: str,
        vcluster: str,
        connection_timeout: int = 30,
        query_timeout: int = 300,
    ):
        """
        Initialize ClickZetta engine.

        Args:
            service: ClickZetta service name
            instance: ClickZetta instance name
            workspace: ClickZetta workspace name
            schema: ClickZetta schema name
            username: Username for authentication
            password: Password for authentication
            vcluster: Virtual cluster name
            connection_timeout: Connection timeout in seconds
            query_timeout: Query timeout in seconds
        """
        self.service = service
        self.instance = instance
        self.workspace = workspace
        self.schema = schema
        self.username = username
        self.password = password
        self.vcluster = vcluster
        self.connection_timeout = connection_timeout
        self.query_timeout = query_timeout

        # Create session
        self.session = self._create_session()

    def _reconnect(self):
        """Reconnect to ClickZetta database."""
        try:
            if hasattr(self, 'session') and self.session:
                try:
                    self.session.close()
                except Exception:
                    pass  # Ignore errors when closing old session

            self.session = self._create_session()
            logger.info("Successfully reconnected to ClickZetta")
        except Exception as e:
            logger.error(f"Failed to reconnect: {e}")
            raise ClickZettaConnectionError(f"Reconnection failed: {e}") from e

    def _create_session(self) -> Session:
        """Create a ClickZetta session."""
        try:
            session = clickzetta.connect(
                service=self.service,
                instance=self.instance,
                workspace=self.workspace,
                schema=self.schema,
                username=self.username,
                password=self.password,
                vcluster=self.vcluster,
                connection_timeout=self.connection_timeout,
                query_timeout=self.query_timeout,
            )
            logger.info(f"Connected to ClickZetta: {self.service}.{self.instance}.{self.workspace}.{self.schema}")
            return session
        except ConnectionError as e:
            logger.error(f"Failed to connect to ClickZetta: {e}")
            raise ClickZettaConnectionError(f"Connection failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error connecting to ClickZetta: {e}")
            raise ClickZettaConnectionError(f"Unexpected connection error: {e}") from e

    @retry_on_connection_error(max_retries=3, delay=1.0, backoff=2.0)
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a query and return results."""
        try:
            logger.debug(f"Executing query: {query}")

            # Use cursor to execute query
            cursor = self.session.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            # Get results and column names
            results = cursor.fetchall()
            columns = (
                [desc[0] for desc in cursor.description] if cursor.description else []
            )

            # Convert result to list of dictionaries
            if results and columns:
                # Convert rows to dictionaries using column names
                records = []
                for row in results:
                    if isinstance(row, (tuple, list)):
                        # Convert tuple/list to dict using column names
                        record = dict(zip(columns, row))
                    else:
                        # Row is already a dict-like object
                        record = dict(row)
                    records.append(record)
            else:
                records = []

            logger.debug(f"Query executed successfully, returned {len(records)} rows")
            return records

        except ConnectionError as e:
            logger.error(f"Database connection lost: {e}")
            raise ClickZettaConnectionError(f"Connection lost during query: {e}") from e
        except ValueError as e:
            logger.error(f"Invalid query parameters: {e}")
            raise ClickZettaQueryError(f"Invalid parameters: {e}") from e
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise ClickZettaQueryError(f"Query failed: {e}") from e

    def close(self):
        """Close the session."""
        if hasattr(self, 'session') and self.session:
            try:
                self.session.close()
                logger.info("ClickZetta session closed")
            except Exception as e:
                logger.warning(f"Error closing session: {e}")


class ClickZetta(VectorStoreBase):
    """ClickZetta vector store implementation for Mem0."""

    def __init__(
        self,
        collection_name: str,
        embedding_model_dims: int,
        service: str,
        instance: str,
        workspace: str,
        username: str,
        password: str,
        vcluster: str,
        schema: Optional[str] = None,
        database_schema: Optional[str] = None,
        engine: Optional[ClickZettaEngine] = None,
        embedding_column: str = "embedding",
        content_column: str = "content",
        metadata_column: str = "metadata",
        id_column: str = "id",
        distance_metric: str = "cosine",
        vector_element_type: str = "float",
        connection_timeout: int = 30,
        query_timeout: int = 300,
        batch_size: int = 1000,
        enable_vector_index: bool = True,
        index_build_timeout: int = 600,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0,
        enable_query_logging: bool = False,
        log_level: str = "INFO",
    ):
        """
        Initialize ClickZetta vector store.

        Args:
            collection_name: Name of the table/collection
            embedding_model_dims: Dimension of embedding vectors
            service: ClickZetta service name
            instance: ClickZetta instance name
            workspace: ClickZetta workspace name
            schema: ClickZetta schema name
            username: Username for authentication
            password: Password for authentication
            vcluster: Virtual cluster name
            engine: Optional existing ClickZettaEngine instance
            embedding_column: Name of the embedding column
            content_column: Name of the content column
            metadata_column: Name of the metadata column
            id_column: Name of the ID column
            distance_metric: Distance metric (cosine, euclidean, manhattan)
            vector_element_type: Vector element type (float, int, tinyint)
            connection_timeout: Connection timeout in seconds
            query_timeout: Query timeout in seconds
        """
        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.embedding_column = embedding_column
        self.content_column = content_column
        self.metadata_column = metadata_column
        self.id_column = id_column
        self.distance_metric = distance_metric.lower()
        self.vector_element_type = vector_element_type
        self.batch_size = batch_size
        self.enable_vector_index = enable_vector_index
        self.index_build_timeout = index_build_timeout
        self.enable_query_logging = enable_query_logging
        self.log_level = log_level

        # Set up logging level if specified
        if log_level and hasattr(logging, log_level.upper()):
            logger.setLevel(getattr(logging, log_level.upper()))

        # Handle schema parameter - support both 'schema' and 'database_schema'
        actual_schema = database_schema or schema
        if not actual_schema:
            raise ValueError("Either 'schema' or 'database_schema' parameter must be provided")

        # Validate distance metric
        if self.distance_metric not in ["cosine", "euclidean", "l2", "manhattan"]:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")

        # Create or use existing engine
        if engine:
            self.engine = engine
        else:
            self.engine = ClickZettaEngine(
                service=service,
                instance=instance,
                workspace=workspace,
                schema=actual_schema,
                username=username,
                password=password,
                vcluster=vcluster,
                connection_timeout=connection_timeout,
                query_timeout=query_timeout,
            )

        # Set up retry configuration for the engine
        if hasattr(self.engine, 'max_retries'):
            self.engine.max_retries = max_retries
            self.engine.retry_delay = retry_delay
            self.engine.retry_backoff = retry_backoff

        # Create collection if it doesn't exist
        if self.enable_vector_index:
            self.create_col(collection_name, embedding_model_dims, distance_metric)
        else:
            # Create table without vector index
            self._create_table_only(collection_name, embedding_model_dims)

    def _get_distance_function(self) -> str:
        """Get the appropriate distance function for ClickZetta."""
        distance_map = {
            "cosine": "COSINE_DISTANCE",
            "euclidean": "L2_DISTANCE",
            "l2": "L2_DISTANCE",
            "manhattan": "L1_DISTANCE",
        }
        return distance_map[self.distance_metric]

    def _format_vector(self, vector: List[float]) -> str:
        """Format vector for ClickZetta insertion."""
        formatted_values = [str(float(val)) for val in vector]
        return f"vector({','.join(formatted_values)})"

    def _parse_output(self, results: List[Dict]) -> List[OutputData]:
        """Parse query results into OutputData format."""
        output_data = []
        for result in results:
            output_data.append(
                OutputData(
                    id=str(result.get(self.id_column)),
                    score=result.get("score"),
                    payload=json.loads(result.get(self.metadata_column, "{}")) if result.get(self.metadata_column) else {},
                )
            )
        return output_data

    def _create_filter_conditions(self, filters: Dict[str, Any]) -> str:
        """Create WHERE clause conditions from filters."""
        if not filters:
            return ""

        conditions = []
        for key, value in filters.items():
            if isinstance(value, str):
                conditions.append(f"json_extract_string(CAST({self.metadata_column} AS JSON), '$.{key}') = '{value}'")
            elif isinstance(value, (int, float)):
                conditions.append(f"json_extract_float(CAST({self.metadata_column} AS JSON), '$.{key}') = {value}")
            elif isinstance(value, bool):
                conditions.append(f"json_extract_bool(CAST({self.metadata_column} AS JSON), '$.{key}') = {str(value).lower()}")
            elif isinstance(value, dict):
                # Handle range queries
                if "gte" in value and "lte" in value:
                    conditions.append(
                        f"json_extract_float(CAST({self.metadata_column} AS JSON), '$.{key}') >= {value['gte']} "
                        f"AND json_extract_float(CAST({self.metadata_column} AS JSON), '$.{key}') <= {value['lte']}"
                    )
                elif "gt" in value:
                    conditions.append(f"json_extract_float(CAST({self.metadata_column} AS JSON), '$.{key}') > {value['gt']}")
                elif "lt" in value:
                    conditions.append(f"json_extract_float(CAST({self.metadata_column} AS JSON), '$.{key}') < {value['lt']}")
                elif "eq" in value:
                    conditions.append(f"json_extract_string(CAST({self.metadata_column} AS JSON), '$.{key}') = '{value['eq']}'")

        return " AND ".join(conditions) if conditions else ""

    def create_col(self, name: str, vector_size: int, distance: str) -> None:
        """
        Create a new collection (table) with vector support.

        Args:
            name: Collection name
            vector_size: Size of the vectors
            distance: Distance metric (for compatibility, stored in self.distance_metric)
        """
        try:
            # Note: distance parameter is for interface compatibility
            # The actual distance metric is stored in self.distance_metric
            # Check if table already exists using SHOW TABLES with WHERE filter (more efficient)
            show_tables_query = f"SHOW TABLES WHERE table_name = '{name}'"
            tables_result = self.engine.execute_query(show_tables_query)

            if tables_result and len(tables_result) > 0:
                logger.debug(f"Table {name} already exists. Skipping creation.")
                return

            # Create table with vector column
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {name} (
                {self.id_column} String,
                {self.content_column} String,
                {self.metadata_column} String,
                {self.embedding_column} vector({self.vector_element_type}, {vector_size})
            )
            """

            self.engine.execute_query(create_table_query)
            logger.info(f"Created table {name}")

            # Create vector index
            self._create_vector_index(name)

        except Exception as e:
            logger.error(f"Error creating collection {name}: {e}")
            raise

    def _create_table_only(self, name: str, vector_size: int) -> None:
        """
        Create a table without vector index.

        Args:
            name: Table name
            vector_size: Size of the vectors
        """
        try:
            # Check if table already exists
            show_tables_query = f"SHOW TABLES WHERE table_name = '{name}'"
            tables_result = self.engine.execute_query(show_tables_query)

            if tables_result and len(tables_result) > 0:
                logger.debug(f"Table {name} already exists. Skipping creation.")
                return

            # Create table without vector index
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {name} (
                {self.id_column} String,
                {self.content_column} String,
                {self.metadata_column} String,
                {self.embedding_column} vector({self.vector_element_type}, {vector_size})
            )
            """

            self.engine.execute_query(create_table_query)
            logger.info(f"Created table {name} without vector index")

        except Exception as e:
            logger.error(f"Error creating table {name}: {e}")
            raise

    def _get_create_table_statement(self, table_name: str) -> str:
        """
        Get the CREATE TABLE statement for a table.

        Args:
            table_name: Name of the table

        Returns:
            CREATE TABLE statement as string

        Raises:
            ClickZettaQueryError: If unable to get table definition
        """
        try:
            show_create_query = f"SHOW CREATE TABLE {table_name}"
            results = self.engine.execute_query(show_create_query)

            if not results or len(results) == 0:
                raise ClickZettaQueryError(f"Could not get CREATE TABLE statement for {table_name}")

            # Extract CREATE TABLE statement from results
            for result in results:
                if isinstance(result, dict):
                    # Try common column names for CREATE TABLE statement
                    create_statement = (result.get('Create Table') or
                                      result.get('create_table') or
                                      result.get('statement') or
                                      result.get('Create_Table') or
                                      str(result))
                    if create_statement:
                        return create_statement

            raise ClickZettaQueryError(f"Could not extract CREATE TABLE statement from results: {results}")

        except Exception as e:
            raise ClickZettaQueryError(f"Error getting CREATE TABLE statement for {table_name}: {e}") from e

    def _check_vector_index_pattern(self, create_statement: str) -> bool:
        """
        Check if CREATE TABLE statement contains vector index pattern.

        Args:
            create_statement: The CREATE TABLE statement

        Returns:
            True if vector index exists, False otherwise
        """
        create_statement_upper = create_statement.upper()
        embedding_column_upper = self.embedding_column.upper()

        # Pattern to match: INDEX ... (`column_name`) Vector
        pattern = rf'INDEX\s+[^(]*\(\s*[`"]?{re.escape(embedding_column_upper)}[`"]?\s*\)[^)]*VECTOR'

        return bool(re.search(pattern, create_statement_upper, re.IGNORECASE | re.DOTALL))

    def _has_vector_index(self, table_name: str) -> bool:
        """
        Check if the table already has a vector index on the embedding column.

        Args:
            table_name: Name of the table to check

        Returns:
            True if vector index exists, False otherwise
        """
        try:
            # Use SHOW CREATE TABLE to get the table definition
            show_create_query = f"SHOW CREATE TABLE {table_name}"
            results = self.engine.execute_query(show_create_query)

            if not results or len(results) == 0:
                logger.warning(f"Could not get CREATE TABLE statement for {table_name}")
                return False

            # Get the CREATE TABLE statement
            # The result structure may vary, try different possible column names
            create_statement = ""
            for result in results:
                if isinstance(result, dict):
                    # Try common column names for CREATE TABLE statement
                    create_statement = (result.get('Create Table') or
                                      result.get('create_table') or
                                      result.get('statement') or
                                      result.get('Create_Table') or
                                      str(result))
                    break

            if not create_statement:
                logger.warning(f"Could not extract CREATE TABLE statement from results: {results}")
                return False

            logger.debug(f"CREATE TABLE statement for {table_name}: {create_statement}")

            # Check if there's an INDEX definition that includes the embedding column
            # Look for patterns like: INDEX `embedding_idx_xxx` (`embedding`) Vector
            create_statement_upper = create_statement.upper()
            embedding_column_upper = self.embedding_column.upper()

            # More precise pattern matching for vector index on the specific embedding column
            # Look for: INDEX ... (`embedding_column`) Vector

            # Pattern to match: INDEX ... (`column_name`) Vector
            # This pattern looks for INDEX followed by column name in parentheses, then Vector
            pattern = rf'INDEX\s+[^(]*\(\s*[`"]?{re.escape(embedding_column_upper)}[`"]?\s*\)[^)]*VECTOR'

            has_vector_index = bool(re.search(pattern, create_statement_upper, re.IGNORECASE | re.DOTALL))

            if has_vector_index:
                logger.info(f"Vector index already exists on column '{self.embedding_column}' for table {table_name}")
                return True
            else:
                logger.debug(f"No vector index found on column '{self.embedding_column}' for table {table_name}")
                return False

        except Exception as e:
            logger.warning(f"Error checking for existing vector index on {table_name}: {e}")
            # If we can't determine, assume no index exists to be safe
            return False

    def _create_vector_index(self, table_name: str):
        """Create vector index for the table."""
        try:
            # First check if vector index already exists
            if self._has_vector_index(table_name):
                logger.info(f"Vector index already exists for table {table_name}, skipping creation")
                return

            # Generate index name using table hash to avoid conflicts
            table_hash = hashlib.md5(table_name.encode()).hexdigest()
            index_name = f"embedding_idx_{table_hash[:8]}"

            # Try to create vector index, ignore if already exists
            create_index_query = f"""
            CREATE VECTOR INDEX IF NOT EXISTS {index_name} ON TABLE {table_name}({self.embedding_column})
            PROPERTIES(
                "scalar.type" = "f32",
                "distance.function" = "{self.distance_metric}_distance"
            )
            """

            try:
                self.engine.execute_query(create_index_query)
                logger.info(f"Created vector index {index_name}")

                # Build index
                build_index_query = f"BUILD INDEX {index_name} ON {table_name}"
                self.engine.execute_query(build_index_query)
                logger.info(f"Built vector index {index_name}")

            except Exception as index_error:
                error_msg = str(index_error).lower()
                # ClickZetta specific error message: "cannot create Vector index on column embedding, which already has index embedding_idx_xxx with the same type"
                if ("cannot create" in error_msg and "index" in error_msg and "already has index" in error_msg) or \
                   ("already exists" in error_msg) or \
                   ("duplicate" in error_msg and "index" in error_msg):
                    logger.debug(f"Vector index {index_name} already exists, skipping creation")
                else:
                    # Re-raise if it's not a "already exists" error
                    raise index_error

        except Exception as e:
            logger.warning(f"Error creating vector index: {e}")
            # Index creation is optional, continue without it

    def insert(self, vectors: List[List[float]], payloads: Optional[List[Dict]] = None, ids: Optional[List[str]] = None):
        """
        Insert vectors into the collection with optimized batch processing.

        Args:
            vectors: List of vectors to insert
            payloads: Optional list of metadata payloads
            ids: Optional list of IDs
        """
        try:
            logger.info(f"Inserting {len(vectors)} vectors into collection {self.collection_name}")

            # Process in batches for better performance
            batch_size = self.batch_size  # Use configured batch size
            total_vectors = len(vectors)

            for batch_start in range(0, total_vectors, batch_size):
                batch_end = min(batch_start + batch_size, total_vectors)
                batch_vectors = vectors[batch_start:batch_end]
                batch_payloads = payloads[batch_start:batch_end] if payloads else None
                batch_ids = ids[batch_start:batch_end] if ids else None

                self._insert_batch(batch_vectors, batch_payloads, batch_ids)

                logger.debug(f"Inserted batch {batch_start//batch_size + 1}/{(total_vectors + batch_size - 1)//batch_size}")

            logger.info(f"Successfully inserted {total_vectors} vectors")

        except Exception as e:
            logger.error(f"Error inserting vectors: {e}")
            raise

    def _insert_batch(self, vectors: List[List[float]], payloads: Optional[List[Dict]] = None, ids: Optional[List[str]] = None):
        """
        Insert a batch of vectors with optimized SQL generation.

        Args:
            vectors: Batch of vectors to insert
            payloads: Optional batch of metadata payloads
            ids: Optional batch of IDs
        """
        # Pre-allocate list for better performance
        data_rows = []

        for idx, vector in enumerate(vectors):
            vector_id = ids[idx] if ids else str(uuid.uuid4())
            payload = payloads[idx] if payloads else {}

            # Extract content from payload - mem0 uses "data" key as primary content field
            content = payload.get("data") or payload.get("content") or payload.get("text") or ""

            # Create metadata without the content field to avoid duplication
            metadata_payload = {k: v for k, v in payload.items() if k not in ["data", "content", "text"]}
            metadata_json = json.dumps(metadata_payload, separators=(',', ':'))  # Compact JSON

            # Use more efficient vector formatting
            formatted_vector = self._format_vector_optimized(vector)

            # Escape single quotes in content and metadata for SQL safety
            content_escaped = content.replace("'", "''")
            metadata_escaped = metadata_json.replace("'", "''")

            data_rows.append(f"('{vector_id}', '{content_escaped}', '{metadata_escaped}', {formatted_vector})")

        # Use single INSERT statement for the entire batch
        insert_query = f"""
        INSERT INTO {self.collection_name}
        ({self.id_column}, {self.content_column}, {self.metadata_column}, {self.embedding_column})
        VALUES {', '.join(data_rows)}
        """

        self.engine.execute_query(insert_query)

    def _format_vector_optimized(self, vector: List[float]) -> str:
        """
        Optimized vector formatting for better performance.

        Args:
            vector: Vector to format

        Returns:
            Formatted vector string
        """
        # Use list comprehension and join for better performance
        return f"vector({','.join(str(float(val)) for val in vector)})"

    def search(
        self,
        query: str,
        vectors: List[float],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[OutputData]:
        """
        Search for similar vectors.

        Args:
            query: Query string (for compatibility, not used in vector search)
            vectors: Query vector
            limit: Number of results to return
            filters: Optional filters to apply

        Returns:
            List of OutputData objects
        """
        try:
            # Note: query parameter is for interface compatibility but not used in vector search
            distance_function = self._get_distance_function()
            formatted_vector = self._format_vector(vectors)

            # Build query
            select_query = f"""
            SELECT
                {self.id_column},
                {self.content_column},
                {self.metadata_column},
                {distance_function}({self.embedding_column}, {formatted_vector}) AS distance
            FROM {self.collection_name}
            """

            # Add filters if provided
            filter_conditions = self._create_filter_conditions(filters) if filters else ""
            if filter_conditions:
                select_query += f" WHERE {filter_conditions}"

            select_query += f" ORDER BY distance ASC LIMIT {limit}"

            results = self.engine.execute_query(select_query)

            # Convert distance to score
            processed_results = []
            for result in results:
                distance = result.get("distance", 0)
                # Convert distance to similarity score
                if self.distance_metric == "cosine":
                    score = max(0, 1.0 - distance / 2.0)
                else:  # euclidean, l2, manhattan
                    score = 1.0 / (1.0 + distance)

                processed_results.append({
                    self.id_column: result.get(self.id_column),
                    self.content_column: result.get(self.content_column),
                    self.metadata_column: result.get(self.metadata_column),
                    "score": score,
                })

            return self._parse_output(processed_results)

        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            raise

    def delete(self, vector_id: str):
        """
        Delete a vector by ID.

        Args:
            vector_id: ID of the vector to delete
        """
        try:
            delete_query = f"DELETE FROM {self.collection_name} WHERE {self.id_column} = '{vector_id}'"
            self.engine.execute_query(delete_query)
            logger.info(f"Deleted vector {vector_id}")
        except Exception as e:
            logger.error(f"Error deleting vector {vector_id}: {e}")
            raise

    def update(self, vector_id: str, vector: Optional[List[float]] = None, payload: Optional[Dict] = None):
        """
        Update a vector and its payload.

        Args:
            vector_id: ID of the vector to update
            vector: Optional new vector
            payload: Optional new payload
        """
        try:
            update_parts = []

            if vector:
                formatted_vector = self._format_vector(vector)
                update_parts.append(f"{self.embedding_column} = {formatted_vector}")

            if payload:
                # Extract content from payload - mem0 uses "data" key as primary content field
                content = payload.get("data") or payload.get("content") or payload.get("text") or ""

                # Create metadata without the content field to avoid duplication
                metadata_payload = {k: v for k, v in payload.items() if k not in ["data", "content", "text"]}
                metadata_json = json.dumps(metadata_payload)

                update_parts.append(f"{self.content_column} = '{content}'")
                update_parts.append(f"{self.metadata_column} = '{metadata_json}'")

            if update_parts:
                update_query = f"""
                UPDATE {self.collection_name}
                SET {', '.join(update_parts)}
                WHERE {self.id_column} = '{vector_id}'
                """
                self.engine.execute_query(update_query)
                logger.info(f"Updated vector {vector_id}")

        except Exception as e:
            logger.error(f"Error updating vector {vector_id}: {e}")
            raise

    def get(self, vector_id: str) -> Optional[OutputData]:
        """
        Retrieve a vector by ID.

        Args:
            vector_id: ID of the vector to retrieve

        Returns:
            OutputData object or None if not found
        """
        try:
            select_query = f"""
            SELECT {self.id_column}, {self.content_column}, {self.metadata_column}
            FROM {self.collection_name}
            WHERE {self.id_column} = '{vector_id}'
            """

            results = self.engine.execute_query(select_query)

            if results and len(results) > 0:
                result = results[0]
                return OutputData(
                    id=str(result.get(self.id_column)),
                    score=None,
                    payload=json.loads(result.get(self.metadata_column, "{}")) if result.get(self.metadata_column) else {},
                )

            return None

        except Exception as e:
            logger.error(f"Error retrieving vector {vector_id}: {e}")
            return None

    def list_cols(self) -> List[str]:
        """
        List all collections (tables).

        Returns:
            List of collection names
        """
        try:
            # Use SHOW TABLES instead of information_schema for better reliability
            show_tables_query = "SHOW TABLES"
            results = self.engine.execute_query(show_tables_query)
            return [result["table_name"] for result in results] if results else []
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []

    def delete_col(self):
        """Delete the collection (table)."""
        try:
            drop_query = f"DROP TABLE IF EXISTS {self.collection_name}"
            self.engine.execute_query(drop_query)
            logger.info(f"Deleted collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection {self.collection_name}: {e}")
            raise

    def col_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.

        Returns:
            Dictionary with collection information
        """
        try:
            # Use SHOW TABLES with WHERE filter to check if collection exists (more efficient)
            show_tables_query = f"SHOW TABLES WHERE table_name = '{self.collection_name}'"
            results = self.engine.execute_query(show_tables_query)

            if results and len(results) > 0:
                table_info = results[0]
                return {
                    "name": self.collection_name,
                    "schema": table_info.get('schema_name', self.engine.schema),
                    "type": "TABLE",
                    "embedding_dims": self.embedding_model_dims,
                    "distance_metric": self.distance_metric,
                }

            return {}

        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}

    def list(self, filters: Optional[Dict[str, Any]] = None, limit: int = 100) -> List[OutputData]:
        """
        List all vectors in the collection.

        Args:
            filters: Optional filters to apply
            limit: Maximum number of results to return

        Returns:
            List of OutputData objects
        """
        try:
            select_query = f"""
            SELECT {self.id_column}, {self.content_column}, {self.metadata_column}
            FROM {self.collection_name}
            """

            # Add filters if provided
            filter_conditions = self._create_filter_conditions(filters) if filters else ""
            if filter_conditions:
                select_query += f" WHERE {filter_conditions}"

            select_query += f" LIMIT {limit}"

            results = self.engine.execute_query(select_query)
            return self._parse_output(results) if results else []

        except Exception as e:
            logger.error(f"Error listing vectors: {e}")
            return []

    def reset(self):
        """Reset the collection by deleting and recreating it."""
        logger.warning(f"Resetting collection {self.collection_name}...")
        try:
            self.delete_col()
            self.create_col(self.collection_name, self.embedding_model_dims, self.distance_metric)
            logger.info(f"Successfully reset collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise

    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'engine') and self.engine:
            try:
                self.engine.close()
            except Exception as e:
                logger.warning(f"Error closing ClickZetta engine: {e}")