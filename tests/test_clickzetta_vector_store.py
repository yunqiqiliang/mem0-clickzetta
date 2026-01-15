"""
Unit tests for ClickZetta vector store implementation.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from mem0.vector_stores.clickzetta import (
    ClickZetta,
    ClickZettaEngine,
    ClickZettaConnectionError,
    ClickZettaQueryError,
    ClickZettaConfigurationError,
    OutputData
)


class TestClickZettaEngine:
    """Test cases for ClickZettaEngine class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine_config = {
            "service": "test_service",
            "instance": "test_instance",
            "workspace": "test_workspace",
            "schema": "test_schema",
            "username": "test_user",
            "password": "test_pass",
            "vcluster": "test_vcluster"
        }

    @patch('mem0.vector_stores.clickzetta.clickzetta.connect')
    def test_create_session_success(self, mock_connect):
        """Test successful session creation."""
        mock_session = Mock()
        mock_connect.return_value = mock_session

        engine = ClickZettaEngine(**self.engine_config)

        assert engine.session == mock_session
        mock_connect.assert_called_once_with(
            service="test_service",
            instance="test_instance",
            workspace="test_workspace",
            schema="test_schema",
            username="test_user",
            password="test_pass",
            vcluster="test_vcluster",
            connection_timeout=30,
            query_timeout=300
        )

    @patch('mem0.vector_stores.clickzetta.clickzetta.connect')
    def test_create_session_connection_error(self, mock_connect):
        """Test session creation with connection error."""
        mock_connect.side_effect = ConnectionError("Connection failed")

        with pytest.raises(ClickZettaConnectionError, match="Connection failed"):
            ClickZettaEngine(**self.engine_config)

    @patch('mem0.vector_stores.clickzetta.clickzetta.connect')
    def test_execute_query_success(self, mock_connect):
        """Test successful query execution."""
        mock_session = Mock()
        mock_cursor = Mock()
        mock_session.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [("id1", "content1"), ("id2", "content2")]
        mock_cursor.description = [("id",), ("content",)]
        mock_connect.return_value = mock_session

        engine = ClickZettaEngine(**self.engine_config)
        results = engine.execute_query("SELECT * FROM test")

        expected = [
            {"id": "id1", "content": "content1"},
            {"id": "id2", "content": "content2"}
        ]
        assert results == expected

    @patch('mem0.vector_stores.clickzetta.clickzetta.connect')
    def test_execute_query_with_params(self, mock_connect):
        """Test query execution with parameters."""
        mock_session = Mock()
        mock_cursor = Mock()
        mock_session.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        mock_cursor.description = None
        mock_connect.return_value = mock_session

        engine = ClickZettaEngine(**self.engine_config)
        params = {"id": "test_id"}
        results = engine.execute_query("SELECT * FROM test WHERE id = %(id)s", params)

        mock_cursor.execute.assert_called_once_with("SELECT * FROM test WHERE id = %(id)s", params)
        assert results == []

    @patch('mem0.vector_stores.clickzetta.clickzetta.connect')
    def test_execute_query_connection_error(self, mock_connect):
        """Test query execution with connection error."""
        mock_session = Mock()
        mock_cursor = Mock()
        mock_session.cursor.return_value = mock_cursor
        mock_cursor.execute.side_effect = ConnectionError("Connection lost")
        mock_connect.return_value = mock_session

        engine = ClickZettaEngine(**self.engine_config)

        with pytest.raises(ClickZettaConnectionError, match="Connection lost during query"):
            engine.execute_query("SELECT * FROM test")

    @patch('mem0.vector_stores.clickzetta.clickzetta.connect')
    def test_close_session(self, mock_connect):
        """Test session closing."""
        mock_session = Mock()
        mock_connect.return_value = mock_session

        engine = ClickZettaEngine(**self.engine_config)
        engine.close()

        mock_session.close.assert_called_once()


class TestClickZetta:
    """Test cases for ClickZetta vector store class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "collection_name": "test_collection",
            "embedding_model_dims": 1536,
            "service": "test_service",
            "instance": "test_instance",
            "workspace": "test_workspace",
            "schema": "test_schema",
            "username": "test_user",
            "password": "test_pass",
            "vcluster": "test_vcluster"
        }

    @patch('mem0.vector_stores.clickzetta.ClickZettaEngine')
    def test_init_success(self, mock_engine_class):
        """Test successful ClickZetta initialization."""
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine

        with patch.object(ClickZetta, 'create_col'):
            vector_store = ClickZetta(**self.config)

        assert vector_store.collection_name == "test_collection"
        assert vector_store.embedding_model_dims == 1536
        assert vector_store.distance_metric == "cosine"

    def test_init_invalid_distance_metric(self):
        """Test initialization with invalid distance metric."""
        config = self.config.copy()
        config["distance_metric"] = "invalid_metric"

        with pytest.raises(ValueError, match="Unsupported distance metric"):
            ClickZetta(**config)

    @patch('mem0.vector_stores.clickzetta.ClickZettaEngine')
    def test_format_vector(self, mock_engine_class):
        """Test vector formatting."""
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine

        with patch.object(ClickZetta, 'create_col'):
            vector_store = ClickZetta(**self.config)

        vector = [1.0, 2.5, -0.5]
        formatted = vector_store._format_vector(vector)

        assert formatted == "vector(1.0,2.5,-0.5)"

    @patch('mem0.vector_stores.clickzetta.ClickZettaEngine')
    def test_parse_output(self, mock_engine_class):
        """Test output parsing."""
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine

        with patch.object(ClickZetta, 'create_col'):
            vector_store = ClickZetta(**self.config)

        results = [
            {"id": "test_1", "score": 0.95, "metadata": '{"user_id": "user1"}'},
            {"id": "test_2", "score": 0.85, "metadata": '{"user_id": "user2"}'}
        ]

        output_data = vector_store._parse_output(results)

        assert len(output_data) == 2
        assert output_data[0].id == "test_1"
        assert output_data[0].score == 0.95
        assert output_data[0].payload == {"user_id": "user1"}

    @patch('mem0.vector_stores.clickzetta.ClickZettaEngine')
    def test_create_filter_conditions(self, mock_engine_class):
        """Test filter conditions creation."""
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine

        with patch.object(ClickZetta, 'create_col'):
            vector_store = ClickZetta(**self.config)

        # Test string filter
        filters = {"user_id": "test_user"}
        conditions = vector_store._create_filter_conditions(filters)
        assert "json_extract_string" in conditions
        assert "user_id" in conditions
        assert "test_user" in conditions

        # Test numeric filter
        filters = {"score": 0.5}
        conditions = vector_store._create_filter_conditions(filters)
        assert "json_extract_float" in conditions

        # Test range filter
        filters = {"score": {"gte": 0.5, "lte": 0.9}}
        conditions = vector_store._create_filter_conditions(filters)
        assert ">= 0.5" in conditions
        assert "<= 0.9" in conditions

    @patch('mem0.vector_stores.clickzetta.ClickZettaEngine')
    def test_insert_vectors(self, mock_engine_class):
        """Test vector insertion."""
        mock_engine = Mock()
        mock_engine.execute_query = Mock()
        mock_engine_class.return_value = mock_engine

        with patch.object(ClickZetta, 'create_col'):
            vector_store = ClickZetta(**self.config)

        vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        payloads = [
            {"data": "content1", "user_id": "user1"},
            {"data": "content2", "user_id": "user2"}
        ]
        ids = ["id1", "id2"]

        vector_store.insert(vectors, payloads, ids)

        mock_engine.execute_query.assert_called_once()
        call_args = mock_engine.execute_query.call_args[0][0]
        assert "INSERT INTO test_collection" in call_args
        assert "content1" in call_args
        assert "content2" in call_args

    @patch('mem0.vector_stores.clickzetta.ClickZettaEngine')
    def test_search_vectors(self, mock_engine_class):
        """Test vector search."""
        mock_engine = Mock()
        mock_engine.execute_query.return_value = [
            {"id": "id1", "content": "content1", "metadata": '{"user_id": "user1"}', "distance": 0.1}
        ]
        mock_engine_class.return_value = mock_engine

        with patch.object(ClickZetta, 'create_col'):
            vector_store = ClickZetta(**self.config)

        query_vector = [1.0, 2.0, 3.0]
        results = vector_store.search("test query", query_vector, limit=5)

        assert len(results) == 1
        assert results[0].id == "id1"
        assert results[0].score > 0  # Should be converted from distance

    @patch('mem0.vector_stores.clickzetta.ClickZettaEngine')
    def test_get_vector(self, mock_engine_class):
        """Test vector retrieval."""
        mock_engine = Mock()
        mock_engine.execute_query.return_value = [
            {"id": "id1", "content": "content1", "metadata": '{"user_id": "user1"}'}
        ]
        mock_engine_class.return_value = mock_engine

        with patch.object(ClickZetta, 'create_col'):
            vector_store = ClickZetta(**self.config)

        result = vector_store.get("id1")

        assert result is not None
        assert result.id == "id1"
        assert result.payload == {"user_id": "user1"}

    @patch('mem0.vector_stores.clickzetta.ClickZettaEngine')
    def test_get_vector_not_found(self, mock_engine_class):
        """Test vector retrieval when not found."""
        mock_engine = Mock()
        mock_engine.execute_query.return_value = []
        mock_engine_class.return_value = mock_engine

        with patch.object(ClickZetta, 'create_col'):
            vector_store = ClickZetta(**self.config)

        result = vector_store.get("nonexistent_id")

        assert result is None

    @patch('mem0.vector_stores.clickzetta.ClickZettaEngine')
    def test_delete_vector(self, mock_engine_class):
        """Test vector deletion."""
        mock_engine = Mock()
        mock_engine.execute_query = Mock()
        mock_engine_class.return_value = mock_engine

        with patch.object(ClickZetta, 'create_col'):
            vector_store = ClickZetta(**self.config)

        vector_store.delete("id1")

        mock_engine.execute_query.assert_called_once()
        call_args = mock_engine.execute_query.call_args[0][0]
        assert "DELETE FROM test_collection" in call_args
        assert "id = 'id1'" in call_args

    @patch('mem0.vector_stores.clickzetta.ClickZettaEngine')
    def test_update_vector(self, mock_engine_class):
        """Test vector update."""
        mock_engine = Mock()
        mock_engine.execute_query = Mock()
        mock_engine_class.return_value = mock_engine

        with patch.object(ClickZetta, 'create_col'):
            vector_store = ClickZetta(**self.config)

        new_vector = [1.0, 2.0, 3.0]
        new_payload = {"data": "updated content", "user_id": "user1"}

        vector_store.update("id1", vector=new_vector, payload=new_payload)

        mock_engine.execute_query.assert_called_once()
        call_args = mock_engine.execute_query.call_args[0][0]
        assert "UPDATE test_collection" in call_args
        assert "updated content" in call_args

    @patch('mem0.vector_stores.clickzetta.ClickZettaEngine')
    def test_list_collections(self, mock_engine_class):
        """Test collection listing."""
        mock_engine = Mock()
        mock_engine.execute_query.return_value = [
            {"table_name": "collection1"},
            {"table_name": "collection2"}
        ]
        mock_engine_class.return_value = mock_engine

        with patch.object(ClickZetta, 'create_col'):
            vector_store = ClickZetta(**self.config)

        collections = vector_store.list_cols()

        assert collections == ["collection1", "collection2"]

    @patch('mem0.vector_stores.clickzetta.ClickZettaEngine')
    def test_collection_info(self, mock_engine_class):
        """Test collection info retrieval."""
        mock_engine = Mock()
        mock_engine.execute_query.return_value = [
            {"table_name": "test_collection", "schema_name": "test_schema"}
        ]
        mock_engine_class.return_value = mock_engine

        with patch.object(ClickZetta, 'create_col'):
            vector_store = ClickZetta(**self.config)

        info = vector_store.col_info()

        assert info["name"] == "test_collection"
        assert info["schema"] == "test_schema"
        assert info["embedding_dims"] == 1536

    @patch('mem0.vector_stores.clickzetta.ClickZettaEngine')
    def test_list_vectors(self, mock_engine_class):
        """Test vector listing."""
        mock_engine = Mock()
        mock_engine.execute_query.return_value = [
            {"id": "id1", "content": "content1", "metadata": '{"user_id": "user1"}'},
            {"id": "id2", "content": "content2", "metadata": '{"user_id": "user2"}'}
        ]
        mock_engine_class.return_value = mock_engine

        with patch.object(ClickZetta, 'create_col'):
            vector_store = ClickZetta(**self.config)

        vectors = vector_store.list(limit=10)

        assert len(vectors) == 2
        assert vectors[0].id == "id1"
        assert vectors[1].id == "id2"

    @patch('mem0.vector_stores.clickzetta.ClickZettaEngine')
    def test_reset_collection(self, mock_engine_class):
        """Test collection reset."""
        mock_engine = Mock()
        mock_engine.execute_query = Mock()
        # Mock the execute_query to return empty list for SHOW TABLES query
        mock_engine.execute_query.return_value = []
        mock_engine_class.return_value = mock_engine

        with patch.object(ClickZetta, 'create_col') as mock_create_col:
            vector_store = ClickZetta(**self.config)

        with patch.object(vector_store, 'delete_col') as mock_delete_col:
            vector_store.reset()

        mock_delete_col.assert_called_once()
        mock_create_col.assert_called()  # Called twice: once in __init__, once in reset


class TestOutputData:
    """Test cases for OutputData model."""

    def test_output_data_creation(self):
        """Test OutputData model creation."""
        data = OutputData(
            id="test_id",
            score=0.95,
            payload={"user_id": "user1", "content": "test content"}
        )

        assert data.id == "test_id"
        assert data.score == 0.95
        assert data.payload["user_id"] == "user1"

    def test_output_data_optional_fields(self):
        """Test OutputData with optional fields."""
        data = OutputData()

        assert data.id is None
        assert data.score is None
        assert data.payload is None


class TestErrorHandling:
    """Test cases for error handling."""

    def test_clickzetta_connection_error(self):
        """Test ClickZettaConnectionError."""
        error = ClickZettaConnectionError("Connection failed")
        assert str(error) == "Connection failed"

    def test_clickzetta_query_error(self):
        """Test ClickZettaQueryError."""
        error = ClickZettaQueryError("Query failed")
        assert str(error) == "Query failed"

    def test_clickzetta_configuration_error(self):
        """Test ClickZettaConfigurationError."""
        error = ClickZettaConfigurationError("Invalid configuration")
        assert str(error) == "Invalid configuration"


class TestIntegration:
    """Integration test cases."""

    @patch('mem0.vector_stores.clickzetta.clickzetta.connect')
    def test_end_to_end_workflow(self, mock_connect):
        """Test complete workflow from creation to cleanup."""
        # Mock the session and cursor
        mock_session = Mock()
        mock_cursor = Mock()
        mock_session.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_session

        # Mock query responses
        def mock_execute_query_side_effect(query, params=None):
            if "SHOW TABLES" in query:
                return []  # No existing tables
            elif "CREATE TABLE" in query:
                return []  # Table creation success
            elif "INSERT INTO" in query:
                return []  # Insert success
            elif "SELECT" in query and "COSINE_DISTANCE" in query:
                return [{"id": "test_1", "content": "test content", "metadata": '{"user_id": "user1"}', "distance": 0.1}]
            else:
                return []

        mock_cursor.fetchall.side_effect = lambda: []
        mock_cursor.description = None

        config = {
            "collection_name": "test_integration",
            "embedding_model_dims": 3,
            "service": "test_service",
            "instance": "test_instance",
            "workspace": "test_workspace",
            "schema": "test_schema",
            "username": "test_user",
            "password": "test_pass",
            "vcluster": "test_vcluster"
        }

        # Create vector store
        with patch.object(ClickZettaEngine, 'execute_query', side_effect=mock_execute_query_side_effect):
            vector_store = ClickZetta(**config)

            # Insert vectors
            vectors = [[1.0, 2.0, 3.0]]
            payloads = [{"data": "test content", "user_id": "user1"}]
            ids = ["test_1"]

            vector_store.insert(vectors, payloads, ids)

            # Search vectors
            results = vector_store.search("test query", [1.0, 2.0, 3.0], limit=1)

            # Verify results
            assert len(results) == 1
            assert results[0].id == "test_1"


if __name__ == "__main__":
    pytest.main([__file__])