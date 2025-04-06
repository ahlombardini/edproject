import unittest
from unittest.mock import patch, MagicMock
import pytest
import numpy as np
import json
from sqlalchemy.orm import Session
from app.api.main import get_threads_by_category
from app.models.thread import Thread


class TestCategoryFilter(unittest.TestCase):
    """Test the category filtering endpoint."""

    def setUp(self):
        """Set up mock data for testing."""
        # Create mock embeddings
        embedding1 = json.dumps(np.random.rand(768).tolist())
        embedding2 = json.dumps(np.random.rand(768).tolist())
        embedding3 = json.dumps(np.random.rand(768).tolist())

        self.mock_threads = [
            MagicMock(
                id=1,
                ed_thread_id="123",
                title="Project Part 1 Question",
                category="Projet",
                subcategory="Étape 1",
                embedding=embedding1,
                created_at="2023-01-01",
                to_dict=lambda: {
                    "id": 1,
                    "ed_thread_id": "123",
                    "title": "Project Part 1 Question",
                    "category": "Projet",
                    "subcategory": "Étape 1",
                    "created_at": "2023-01-01"
                }
            ),
            MagicMock(
                id=2,
                ed_thread_id="456",
                title="General Question",
                category="Cours",
                subcategory=None,
                embedding=embedding2,
                created_at="2023-01-02",
                to_dict=lambda: {
                    "id": 2,
                    "ed_thread_id": "456",
                    "title": "General Question",
                    "category": "Cours",
                    "subcategory": None,
                    "created_at": "2023-01-02"
                }
            ),
            MagicMock(
                id=3,
                ed_thread_id="789",
                title="Project Part 2 Question",
                category="Projet",
                subcategory="Étape 2",
                embedding=embedding3,
                created_at="2023-01-03",
                to_dict=lambda: {
                    "id": 3,
                    "ed_thread_id": "789",
                    "title": "Project Part 2 Question",
                    "category": "Projet",
                    "subcategory": "Étape 2",
                    "created_at": "2023-01-03"
                }
            )
        ]

    @patch('app.api.main.get_db')
    def test_get_threads_by_category(self, mock_get_db):
        """Test getting threads by category."""
        # Set up the mock database session
        mock_db = MagicMock(spec=Session)
        mock_get_db.return_value = mock_db

        # Set up the query mock
        mock_query = MagicMock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = [self.mock_threads[0], self.mock_threads[2]]
        mock_query.count.return_value = 2
        mock_query.order_by.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = [self.mock_threads[0], self.mock_threads[2]]

        # Test the endpoint
        result = get_threads_by_category(
            category="Projet",
            subcategory=None,
            query=None,
            limit=10,
            skip=0,
            db=mock_db,
            api_key="test-key"
        )

        # Verify the result
        assert result["total_count"] == 2
        assert result["category"] == "Projet"
        assert len(result["threads"]) == 2
        assert result["threads"][0]["title"] == "Project Part 1 Question"
        assert result["threads"][1]["title"] == "Project Part 2 Question"

    @patch('app.api.main.get_db')
    def test_get_threads_by_category_and_subcategory(self, mock_get_db):
        """Test getting threads by category and subcategory."""
        # Set up the mock database session
        mock_db = MagicMock(spec=Session)
        mock_get_db.return_value = mock_db

        # Set up the query mock
        mock_query = MagicMock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = [self.mock_threads[0]]
        mock_query.count.return_value = 1
        mock_query.order_by.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = [self.mock_threads[0]]

        # Test the endpoint
        result = get_threads_by_category(
            category="Projet",
            subcategory="Étape 1",
            query=None,
            limit=10,
            skip=0,
            db=mock_db,
            api_key="test-key"
        )

        # Verify the result
        assert result["total_count"] == 1
        assert result["category"] == "Projet"
        assert result["subcategory"] == "Étape 1"
        assert len(result["threads"]) == 1
        assert result["threads"][0]["title"] == "Project Part 1 Question"

    @patch('app.api.main.model.encode')
    @patch('app.api.main.get_db')
    @patch('app.api.main.cosine_similarity')
    def test_get_threads_with_search_query(self, mock_similarity, mock_get_db, mock_encode):
        """Test getting threads with a search query."""
        # Set up mock for model encoding
        mock_encode.return_value = np.random.rand(768)

        # Set up mock for cosine similarity - thread 0 has higher similarity than thread 2
        mock_similarity.side_effect = [
            np.array([[0.8]]),  # First thread similarity
            np.array([[0.6]])   # Second thread similarity
        ]

        # Set up the mock database session
        mock_db = MagicMock(spec=Session)
        mock_get_db.return_value = mock_db

        # Set up the query mock
        mock_query = MagicMock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = [self.mock_threads[0], self.mock_threads[2]]

        # Test the endpoint with a search query
        result = get_threads_by_category(
            category="Projet",
            subcategory=None,
            query="how to implement cache",
            limit=10,
            skip=0,
            db=mock_db,
            api_key="test-key"
        )

        # Verify the result
        assert result["total_count"] == 2
        assert result["category"] == "Projet"
        assert result["query"] == "how to implement cache"
        assert len(result["threads"]) == 2

        # First result should be thread 0 (higher similarity)
        assert result["threads"][0]["title"] == "Project Part 1 Question"
        assert result["threads"][0]["similarity"] == 0.8

        # Second result should be thread 2 (lower similarity)
        assert result["threads"][1]["title"] == "Project Part 2 Question"
        assert result["threads"][1]["similarity"] == 0.6

    @patch('app.api.main.get_db')
    def test_get_threads_empty_result(self, mock_get_db):
        """Test getting threads with empty result."""
        # Set up the mock database session
        mock_db = MagicMock(spec=Session)
        mock_get_db.return_value = mock_db

        # Set up the query mock
        mock_query = MagicMock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = []
        mock_query.count.return_value = 0
        mock_query.order_by.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = []

        # Test the endpoint
        result = get_threads_by_category(
            category="nonexistent",
            subcategory=None,
            query=None,
            limit=10,
            skip=0,
            db=mock_db,
            api_key="test-key"
        )

        # Verify the result
        assert result["total_count"] == 0
        assert result["category"] == "nonexistent"
        assert len(result["threads"]) == 0


if __name__ == "__main__":
    unittest.main()
