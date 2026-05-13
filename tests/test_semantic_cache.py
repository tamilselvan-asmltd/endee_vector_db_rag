import sys
import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.semantic_cache import RedisSemanticCache
from config.settings import settings

class TestSemanticCache:
    @patch("redis.Redis")
    def test_initialization(self, mock_redis):
        # Setup mock
        mock_ft = MagicMock()
        mock_redis.return_value.ft.return_value = mock_ft
        import redis
        mock_ft.info.side_effect = redis.exceptions.ResponseError("Unknown Index") # Simulate index creation need
        
        cache = RedisSemanticCache()
        
        assert cache.index_name == settings.semantic_cache_index_name
        mock_ft.create_index.assert_called_once()

    @patch("redis.Redis")
    def test_store_and_search_hit(self, mock_redis):
        # Setup mock
        mock_ft = MagicMock()
        mock_redis.return_value.ft.return_value = mock_ft
        
        # Mock search results for a hit
        mock_doc = MagicMock()
        mock_doc.query = "What is CNC?"
        mock_doc.response = "Computer Numerical Control"
        mock_doc.score = "0.05" # Distance 0.05 -> Similarity 0.95
        
        mock_results = MagicMock()
        mock_results.docs = [mock_doc]
        mock_ft.search.return_value = mock_results
        
        cache = RedisSemanticCache()
        embedding = [0.1] * settings.dense_dim
        
        result = cache.search(embedding)
        
        assert result is not None
        assert result["response"] == "Computer Numerical Control"
        assert result["similarity"] == 0.95

    @patch("redis.Redis")
    def test_search_miss_threshold(self, mock_redis):
        # Setup mock
        mock_ft = MagicMock()
        mock_redis.return_value.ft.return_value = mock_ft
        
        # Mock search results for a miss (low similarity)
        mock_doc = MagicMock()
        mock_doc.score = "0.3" # Distance 0.3 -> Similarity 0.7 (below 0.9 default)
        
        mock_results = MagicMock()
        mock_results.docs = [mock_doc]
        mock_ft.search.return_value = mock_results
        
        cache = RedisSemanticCache()
        embedding = [0.1] * settings.dense_dim
        
        result = cache.search(embedding)
        
        assert result is None

    @patch("redis.Redis")
    def test_invalidate_version(self, mock_redis):
        mock_ft = MagicMock()
        mock_redis.return_value.ft.return_value = mock_ft
        
        mock_doc = MagicMock()
        mock_doc.id = "cache:default:123"
        mock_results = MagicMock()
        mock_results.docs = [mock_doc]
        mock_ft.search.return_value = mock_results
        
        cache = RedisSemanticCache()
        cache.invalidate_version("1.0")
        
        mock_redis.return_value.delete.assert_called_with("cache:default:123")

if __name__ == "__main__":
    # If run directly, try to use a real Redis if available, otherwise just exit
    print("[*] Running mock tests for Semantic Cache...")
    pytest.main([__file__])
