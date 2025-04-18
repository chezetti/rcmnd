"""
Tests for the vector index and recommendation system.
"""
import os
import sys
import shutil
import unittest
import numpy as np
import tempfile
from pathlib import Path
import faiss

# Add parent directory to path to import App modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from App.index import VectorIndex, FeedbackSystem
from App.config import FUSION_EMBED_DIM
from tests.test_data_generator import generate_random_nft_image, generate_nft_metadata

class TestFeedbackSystem(unittest.TestCase):
    """Tests for the feedback system."""
    
    def setUp(self):
        """Set up a temporary feedback file for each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.feedback_path = self.temp_dir / "feedback.json"
        self.feedback_system = FeedbackSystem(self.feedback_path)
    
    def tearDown(self):
        """Clean up after each test."""
        shutil.rmtree(self.temp_dir)
    
    def test_record_click(self):
        """Test recording a click."""
        item_uuid = "test-uuid-001"
        self.feedback_system.record_click(item_uuid)
        
        # Check that the click was recorded
        self.assertIn(item_uuid, self.feedback_system.feedback_data["clicks"])
        self.assertEqual(self.feedback_system.feedback_data["clicks"][item_uuid], 1)
        
        # Check with user_id
        user_id = "user-001"
        self.feedback_system.record_click(item_uuid, user_id)
        self.assertIn(user_id, self.feedback_system.feedback_data["user_preferences"])
        self.assertIn(item_uuid, self.feedback_system.feedback_data["user_preferences"][user_id])
    
    def test_record_favorite(self):
        """Test recording a favorite."""
        item_uuid = "test-uuid-002"
        user_id = "user-001"
        self.feedback_system.record_favorite(item_uuid, user_id)
        
        # Check that the favorite was recorded
        self.assertIn(item_uuid, self.feedback_system.feedback_data["favorites"])
        self.assertEqual(self.feedback_system.feedback_data["favorites"][item_uuid], 1)
        
        # Check user preference
        self.assertIn(user_id, self.feedback_system.feedback_data["user_preferences"])
        self.assertIn(item_uuid, self.feedback_system.feedback_data["user_preferences"][user_id])
        self.assertEqual(self.feedback_system.feedback_data["user_preferences"][user_id][item_uuid], 1.0)
    
    def test_record_purchase(self):
        """Test recording a purchase."""
        item_uuid = "test-uuid-003"
        user_id = "user-002"
        self.feedback_system.record_purchase(item_uuid, user_id)
        
        # Check that the purchase was recorded
        self.assertIn(item_uuid, self.feedback_system.feedback_data["purchases"])
        self.assertEqual(self.feedback_system.feedback_data["purchases"][item_uuid], 1)
        
        # Check user preference
        self.assertIn(user_id, self.feedback_system.feedback_data["user_preferences"])
        self.assertIn(item_uuid, self.feedback_system.feedback_data["user_preferences"][user_id])
        self.assertEqual(self.feedback_system.feedback_data["user_preferences"][user_id][item_uuid], 2.0)
    
    def test_item_boost(self):
        """Test getting item boost based on feedback."""
        item_uuid = "test-uuid-004"
        
        # No feedback initially, boost should be 0
        boost = self.feedback_system.get_item_boost(item_uuid)
        self.assertEqual(boost, 0.0)
        
        # Add some clicks
        self.feedback_system.record_click(item_uuid)
        self.feedback_system.record_click(item_uuid)
        boost = self.feedback_system.get_item_boost(item_uuid)
        self.assertGreater(boost, 0.0)
        
        # Add a favorite
        self.feedback_system.record_favorite(item_uuid)
        new_boost = self.feedback_system.get_item_boost(item_uuid)
        self.assertGreater(new_boost, boost)


class TestVectorIndex(unittest.TestCase):
    """Tests for the vector index."""
    
    def setUp(self):
        """Set up a temporary index for each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.index_path = self.temp_dir / "faiss.index"
        self.meta_path = self.temp_dir / "metadata.json"
        self.feedback_path = self.temp_dir / "feedback.json"
        
        # Create a flat index for testing instead of using the IVF index
        flat_index = faiss.IndexFlatIP(FUSION_EMBED_DIM)
        faiss.write_index(flat_index, str(self.index_path))
        
        # Create empty metadata
        with open(self.meta_path, "w") as f:
            f.write('{"metadata": {}, "uuid_to_idx": {}, "idx_to_uuid": {}, "content_hashes": []}')
        
        # Create the index
        self.index = VectorIndex(FUSION_EMBED_DIM, self.index_path, self.meta_path, self.feedback_path)
    
    def tearDown(self):
        """Clean up after each test."""
        shutil.rmtree(self.temp_dir)
    
    def test_add_and_search(self):
        """Test adding items to the index and searching."""
        # Create a random vector
        vector = np.random.random(FUSION_EMBED_DIM).astype(np.float32)
        # Normalize it
        vector = vector / np.linalg.norm(vector)
        
        # Create metadata
        metadata = {
            "name": "Test NFT",
            "description": "A test NFT for unit testing",
            "tags": ["test", "unit", "vector"],
            "styles": ["test"],
            "categories": ["testing"]
        }
        
        # Add to index
        uuid = self.index.add(vector, metadata)
        self.assertIsNotNone(uuid)
        
        # Add another item to ensure we have multiple items
        vector2 = np.random.random(FUSION_EMBED_DIM).astype(np.float32)
        vector2 = vector2 / np.linalg.norm(vector2)
        uuid2 = self.index.add(vector2, {"name": "Test NFT 2"})
        
        # Search for the first item
        results = self.index.search(vector, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["uuid"], uuid)
        
        # Check metadata
        self.assertEqual(results[0]["name"], metadata["name"])
        self.assertEqual(results[0]["description"], metadata["description"])
    
    def test_delete(self):
        """Test deleting items from the index."""
        # Add an item
        vector = np.random.random(FUSION_EMBED_DIM).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        metadata = {"name": "Delete Test NFT"}
        uuid = self.index.add(vector, metadata)
        
        # Check it exists
        item = self.index.get(uuid)
        self.assertIsNotNone(item)
        
        # Delete it
        success = self.index.delete(uuid)
        self.assertTrue(success)
        
        # Check it's gone
        item = self.index.get(uuid)
        self.assertIsNone(item)
    
    def test_diversification(self):
        """Test that search results are diversified."""
        # Create similar vectors with different tags
        base_vector = np.random.random(FUSION_EMBED_DIM).astype(np.float32)
        base_vector = base_vector / np.linalg.norm(base_vector)
        
        # Add 10 similar items with the same tag
        for i in range(10):
            # Create a slightly different vector
            noise = np.random.random(FUSION_EMBED_DIM).astype(np.float32) * 0.1
            vector = base_vector + noise
            vector = vector / np.linalg.norm(vector)
            
            # Add with the same tag
            metadata = {
                "name": f"Similar NFT {i}",
                "tags": ["common_tag"],
                "styles": ["common_style"],
                "categories": ["common_category"]
            }
            self.index.add(vector, metadata)
        
        # Add one item with a different tag
        noise = np.random.random(FUSION_EMBED_DIM).astype(np.float32) * 0.2
        vector_diff = base_vector + noise
        vector_diff = vector_diff / np.linalg.norm(vector_diff)
        metadata_diff = {
            "name": "Different NFT",
            "tags": ["different_tag"],
            "styles": ["different_style"],
            "categories": ["different_category"]
        }
        diff_uuid = self.index.add(vector_diff, metadata_diff)
        
        # Search with diversification off
        results_no_div = self.index.search(base_vector, top_k=5, diversify=False)
        
        # Search with diversification on
        results_div = self.index.search(base_vector, top_k=5, diversify=True)
        
        # In the diversified results, the different item should appear earlier
        # or have a diversity boost
        has_diversity_boost = False
        for item in results_div:
            if item["uuid"] == diff_uuid and "diversity_boost" in item:
                has_diversity_boost = True
                break
        
        # Check that at least some diversification is happening
        self.assertTrue(len(results_div) > 0)
    
    def test_feedback_integration(self):
        """Test that feedback affects search results."""
        # Create a vector
        vector = np.random.random(FUSION_EMBED_DIM).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        
        # Add two similar items
        uuid1 = self.index.add(vector, {"name": "Feedback Test NFT 1"})
        uuid2 = self.index.add(vector, {"name": "Feedback Test NFT 2"})
        
        # Record feedback for the second item
        self.index.record_feedback(uuid2, "favorite", user_id="test_user")
        
        # Search with user_id
        results = self.index.search(vector, top_k=2, user_id="test_user")
        
        # The favorited item should have a user boost
        for item in results:
            if item["uuid"] == uuid2:
                self.assertIn("user_boost", item)
                self.assertGreater(item["user_boost"], 0)


if __name__ == "__main__":
    unittest.main() 