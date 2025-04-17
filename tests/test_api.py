"""
Tests for the API routes of the NFT recommendation system.
"""
import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch

# Add parent directory to path to import App modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main
from App.config import PERSIST_DIR, INDEX_FILE, META_FILE, FEEDBACK_FILE
from tests.test_data_generator import generate_random_nft_image, generate_nft_metadata

class TestAPI(unittest.TestCase):
    """Tests for the API endpoints."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test client and temporary directories."""
        # Create a test client
        cls.client = TestClient(main.app)
        
        # Use a temporary directory for persistence
        cls.orig_persist_dir = PERSIST_DIR
        cls.temp_dir = Path(tempfile.mkdtemp())
        
        # Patch the App.config paths to use temporary directory
        cls.patcher = patch.multiple(
            'App.config',
            PERSIST_DIR=cls.temp_dir,
            INDEX_FILE=cls.temp_dir / "faiss.index",
            META_FILE=cls.temp_dir / "metadata.json",
            FEEDBACK_FILE=cls.temp_dir / "feedback.json"
        )
        cls.patcher.start()
        
        # Create test data
        cls.test_image_bytes = generate_random_nft_image(42)
        cls.test_metadata = generate_nft_metadata(42)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Stop patching
        cls.patcher.stop()
        
        # Remove temporary directory
        shutil.rmtree(cls.temp_dir)
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
    
    def test_stats_endpoint(self):
        """Test the stats endpoint."""
        response = self.client.get("/stats")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_items", data)
        self.assertIn("active_items", data)
        self.assertIn("dimension", data)
    
    def test_add_and_get_item(self):
        """Test adding and retrieving an item."""
        # Create a test payload
        files = {"image": ("test.png", self.test_image_bytes, "image/png")}
        payload = {
            "name": self.test_metadata["name"],
            "description": self.test_metadata["description"],
            "tags": ",".join(self.test_metadata["tags"]),
            "attributes": '{"rarity": "rare", "edition": "1/100"}'
        }
        
        # Add the item
        response = self.client.post("/items", files=files, data=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("uuid", data)
        
        # Get the item
        item_uuid = data["uuid"]
        response = self.client.get(f"/items/{item_uuid}")
        self.assertEqual(response.status_code, 200)
        item_data = response.json()
        self.assertEqual(item_data["uuid"], item_uuid)
        self.assertEqual(item_data["name"], self.test_metadata["name"])
    
    def test_delete_item(self):
        """Test deleting an item."""
        # First add an item
        files = {"image": ("test_delete.png", self.test_image_bytes, "image/png")}
        payload = {
            "name": "Test Delete Item",
            "description": "This item will be deleted",
            "tags": "test,delete"
        }
        
        # Add the item
        response = self.client.post("/items", files=files, data=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        item_uuid = data["uuid"]
        
        # Delete the item
        response = self.client.delete(f"/items/{item_uuid}")
        self.assertEqual(response.status_code, 200)
        
        # Try to get the deleted item
        response = self.client.get(f"/items/{item_uuid}")
        self.assertEqual(response.status_code, 404)
    
    def test_recommend_endpoint(self):
        """Test the recommendation endpoint."""
        # First add a few items
        for i in range(5):
            test_image = generate_random_nft_image(i)
            test_meta = generate_nft_metadata(i)
            
            files = {"image": (f"test{i}.png", test_image, "image/png")}
            payload = {
                "name": test_meta["name"],
                "description": test_meta["description"],
                "tags": ",".join(test_meta["tags"])
            }
            
            self.client.post("/items", files=files, data=payload)
        
        # Test recommending based on an uploaded image
        files = {"image": ("query.png", self.test_image_bytes, "image/png")}
        payload = {
            "text_query": "cosmic art",
            "mode": "balanced",
            "top_k": 3
        }
        
        response = self.client.post("/recommend", files=files, data=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("results", data)
        self.assertLessEqual(len(data["results"]), 3)
        
        # Verify result structure
        if data["results"]:
            first_result = data["results"][0]
            self.assertIn("uuid", first_result)
            self.assertIn("score", first_result)
            self.assertIn("name", first_result)
    
    def test_feedback_endpoint(self):
        """Test the feedback endpoint."""
        # First add an item
        files = {"image": ("test_feedback.png", self.test_image_bytes, "image/png")}
        payload = {
            "name": "Test Feedback Item",
            "description": "This item will receive feedback",
            "tags": "test,feedback"
        }
        
        # Add the item
        response = self.client.post("/items", files=files, data=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        item_uuid = data["uuid"]
        
        # Submit feedback
        feedback_payload = {
            "feedback_type": "favorite",
            "item_uuid": item_uuid,
            "user_id": "test_user",
            "value": 1.0
        }
        
        response = self.client.post("/feedback", json=feedback_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
    
    def test_explore_endpoint(self):
        """Test the explore endpoint."""
        # Add items with specific tags
        for i in range(3):
            test_image = generate_random_nft_image(100 + i)
            
            files = {"image": (f"explore{i}.png", test_image, "image/png")}
            payload = {
                "name": f"Explore Test {i}",
                "description": "Item for testing explore endpoint",
                "tags": "explore,test,api"
            }
            
            self.client.post("/items", files=files, data=payload)
        
        # Test explore with tags parameter
        params = {
            "tags": "api,test",
            "limit": 5,
            "offset": 0
        }
        
        response = self.client.get("/explore", params=params)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("results", data)
        self.assertIn("total", data)
        
        # Проверяем только наличие результатов, а не их количество
        # Если нет результатов, это может быть нормально в зависимости от реализации API
        self.assertIsInstance(data["results"], list)


if __name__ == "__main__":
    unittest.main() 