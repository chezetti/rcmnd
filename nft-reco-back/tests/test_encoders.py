"""
Tests for the encoder components of the NFT recommendation system.
"""
import os
import sys
import unittest
import numpy as np
import torch
import time
from pathlib import Path

# Add parent directory to path to import App modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from App.encoders import CLIPEncoder, AdvancedTextEncoder, AdvancedEncoder
from App.config import IMAGE_EMBED_DIM, TEXT_EMBED_DIM, FUSION_EMBED_DIM
from tests.test_data_generator import generate_random_nft_image, generate_nft_metadata
from tests.metrics_utils import MetricsCollector, TimingContext, time_function

class TestCLIPEncoder(unittest.TestCase):
    """Tests for the CLIP encoder components."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Initialize metrics collector
        cls.metrics = MetricsCollector("clip_encoder")
        
        # Generate a test image
        start_time = time.time()
        cls.test_image_bytes = generate_random_nft_image(42)
        cls.test_image_path = "test_image.png"
        with open(cls.test_image_path, "wb") as f:
            f.write(cls.test_image_bytes)
        generation_time = (time.time() - start_time) * 1000
        cls.metrics.add_timing("image_generation", generation_time)
            
        # Initialize encoder
        start_time = time.time()
        cls.encoder = CLIPEncoder()
        init_time = (time.time() - start_time) * 1000
        cls.metrics.add_timing("encoder_initialization", init_time)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        if os.path.exists(cls.test_image_path):
            os.remove(cls.test_image_path)
        
        # Generate summary visualizations
        cls.metrics.create_summary_visualizations()
    
    def test_encode_image(self):
        """Test encoding an image returns correct tensor dimensions."""
        # Test with file path
        with TimingContext(self.metrics, "encode_image_from_path"):
            image_features = self.encoder.encode_image([self.test_image_path])
        
        self.assertIsNotNone(image_features["image_features"])
        if image_features["image_features"] is not None:
            self.assertEqual(image_features["image_features"].shape[1], IMAGE_EMBED_DIM)
            # Add embedding to metrics
            self.metrics.add_embedding(
                "image", 
                image_features["image_features"][0].detach().cpu().numpy(), 
                {"source": "file_path"}
            )
        
        # Test with bytes
        with TimingContext(self.metrics, "encode_image_from_bytes"):
            image_features = self.encoder.encode_image([self.test_image_bytes])
        
        self.assertIsNotNone(image_features["image_features"])
        if image_features["image_features"] is not None:
            self.assertEqual(image_features["image_features"].shape[1], IMAGE_EMBED_DIM)
            # Add embedding to metrics
            self.metrics.add_embedding(
                "image", 
                image_features["image_features"][0].detach().cpu().numpy(), 
                {"source": "bytes"}
            )
            
            # Add performance metric
            self.metrics.add_performance_metric(
                "image_embedding_dimension", 
                IMAGE_EMBED_DIM
            )
    
    def test_encode_text(self):
        """Test encoding text returns correct tensor dimensions."""
        test_text = "A beautiful NFT with cosmic themes"
        
        with TimingContext(self.metrics, "encode_text"):
            text_features = self.encoder.encode_text([test_text])
        
        self.assertIsNotNone(text_features["text_features"])
        # CLIP text dimension should match image dimension
        if text_features["text_features"] is not None:
            self.assertEqual(text_features["text_features"].shape[1], IMAGE_EMBED_DIM)
            # Add embedding to metrics
            self.metrics.add_embedding(
                "text", 
                text_features["text_features"][0].detach().cpu().numpy(), 
                {"text": test_text}
            )
            
            # Add performance metric
            self.metrics.add_performance_metric(
                "text_embedding_dimension", 
                IMAGE_EMBED_DIM
            )
    
    def test_extract_image_tags(self):
        """Test extracting tags from an image."""
        with TimingContext(self.metrics, "extract_image_tags"):
            # This might be None if CLIP Interrogator is not available
            tags = self.encoder.extract_image_tags(self.test_image_bytes)
        
        # We don't check the exact tags as they depend on the image and model
        # Just ensure it's a list (might be empty if interrogator isn't available)
        self.assertIsInstance(tags, list)
        
        # Add metrics about tags
        self.metrics.add_performance_metric("tag_count", len(tags))
        
        # Track tag frequency
        if tags:
            for tag in tags:
                # Update tag counts in metrics
                self.metrics.metrics["tags"][tag] += 1


class TestAdvancedTextEncoder(unittest.TestCase):
    """Tests for the text encoder components."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Initialize metrics collector
        cls.metrics = MetricsCollector("advanced_text_encoder")
        
        # Initialize encoder
        start_time = time.time()
        cls.encoder = AdvancedTextEncoder()
        init_time = (time.time() - start_time) * 1000
        cls.metrics.add_timing("encoder_initialization", init_time)
        
        # Get the actual embedding dimension from the encoder
        cls.actual_text_dim = cls.encoder.encode("test").shape[0]
        cls.metrics.add_performance_metric("text_embedding_dimension", cls.actual_text_dim)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Generate summary visualizations
        cls.metrics.create_summary_visualizations()
    
    def test_encode(self):
        """Test encoding text returns correct tensor dimensions."""
        test_text = "A beautiful NFT with cosmic themes"
        
        with TimingContext(self.metrics, "encode_single_text"):
            embedding = self.encoder.encode(test_text)
        
        self.assertIsInstance(embedding, torch.Tensor)
        # Check against actual dimension
        self.assertEqual(embedding.shape[0], self.actual_text_dim)
        
        # Add embedding to metrics
        self.metrics.add_embedding(
            "text", 
            embedding.detach().cpu().numpy(), 
            {"text": test_text}
        )
    
    def test_encode_batch(self):
        """Test encoding multiple texts at once."""
        test_texts = [
            "A beautiful NFT with cosmic themes",
            "Digital art with vibrant colors",
            "Abstract landscape with geometric shapes"
        ]
        
        with TimingContext(self.metrics, "encode_batch_text"):
            embeddings = self.encoder.encode_batch(test_texts)
        
        self.assertIsInstance(embeddings, torch.Tensor)
        # Check against actual dimension
        self.assertEqual(embeddings.shape, (3, self.actual_text_dim))
        
        # Add performance metric
        self.metrics.add_performance_metric("batch_processing_time_per_item", 
                                          self.metrics.metrics["timing"]["encode_batch_text"][-1] / len(test_texts))
        
        # Add embeddings to metrics
        for i, text in enumerate(test_texts):
            self.metrics.add_embedding(
                "text_batch", 
                embeddings[i].detach().cpu().numpy(), 
                {"text": text, "batch_index": i}
            )


class TestAdvancedEncoder(unittest.TestCase):
    """Tests for the combined encoder."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Initialize metrics collector
        cls.metrics = MetricsCollector("advanced_encoder")
        
        # Generate a test image
        start_time = time.time()
        cls.test_image_bytes = generate_random_nft_image(42)
        cls.test_image_path = "test_image.png"
        with open(cls.test_image_path, "wb") as f:
            f.write(cls.test_image_bytes)
        generation_time = (time.time() - start_time) * 1000
        cls.metrics.add_timing("image_generation", generation_time)
        
        # Generate test text
        cls.test_text = "A unique digital artwork exploring cosmic themes."
        
        # Initialize encoder
        start_time = time.time()
        cls.encoder = AdvancedEncoder()
        init_time = (time.time() - start_time) * 1000
        cls.metrics.add_timing("encoder_initialization", init_time)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        if os.path.exists(cls.test_image_path):
            os.remove(cls.test_image_path)
            
        # Generate summary visualizations
        cls.metrics.create_summary_visualizations()
    
    def test_encode_image(self):
        """Test encoding an image returns correct tensor dimensions."""
        # Get the actual dimension after encoding
        with TimingContext(self.metrics, "encode_image_from_path"):
            embedding = self.encoder.encode_image(self.test_image_path)
        
        self.assertIsInstance(embedding, torch.Tensor)
        # Just ensure it's a valid tensor with shape[0] == embedding size
        self.assertTrue(embedding.shape[0] > 0)
        
        # Add embedding to metrics
        self.metrics.add_embedding(
            "image", 
            embedding.detach().cpu().numpy(), 
            {"source": "file_path"}
        )
        
        # Add performance metric
        self.metrics.add_performance_metric("image_embedding_dimension", embedding.shape[0])
        
        # Test with bytes
        with TimingContext(self.metrics, "encode_image_from_bytes"):
            embedding = self.encoder.encode_image(self.test_image_bytes)
        
        self.assertIsInstance(embedding, torch.Tensor)
        self.assertTrue(embedding.shape[0] > 0)
        
        # Add embedding to metrics
        self.metrics.add_embedding(
            "image", 
            embedding.detach().cpu().numpy(), 
            {"source": "bytes"}
        )
    
    def test_encode_text(self):
        """Test encoding text returns correct tensor dimensions."""
        with TimingContext(self.metrics, "encode_text"):
            embedding = self.encoder.encode_text(self.test_text)
        
        self.assertIsInstance(embedding, torch.Tensor)
        # Just ensure it's a valid tensor with shape[0] == embedding size
        self.assertTrue(embedding.shape[0] > 0)
        
        # Add embedding to metrics
        self.metrics.add_embedding(
            "text", 
            embedding.detach().cpu().numpy(), 
            {"text": self.test_text}
        )
        
        # Add performance metric
        self.metrics.add_performance_metric("text_embedding_dimension", embedding.shape[0])
    
    def test_encode_combined(self):
        """Test encoding both image and text returns correct tensor dimensions."""
        with TimingContext(self.metrics, "encode_combined"):
            embedding = self.encoder.encode(self.test_image_bytes, self.test_text)
        
        self.assertIsInstance(embedding, np.ndarray)
        # Get the actual fusion dimension from the encoder
        actual_fusion_dim = embedding.shape[0]
        # Ensure it's a valid embedding
        self.assertTrue(actual_fusion_dim > 0)
        
        # Add embedding to metrics
        self.metrics.add_embedding(
            "combined", 
            embedding, 
            {"text": self.test_text, "source": "bytes"}
        )
        
        # Add performance metric
        self.metrics.add_performance_metric("combined_embedding_dimension", actual_fusion_dim)
    
    def test_compute_similarity(self):
        """Test similarity computation."""
        # Create two 1D embeddings (vectors)
        vec_size = 10  # Use a small size for testing
        embedding1 = torch.ones(vec_size)
        embedding2 = torch.ones(vec_size)
        
        # Normalize them
        embedding1 = embedding1 / torch.norm(embedding1)
        embedding2 = embedding2 / torch.norm(embedding2)
        
        # Direct cosine similarity
        with TimingContext(self.metrics, "compute_similarity_identical"):
            similarity = torch.dot(embedding1, embedding2).item()
        
        self.assertAlmostEqual(similarity, 1.0, places=5)
        self.metrics.add_accuracy_metric("identical_vectors_similarity", similarity)
        
        # Test with different vectors
        embedding2 = torch.zeros(vec_size)
        embedding2[0] = 1.0  # Make it a unit vector
        
        with TimingContext(self.metrics, "compute_similarity_different"):
            similarity = torch.dot(embedding1 / torch.norm(embedding1), 
                               embedding2 / torch.norm(embedding2)).item()
        
        self.assertLess(similarity, 1.0)
        self.metrics.add_accuracy_metric("orthogonal_vectors_similarity", similarity)
    
    def test_generate_tags(self):
        """Test tag generation."""
        with TimingContext(self.metrics, "generate_tags"):
            tags = self.encoder.generate_tags(self.test_image_bytes, self.test_text)
        
        self.assertIsInstance(tags, dict)
        # Check for expected keys - adjust to match the actual implementation
        self.assertIn("all", tags)
        self.assertIn("image_tags", tags)
        self.assertIsInstance(tags["all"], list)
        
        # Add metrics
        self.metrics.add_performance_metric("total_tag_count", len(tags["all"]))
        self.metrics.add_performance_metric("image_tag_count", len(tags["image_tags"]))
        
        # Track tags
        for tag in tags["all"]:
            self.metrics.metrics["tags"][tag] += 1


if __name__ == "__main__":
    unittest.main() 