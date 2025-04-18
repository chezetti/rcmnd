"""
Utility module for collecting metrics and creating visualizations
for the NFT recommendation system performance.
"""
import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict

# Настройка стиля для графиков
plt.style.use('ggplot')
sns.set(style="whitegrid")
sns.set_palette("viridis")

# Path to save metrics and visualizations
METRICS_DIR = Path("metrics")
METRICS_DIR.mkdir(exist_ok=True)
CHARTS_DIR = METRICS_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR = METRICS_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

class MetricsCollector:
    """Class for collecting and visualizing metrics."""
    
    def __init__(self, test_name: str):
        """
        Initialize the metrics collector.
        
        Args:
            test_name: Name of the test being run
        """
        self.test_name = test_name
        self.metrics: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "test_name": test_name,
            "performance": {},
            "accuracy": {},
            "embeddings": defaultdict(list),
            "search_results": [],
            "tags": defaultdict(int),
            "categories": defaultdict(int),
            "styles": defaultdict(int),
            "timing": {},
        }
        
        # Set up charts directory for this test
        self.test_charts_dir = CHARTS_DIR / test_name
        self.test_charts_dir.mkdir(exist_ok=True)
    
    def add_embedding(self, embedding_type: str, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """
        Add an embedding vector to the metrics.
        
        Args:
            embedding_type: Type of embedding (e.g., 'image', 'text', 'combined')
            embedding: The embedding vector
            metadata: Optional metadata about the embedding
        """
        if metadata is None:
            metadata = {}
        
        self.metrics["embeddings"][embedding_type].append({
            "vector": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
            "metadata": metadata
        })
    
    def add_search_result(self, query: Dict[str, Any], results: List[Dict[str, Any]], timing_ms: float):
        """
        Add search result metrics.
        
        Args:
            query: The query used for search
            results: The search results
            timing_ms: The time taken for the search in milliseconds
        """
        self.metrics["search_results"].append({
            "query": query,
            "results": results,
            "result_count": len(results),
            "timing_ms": timing_ms
        })
        
        # Track result tags, categories, and styles
        for result in results:
            # Tags
            if "tags" in result and result["tags"]:
                for tag in result["tags"]:
                    self.metrics["tags"][tag] += 1
            
            # Categories
            if "categories" in result and result["categories"]:
                for category in result["categories"]:
                    self.metrics["categories"][category] += 1
            
            # Styles
            if "styles" in result and result["styles"]:
                for style in result["styles"]:
                    self.metrics["styles"][style] += 1
    
    def add_timing(self, operation: str, duration_ms: float):
        """
        Add timing information.
        
        Args:
            operation: The operation being measured
            duration_ms: The duration in milliseconds
        """
        if operation not in self.metrics["timing"]:
            self.metrics["timing"][operation] = []
        
        self.metrics["timing"][operation].append(duration_ms)
    
    def add_accuracy_metric(self, metric_name: str, value: float):
        """
        Add an accuracy-related metric.
        
        Args:
            metric_name: Name of the metric
            value: Value of the metric
        """
        self.metrics["accuracy"][metric_name] = value
    
    def add_performance_metric(self, metric_name: str, value: float):
        """
        Add a performance-related metric.
        
        Args:
            metric_name: Name of the metric
            value: Value of the metric
        """
        self.metrics["performance"][metric_name] = value
    
    def save_metrics(self):
        """Save all metrics to a JSON file."""
        # Generate a filename based on test name and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.test_name}_{timestamp}.json"
        filepath = REPORTS_DIR / filename
        
        # Convert defaultdicts to regular dicts
        metrics_copy = self.metrics.copy()
        metrics_copy["tags"] = dict(metrics_copy["tags"])
        metrics_copy["categories"] = dict(metrics_copy["categories"])
        metrics_copy["styles"] = dict(metrics_copy["styles"])
        
        # Compute summary statistics for timing
        timing_summary = {}
        for operation, times in self.metrics["timing"].items():
            timing_summary[operation] = {
                "mean_ms": np.mean(times),
                "median_ms": np.median(times),
                "min_ms": np.min(times),
                "max_ms": np.max(times),
                "count": len(times)
            }
        metrics_copy["timing_summary"] = timing_summary
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics_copy, f, indent=2, ensure_ascii=False)
        
        print(f"Metrics saved to {filepath}")
        return filepath
    
    def visualize_embeddings(self, embedding_type: str = 'combined'):
        """
        Visualize embeddings using t-SNE.
        
        Args:
            embedding_type: Type of embedding to visualize
        """
        if embedding_type not in self.metrics["embeddings"] or not self.metrics["embeddings"][embedding_type]:
            print(f"No embeddings of type '{embedding_type}' to visualize")
            return
        
        # Extract embeddings and metadata
        embeddings_data = self.metrics["embeddings"][embedding_type]
        vectors = [item["vector"] for item in embeddings_data]
        
        # Check if the vectors are actually arrays or lists
        if not vectors:
            print(f"No vectors found for '{embedding_type}'")
            return
        
        # Try to convert to numpy array, handling possible errors
        try:
            vectors_array = np.array(vectors)
        except Exception as e:
            print(f"Error converting vectors to numpy array: {e}")
            return
        
        # Check if we have valid data
        if vectors_array.size == 0:
            print(f"Empty vectors for '{embedding_type}'")
            return
        
        # Apply t-SNE only if we have enough samples
        n_samples = vectors_array.shape[0]
        if n_samples < 2:
            print(f"Need at least 2 embeddings to apply t-SNE, but found {n_samples}")
            return
        
        # Simple visualization for very few samples
        if n_samples < 5:
            print(f"Too few samples ({n_samples}) for t-SNE visualization, creating simple plot")
            plt.figure(figsize=(10, 8))
            
            try:
                # If the embeddings are 1D, create a line plot
                if len(vectors_array.shape) == 1:
                    plt.plot(range(vectors_array.shape[0]), vectors_array)
                    plt.title(f'1D visualization of {embedding_type} embeddings')
                    plt.xlabel('Index')
                    plt.ylabel('Value')
                # If we have multi-dimensional embeddings
                elif vectors_array.shape[1] >= 2:
                    plt.scatter(vectors_array[:, 0], vectors_array[:, 1])
                    plt.title(f'First 2 dimensions of {embedding_type} embeddings')
                    plt.xlabel('Dimension 1')
                    plt.ylabel('Dimension 2')
                else:
                    # For embeddings with 1 feature dimension
                    indices = np.arange(n_samples)
                    values = vectors_array.flatten()
                    if len(indices) == len(values):
                        plt.bar(indices, values)
                        plt.title(f'Values of {embedding_type} embeddings')
                        plt.xlabel('Sample Index')
                        plt.ylabel('Value')
                    else:
                        print(f"Cannot create visualization: indices and values have different lengths")
                        return
            except Exception as e:
                print(f"Error creating simple visualization: {e}")
                return
                
            plt.tight_layout()
            
            # Save figure
            filename = f"{embedding_type}_embeddings_simple.png"
            filepath = self.test_charts_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Simple visualization saved to {filepath}")
            return
        
        # If we have enough samples, proceed with t-SNE
        # If we have very few samples, adjust perplexity
        # Ensure perplexity is less than n_samples
        perplexity = min(30, max(2, n_samples // 2 - 1))
        
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            embeddings_2d = tsne.fit_transform(vectors_array)
            
            # Create DataFrame for plotting
            df = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
            
            # Add metadata if available
            for i, item in enumerate(embeddings_data):
                metadata = item.get("metadata", {})
                for key, value in metadata.items():
                    if key not in df.columns:
                        df[key] = None
                    df.at[i, key] = value
            
            # Create plot
            plt.figure(figsize=(10, 8))
            
            # Select coloring variable if available
            color_var = None
            for var in ['category', 'style', 'tag']:
                if var in df.columns:
                    color_var = var
                    break
            
            if color_var:
                scatter = sns.scatterplot(data=df, x='x', y='y', hue=color_var, palette='viridis')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                scatter = sns.scatterplot(data=df, x='x', y='y')
            
            plt.title(f't-SNE visualization of {embedding_type} embeddings')
            plt.tight_layout()
            
            # Save figure
            filename = f"{embedding_type}_embeddings_tsne.png"
            filepath = self.test_charts_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Embeddings visualization saved to {filepath}")
        except Exception as e:
            print(f"Error generating t-SNE visualization: {e}")
            # Try a simple plot as fallback
            try:
                plt.figure(figsize=(10, 8))
                
                # If the embeddings are 1D, create a line plot
                if len(vectors_array.shape) == 1:
                    plt.plot(range(vectors_array.shape[0]), vectors_array)
                    plt.title(f'1D visualization of {embedding_type} embeddings')
                    plt.xlabel('Index')
                    plt.ylabel('Value')
                # If we have multi-dimensional embeddings
                elif vectors_array.shape[1] >= 2:
                    plt.scatter(vectors_array[:, 0], vectors_array[:, 1])
                    plt.title(f'First 2 dimensions of {embedding_type} embeddings')
                    plt.xlabel('Dimension 1')
                    plt.ylabel('Dimension 2')
                else:
                    # For embeddings with 1 feature dimension
                    indices = np.arange(n_samples)
                    values = vectors_array.flatten()
                    if len(indices) == len(values):
                        plt.bar(indices, values)
                        plt.title(f'Values of {embedding_type} embeddings')
                        plt.xlabel('Sample Index')
                        plt.ylabel('Value')
                    else:
                        print(f"Cannot create visualization: indices and values have different lengths")
                        return
                
                plt.tight_layout()
                
                # Save figure
                filename = f"{embedding_type}_embeddings_simple.png"
                filepath = self.test_charts_dir / filename
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Simple visualization saved to {filepath}")
            except Exception as e2:
                print(f"Could not create any visualization: {e2}")
    
    def visualize_search_performance(self):
        """Visualize search performance metrics."""
        if not self.metrics["search_results"]:
            print("No search results to visualize")
            return
        
        # Extract timing data
        timings = [result["timing_ms"] for result in self.metrics["search_results"]]
        result_counts = [result["result_count"] for result in self.metrics["search_results"]]
        
        # Setup figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Search timings
        sns.histplot(timings, kde=True, ax=ax1)
        ax1.set_title('Search Execution Time Distribution')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Frequency')
        
        # Plot 2: Result counts
        sns.histplot(result_counts, kde=True, discrete=True, ax=ax2)
        ax2.set_title('Search Result Count Distribution')
        ax2.set_xlabel('Number of Results')
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # Save figure
        filename = "search_performance.png"
        filepath = self.test_charts_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Search performance visualization saved to {filepath}")
    
    def visualize_tag_distribution(self):
        """Visualize tag distribution."""
        if not self.metrics["tags"]:
            print("No tags to visualize")
            return
        
        # Convert to DataFrame
        tags_df = pd.DataFrame({
            'tag': list(self.metrics["tags"].keys()),
            'count': list(self.metrics["tags"].values())
        })
        
        # Sort by count
        tags_df = tags_df.sort_values('count', ascending=False).head(20)  # Top 20 tags
        
        # Create plot
        plt.figure(figsize=(12, 6))
        sns.barplot(data=tags_df, x='tag', y='count')
        plt.title('Top 20 Tags in Search Results')
        plt.xlabel('Tag')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save figure
        filename = "tag_distribution.png"
        filepath = self.test_charts_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Tag distribution visualization saved to {filepath}")
    
    def visualize_category_style_distribution(self):
        """Visualize category and style distribution."""
        if not self.metrics["categories"] and not self.metrics["styles"]:
            print("No categories or styles to visualize")
            return
        
        # Setup figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot categories
        if self.metrics["categories"]:
            categories_df = pd.DataFrame({
                'category': list(self.metrics["categories"].keys()),
                'count': list(self.metrics["categories"].values())
            })
            categories_df = categories_df.sort_values('count', ascending=False)
            
            sns.barplot(data=categories_df, x='category', y='count', ax=ax1)
            ax1.set_title('Categories Distribution')
            ax1.set_xlabel('Category')
            ax1.set_ylabel('Frequency')
            ax1.tick_params(axis='x', rotation=45)
        else:
            ax1.text(0.5, 0.5, 'No category data', horizontalalignment='center', verticalalignment='center')
        
        # Plot styles
        if self.metrics["styles"]:
            styles_df = pd.DataFrame({
                'style': list(self.metrics["styles"].keys()),
                'count': list(self.metrics["styles"].values())
            })
            styles_df = styles_df.sort_values('count', ascending=False)
            
            sns.barplot(data=styles_df, x='style', y='count', ax=ax2)
            ax2.set_title('Styles Distribution')
            ax2.set_xlabel('Style')
            ax2.set_ylabel('Frequency')
            ax2.tick_params(axis='x', rotation=45)
        else:
            ax2.text(0.5, 0.5, 'No style data', horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        
        # Save figure
        filename = "category_style_distribution.png"
        filepath = self.test_charts_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Category and style distribution visualization saved to {filepath}")
    
    def visualize_timing_metrics(self):
        """Visualize timing metrics."""
        if not self.metrics["timing"]:
            print("No timing metrics to visualize")
            return
        
        # Calculate average times for each operation
        operations = []
        avg_times = []
        
        for operation, times in self.metrics["timing"].items():
            operations.append(operation)
            avg_times.append(np.mean(times))
        
        # Create DataFrame
        timing_df = pd.DataFrame({
            'operation': operations,
            'average_time_ms': avg_times
        })
        
        # Sort by average time
        timing_df = timing_df.sort_values('average_time_ms', ascending=False)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        sns.barplot(data=timing_df, x='operation', y='average_time_ms')
        plt.title('Average Time per Operation')
        plt.xlabel('Operation')
        plt.ylabel('Average Time (ms)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save figure
        filename = "timing_metrics.png"
        filepath = self.test_charts_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Timing metrics visualization saved to {filepath}")
    
    def create_summary_visualizations(self):
        """Create all visualizations and generate a summary report."""
        print(f"\n--- Generating visualizations for {self.test_name} ---")
        
        # Generate all visualizations
        if self.metrics["embeddings"]:
            for embedding_type in self.metrics["embeddings"].keys():
                self.visualize_embeddings(embedding_type)
        
        self.visualize_search_performance()
        self.visualize_tag_distribution()
        self.visualize_category_style_distribution()
        self.visualize_timing_metrics()
        
        # Save metrics
        self.save_metrics()
        
        # Print summary statistics
        if self.metrics["timing"]:
            print("\nPerformance Summary:")
            for operation, times in self.metrics["timing"].items():
                print(f"  {operation}: avg={np.mean(times):.2f}ms, min={np.min(times):.2f}ms, max={np.max(times):.2f}ms")
        
        if self.metrics["accuracy"]:
            print("\nAccuracy Metrics:")
            for metric, value in self.metrics["accuracy"].items():
                print(f"  {metric}: {value:.4f}")
        
        print(f"\nVisualizations saved to {self.test_charts_dir}")


class TimingContext:
    """Context manager for timing code execution."""
    
    def __init__(self, metrics: MetricsCollector, operation: str):
        """
        Initialize timing context.
        
        Args:
            metrics: MetricsCollector to add timing to
            operation: Name of the operation being timed
        """
        self.metrics = metrics
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and add to metrics."""
        end_time = time.time()
        if self.start_time is None:
            duration_ms = 0
        else:
            duration_ms = (end_time - self.start_time) * 1000
        self.metrics.add_timing(self.operation, duration_ms)
        return False  # Don't suppress exceptions


def compute_recommendation_metrics(ground_truth: List[str], recommendations: List[str]) -> Dict[str, float]:
    """
    Compute standard metrics for recommendation systems.
    
    Args:
        ground_truth: List of relevant item IDs
        recommendations: List of recommended item IDs
    
    Returns:
        Dictionary of metrics
    """
    # Convert to sets for intersection operations
    gt_set = set(ground_truth)
    rec_set = set(recommendations)
    
    # Calculate metrics
    precision = len(gt_set.intersection(rec_set)) / len(rec_set) if rec_set else 0
    recall = len(gt_set.intersection(rec_set)) / len(gt_set) if gt_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    
    # Calculate MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for i, item_id in enumerate(recommendations):
        if item_id in gt_set:
            mrr = 1.0 / (i + 1)
            break
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mrr": mrr
    }


def time_function(metrics: MetricsCollector, operation: str):
    """
    Decorator to time function execution.
    
    Args:
        metrics: MetricsCollector to add timing to
        operation: Name of the operation
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with TimingContext(metrics, operation):
                return func(*args, **kwargs)
        return wrapper
    return decorator 