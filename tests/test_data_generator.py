"""
Helper module for generating synthetic NFT data for testing the recommendation system.
Provides functions to generate images and metadata for testing purposes.
"""
import os
import json
import random
import uuid
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import io
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

# Constants for NFT generation
IMAGE_SIZE = (512, 512)
COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
    (0, 128, 0),    # Green
    (128, 128, 0),  # Olive
]

# Lists for NFT metadata
NFT_NAMES_PREFIX = [
    "Cosmic", "Digital", "Cyber", "Pixel", "Virtual", "Meta", "Hyper", 
    "Neo", "Quantum", "Glitch", "Cryptic", "Ethernal", "Future", "Retro"
]

NFT_NAMES_SUFFIX = [
    "Voyager", "Explorer", "Artifact", "Creature", "Entity", "Guardian", 
    "Spirit", "Titan", "Dream", "Dimension", "Portal", "Landscape", "Genesis"
]

STYLES = [
    "pixel", "3d", "abstract", "realistic", "surreal", "minimalist",
    "cartoon", "anime", "cyberpunk", "retro", "futuristic", "fantasy",
    "sci-fi", "hand-drawn", "generative", "photographic", "collage", "glitch"
]

CATEGORIES = [
    "art", "collectible", "game", "metaverse", "defi", "utility", 
    "music", "sports", "virtual-land", "avatar", "photography", "generative",
    "3d", "pixel-art", "abstract", "portrait", "landscape", "animation"
]

TAG_POOL = [
    "space", "cosmic", "digital", "rare", "unique", "futuristic", "tech", 
    "vibrant", "geometric", "abstract", "neon", "cyber", "ethereal", "virtual",
    "dystopian", "utopian", "alien", "humanoid", "machine", "robot", "organic",
    "synthetic", "surreal", "dynamic", "static", "evolving", "primitive", "advanced",
    "glowing", "dark", "light", "colorful", "monochrome", "textured", "smooth"
]

def generate_random_nft_image(seed: int, output_path: Optional[str] = None) -> bytes:
    """
    Generate a random NFT image with unique characteristics based on the seed.
    
    Args:
        seed: Random seed for reproducibility
        output_path: Optional path to save the image
        
    Returns:
        Image data as bytes
    """
    # Set the random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Create a new image with a random background color
    img = Image.new('RGB', IMAGE_SIZE, color=random.choice(COLORS))
    draw = ImageDraw.Draw(img)
    
    # Choose a pattern type for this NFT based on the seed
    pattern_type = seed % 5
    
    if pattern_type == 0:
        # Geometric shapes
        for _ in range(20):
            shape_type = random.randint(0, 2)
            color = random.choice(COLORS)
            # Ensure x1 < x2 and y1 < y2
            x1 = random.randint(0, IMAGE_SIZE[0] - 50)
            y1 = random.randint(0, IMAGE_SIZE[1] - 50)
            x2 = random.randint(x1 + 10, min(x1 + 200, IMAGE_SIZE[0]))
            y2 = random.randint(y1 + 10, min(y1 + 200, IMAGE_SIZE[1]))
            
            if shape_type == 0:  # Rectangle
                draw.rectangle([x1, y1, x2, y2], fill=color)
            elif shape_type == 1:  # Ellipse
                draw.ellipse([x1, y1, x2, y2], fill=color)
            else:  # Line
                draw.line([x1, y1, x2, y2], fill=color, width=random.randint(1, 10))
    
    elif pattern_type == 1:
        # Gradient-like pattern
        for x in range(0, IMAGE_SIZE[0], 4):
            for y in range(0, IMAGE_SIZE[1], 4):
                brightness = (x + y) % 255
                color = (brightness, (brightness + 80) % 255, (brightness + 160) % 255)
                draw.point((x, y), fill=color)
    
    elif pattern_type == 2:
        # Concentric circles
        center_x, center_y = IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2
        max_radius = min(center_x, center_y)
        for radius in range(max_radius, 0, -20):
            color = random.choice(COLORS)
            draw.ellipse((center_x - radius, center_y - radius, 
                          center_x + radius, center_y + radius), 
                         outline=color, width=5)
    
    elif pattern_type == 3:
        # Random pixels
        for _ in range(10000):
            x = random.randint(0, IMAGE_SIZE[0] - 1)
            y = random.randint(0, IMAGE_SIZE[1] - 1)
            color = random.choice(COLORS)
            draw.point((x, y), fill=color)
    
    else:
        # Grid pattern
        cell_size = random.randint(10, 50)
        for x in range(0, IMAGE_SIZE[0], cell_size):
            for y in range(0, IMAGE_SIZE[1], cell_size):
                color = random.choice(COLORS)
                draw.rectangle([x, y, x + cell_size - 1, y + cell_size - 1], 
                              fill=color, outline=(0, 0, 0))
    
    # Add a unique number based on the seed
    # try:
    #     font = ImageFont.truetype("arial.ttf", 40)
    # except IOError:
    #     font = ImageFont.load_default()
    
    # NFT number
    text = f"#{seed:04d}"
    # Get textbox size
    # textbbox = draw.textbbox((0, 0), text, font=font)
    # textsize = (textbbox[2] - textbbox[0], textbbox[3] - textbbox[1])
    
    # # Draw text in the bottom-right corner
    # draw.text((IMAGE_SIZE[0] - textsize[0] - 10, IMAGE_SIZE[1] - textsize[1] - 10), 
    #          text, font=font, fill=(255, 255, 255))
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_data = img_byte_arr.getvalue()
    
    # Save to file if output_path is provided
    if output_path:
        img.save(output_path)
    
    return img_data

def generate_nft_metadata(seed: int, img_data: Optional[bytes] = None) -> Dict[str, Any]:
    """
    Generate metadata for an NFT based on the seed.
    
    Args:
        seed: Random seed for reproducibility
        img_data: Optional image data for auto-tagging
        
    Returns:
        Dictionary with NFT metadata
    """
    # Set the random seed for reproducibility
    random.seed(seed)
    
    # Generate a random name
    name = f"{random.choice(NFT_NAMES_PREFIX)} {random.choice(NFT_NAMES_SUFFIX)} #{seed:04d}"
    
    # Generate a random description
    descriptions = [
        f"A unique {random.choice(STYLES)} artwork exploring {random.choice(TAG_POOL)} themes.",
        f"This {random.choice(STYLES)} piece represents the essence of {random.choice(TAG_POOL)}.",
        f"Dive into a {random.choice(TAG_POOL)} world with this {random.choice(STYLES)} masterpiece.",
        f"A {random.choice(STYLES)} interpretation of {random.choice(TAG_POOL)} concepts.",
        f"This {random.choice(CATEGORIES)} NFT showcases {random.choice(STYLES)} elements with a {random.choice(TAG_POOL)} twist."
    ]
    description = random.choice(descriptions)
    
    # Generate random tags (4-7 tags)
    num_tags = random.randint(4, 7)
    tags = random.sample(TAG_POOL, num_tags)
    
    # Select random styles (1-3 styles)
    num_styles = random.randint(1, 3)
    selected_styles = random.sample(STYLES, num_styles)
    
    # Select random categories (1-2 categories)
    num_categories = random.randint(1, 2)
    selected_categories = random.sample(CATEGORIES, num_categories)
    
    # Generate attributes
    rarity_levels = ["common", "uncommon", "rare", "epic", "legendary"]
    attributes = {
        "rarity": random.choice(rarity_levels),
        "edition": f"Edition {random.randint(1, 100)} of {random.randint(100, 1000)}",
        "creation_date": datetime.now().strftime("%Y-%m-%d")
    }
    
    # Optional: Add more dynamic attributes based on the seed
    if seed % 10 == 0:
        attributes["special"] = "Genesis"
    elif seed % 5 == 0:
        attributes["special"] = "Limited"
    
    return {
        "uuid": str(uuid.uuid4()),
        "name": name,
        "description": description,
        "tags": tags,
        "styles": selected_styles,
        "categories": selected_categories,
        "attributes": attributes
    }

def create_test_nft(seed: int, output_dir: str = "test_data") -> Tuple[str, Dict[str, Any]]:
    """
    Create a complete test NFT with image and metadata.
    
    Args:
        seed: Seed for random generation
        output_dir: Directory to save the NFT data
        
    Returns:
        Tuple of (image path, metadata)
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create image filename
    img_filename = f"nft_{seed:04d}.png"
    img_path = os.path.join(output_dir, img_filename)
    
    # Generate the image
    img_data = generate_random_nft_image(seed, img_path)
    
    # Generate metadata
    metadata = generate_nft_metadata(seed, img_data)
    
    # Save metadata to JSON file
    meta_filename = f"nft_{seed:04d}.json"
    meta_path = os.path.join(output_dir, meta_filename)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return img_path, metadata

def generate_test_dataset(count: int = 100, output_dir: str = "test_data") -> List[Tuple[str, Dict[str, Any]]]:
    """
    Generate a dataset of test NFTs.
    
    Args:
        count: Number of NFTs to generate
        output_dir: Directory to save the NFT data
        
    Returns:
        List of tuples containing (image path, metadata)
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for i in range(count):
        img_path, metadata = create_test_nft(i, output_dir)
        results.append((img_path, metadata))
        print(f"Generated NFT {i+1}/{count}: {metadata['name']}")
    
    # Create a catalog file with all NFTs
    catalog = {
        "count": count,
        "generated_at": datetime.now().isoformat(),
        "nfts": [meta for _, meta in results]
    }
    
    with open(os.path.join(output_dir, "catalog.json"), 'w', encoding='utf-8') as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    
    return results

if __name__ == "__main__":
    # Test the generator
    generate_test_dataset(10) 