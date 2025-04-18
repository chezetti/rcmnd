"""
Utility functions for the application.

This module provides helper functions for:
- Image validation and processing
- Safe file operations
- Error handling
"""

import os
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
import io

from PIL import Image, UnidentifiedImageError


def validate_image(image_data: Optional[bytes]) -> bool:
    """
    Validate that the provided data is a valid image.
    
    Args:
        image_data: Binary image data to validate
        
    Returns:
        True if valid image, False otherwise
    """
    if image_data is None or len(image_data) == 0:
        return False
        
    try:
        # Try to open the image with PIL
        image = Image.open(io.BytesIO(image_data))
        # Verify it's a valid image by loading it
        image.verify()
        return True
    except (UnidentifiedImageError, IOError, SyntaxError, ValueError):
        # Any of these exceptions indicate an invalid image
        return False


def safe_load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Safely load JSON data from a file, returning an empty dict on failure.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary with the loaded data, or empty dict on failure
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, IOError):
        # Return empty dict on any error
        return {}


def safe_save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
    """
    Safely save JSON data to a file.
    
    Args:
        data: Dictionary to save
        file_path: Path where to save the JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Save the data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except (IOError, OSError):
        # Return False on any error
        return False 