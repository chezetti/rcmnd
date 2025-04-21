"""Recommendation system for NFT items."""
from __future__ import annotations

from typing import List, Optional, Dict, Any
import random

from .index import vector_store

def get_recommendations(user_id: str, limit: int = 20) -> List[str]:
    """
    Get personalized recommendations for a user.
    
    Args:
        user_id: User ID for personalization
        limit: Maximum number of recommendations to return
    
    Returns:
        List of item UUIDs recommended for the user
    """
    try:
        # Get user preferences
        user_preferences = vector_store.feedback.get_user_preferences(user_id)
        
        # If user has no preferences, return popular items
        if not user_preferences:
            # Get popular items from feedback data
            clicks = vector_store.feedback.feedback_data.get("clicks", {})
            favorites = vector_store.feedback.feedback_data.get("favorites", {})
            
            # Combine and sort by popularity
            popularity = {}
            for item_uuid, count in clicks.items():
                if not vector_store.metadata.get(item_uuid, {}).get("deleted", False):
                    popularity[item_uuid] = count
            
            for item_uuid, count in favorites.items():
                if not vector_store.metadata.get(item_uuid, {}).get("deleted", False):
                    popularity[item_uuid] = popularity.get(item_uuid, 0) + count * 3  # Favorites have higher weight
            
            # Sort by popularity and get top items
            sorted_items = sorted(popularity.items(), key=lambda x: x[1], reverse=True)
            return [item[0] for item in sorted_items[:limit]]
        
        # If user has preferences, use vector similarity to recommend items
        # Get items the user has interacted with
        interacted_items = set(user_preferences.keys())
        
        # Get all available items
        all_items = [uuid for uuid, meta in vector_store.metadata.items() 
                    if not meta.get("deleted", False) and uuid not in interacted_items]
        
        if not all_items:
            return []
        
        # For now, return a random selection of items the user hasn't interacted with
        # In a real system, we would use vector similarity and collaborative filtering
        random.shuffle(all_items)
        return all_items[:limit]
    
    except Exception as e:
        print(f"Error in get_recommendations: {str(e)}")
        import traceback
        traceback.print_exc()
        return [] 