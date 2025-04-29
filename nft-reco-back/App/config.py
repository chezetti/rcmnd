"""Global configuration and constants for the NFT Recommender service."""
from __future__ import annotations

import os
import secrets
from pathlib import Path
from enum import Enum
from typing import Dict, List, Tuple

import torch

# ---------------------------------------------------------------------------
# Runtime & model parameters
# ---------------------------------------------------------------------------

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Модели для энкодинга
VISION_MODEL = "openai/clip-vit-base-patch32"  # CLIP ViT модель (более мощная, чем ResNet)  
TEXT_MODEL = "all-mpnet-base-v2"  # Мощная текстовая модель от SentenceTransformers

# Размерности векторов для каждой модели
IMAGE_EMBED_DIM: int = 512  # CLIP ViT-B/32 embedding dimension
TEXT_EMBED_DIM: int = 768   # MPNet embedding dimension  
FUSION_EMBED_DIM: int = IMAGE_EMBED_DIM + TEXT_EMBED_DIM

# Веса для гибридного поиска
class SearchWeights:
    # Веса для мультимодального представления
    IMAGE_WEIGHT: float = 0.4  # Вес изображения при создании объединенного представления
    TEXT_WEIGHT: float = 1.0   # Вес текста при создании объединенного представления
    
    # Веса для гибридного поиска
    VISUAL_WEIGHT: float = 0.35  # Вес визуального сходства
    TEXTUAL_WEIGHT: float = 0.40  # Вес текстуального сходства
    TAG_WEIGHT: float = 0.25     # Вес сходства по тегам
    
    # Веса для продвинутого ранжирования
    POPULARITY_WEIGHT: float = 0.15  # Вес популярности элемента
    RECENCY_WEIGHT: float = 0.10     # Вес новизны элемента
    DIVERSITY_WEIGHT: float = 0.20   # Вес разнообразия результатов

# Порог релевантности для рекомендаций
RELEVANCE_THRESHOLD: float = 0.5  # Минимальный порог релевантности для рекомендаций

# Категории NFT для классификации
NFT_CATEGORIES = [
    "art", "collectible", "game", "metaverse", "defi", "utility", 
    "music", "sports", "virtual-land", "avatar", "photography", "generative",
    "3d", "pixel-art", "abstract", "portrait", "landscape", "animation"
]

# Стили NFT для классификации
NFT_STYLES = [
    "pixel", "3d", "abstract", "realistic", "surreal", "minimalist",
    "cartoon", "anime", "cyberpunk", "retro", "futuristic", "fantasy",
    "sci-fi", "hand-drawn", "generative", "photographic", "collage", "glitch"
]

# Параметры для обучения с обратной связью
FEEDBACK_ALPHA: float = 0.1  # Коэффициент обучения для обратной связи
MAX_FEEDBACK_HISTORY: int = 1000  # Максимальное количество записей обратной связи

# Параметры для повышения разнообразия результатов
DIVERSITY_WEIGHT: float = 0.2  # Вес разнообразия при ранжировании

# ---------------------------------------------------------------------------
# Persistence paths
# ---------------------------------------------------------------------------

PERSIST_DIR: Path = Path(os.getenv("PERSIST_DIR", "./persist"))
PERSIST_DIR.mkdir(parents=True, exist_ok=True)
INDEX_FILE: Path = PERSIST_DIR / "faiss.index"
META_FILE: Path = PERSIST_DIR / "metadata.json"
FEEDBACK_FILE: Path = PERSIST_DIR / "feedback.json"
MODEL_CACHE_DIR: Path = PERSIST_DIR / "models"
MODEL_CACHE_DIR.mkdir(exist_ok=True)

# Тип индекса FAISS - можно переключаться между разными типами для разного баланса скорости/точности
FAISS_INDEX_TYPE = "IVF100,Flat"  # Можно выбрать из: "Flat", "IVF100,Flat", "HNSW", "PQ"

# ---------------------------------------------------------------------------
# Feedback configuration
# ---------------------------------------------------------------------------

# Типы обратной связи и их веса
FEEDBACK_TYPES = {
    "click": 0.1,       # Клик по элементу
    "view": 0.2,        # Просмотр деталей
    "favorite": 0.5,    # Добавление в избранное
    "purchase": 1.0,    # Покупка
    "time_spent": 0.3,  # Время, проведенное на странице
    "ignore": -0.1      # Игнорирование рекомендации
}

# Параметры для затухания обратной связи с течением времени
FEEDBACK_DECAY_DAYS = 30  # Количество дней до уменьшения влияния обратной связи вдвое

# Конфигурация диверсификации результатов
DIVERSIFICATION_CONFIG = {
    "enabled": True,          # Включить/выключить диверсификацию
    "tag_weight": 0.7,        # Вес для диверсификации по тегам
    "style_weight": 0.5,      # Вес для диверсификации по стилям
    "category_weight": 0.3,   # Вес для диверсификации по категориям
    "max_similarity": 0.8     # Максимальное сходство между соседними результатами
}

# Режимы гибридного поиска
class SearchMode(str, Enum):
    VISUAL = "visual"          # Поиск только по визуальному сходству
    TEXTUAL = "textual"        # Поиск только по текстовому описанию
    BALANCED = "balanced"      # Сбалансированный подход (по умолчанию)
    VISUAL_FIRST = "visual_first"  # Сначала визуальный, затем текстовый ре-ранкинг
    TEXTUAL_FIRST = "textual_first"  # Сначала текстовый, затем визуальный ре-ранкинг

# Параметры продвинутого поиска
ADVANCED_SEARCH_CONFIG = {
    "default_mode": SearchMode.BALANCED,
    "rerank_factor": 3,  # Множитель для количества результатов при предварительном поиске
    "use_feedback": True  # Использовать обратную связь для улучшения результатов
}

# ---------------------------------------------------------------------------
# Authentication configuration
# ---------------------------------------------------------------------------

# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_MINUTES = 60 * 24  # 24 hours

# Password hashing
PASSWORD_HASH_ALGORITHM = "bcrypt"
PASSWORD_HASH_ROUNDS = 12

# User storage file
USERS_FILE: Path = PERSIST_DIR / "users.json"

# Rate limiting for auth endpoints
AUTH_RATE_LIMIT = {
    "login_attempts": 5,  # Max login attempts
    "window_seconds": 300,  # Time window in seconds (5 minutes)
    "lockout_minutes": 15  # Lockout time in minutes after exceeding attempts
}

# Access control settings
REQUIRE_AUTH_FOR_WRITE = False  # Require authentication for write operations (add, delete)
REQUIRE_AUTH_FOR_FEEDBACK = False  # Require authentication for feedback submission
