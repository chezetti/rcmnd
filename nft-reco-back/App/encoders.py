"""Advanced vision + text encoders for multimodal embeddings."""
from __future__ import annotations

import io
import re
import json
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union, Any, cast, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer, AutoModel, 
    CLIPProcessor, CLIPModel,
    CLIPTokenizer, CLIPImageProcessor
)

from .config import (
    DEVICE, IMAGE_EMBED_DIM, TEXT_EMBED_DIM, 
    VISION_MODEL, TEXT_MODEL, MODEL_CACHE_DIR,
    SearchWeights, NFT_CATEGORIES, NFT_STYLES
)

# Conditionally import clip_interrogator
has_clip_interrogator = False
clip_interrogator_module = None
try:
    # Import the module but don't use it directly in imports
    import importlib
    clip_interrogator_module = importlib.import_module("clip_interrogator")
    has_clip_interrogator = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """
    Предобработка текста для улучшения качества эмбеддингов.
    
    Args:
        text: Исходный текст
        
    Returns:
        Предобработанный текст
    """
    # Приведение к нижнему регистру
    text = text.lower()
    
    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text)
    
    # Удаление специальных символов, кроме базовой пунктуации
    text = re.sub(r'[^\w\s.,!?&\[\]\-\'\":;]', '', text)
    
    # Нормализация кавычек
    text = text.replace('"', '"').replace('"', '"')
    
    return text.strip()

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Извлекает ключевые слова из текста для тегирования.
    
    Args:
        text: Исходный текст
        max_keywords: Максимальное количество ключевых слов
        
    Returns:
        Список ключевых слов
    """
    # Токенизация
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Удаление стоп-слов (базовый список)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                 'has', 'have', 'had', 'be', 'been', 'being', 'in', 'on', 'at', 'to',
                 'for', 'with', 'by', 'about', 'as', 'of', 'from', 'this', 'that'}
    
    filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Подсчет частотности
    word_freq = {}
    for word in filtered_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Сортировка по частоте
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Возврат top N ключевых слов
    return [word for word, _ in sorted_words[:max_keywords]]

# ---------------------------------------------------------------------------
# Advanced encoders
# ---------------------------------------------------------------------------

class CLIPEncoder:
    """CLIP-based encoder for images and text."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        
        # Use the provided model name directly if it has a slash (assumed to be a full HF model ID)
        # Otherwise, map to known model IDs
        if '/' in model_name:
            model_id = model_name
        else:
            # Legacy model name conversion (from OpenAI CLIP style to HF style)
            if model_name.lower() == "vit-b/32":
                model_id = "openai/clip-vit-base-patch32"
            elif model_name.lower() == "vit-l/14":
                model_id = "openai/clip-vit-large-patch14"
            elif model_name.lower() == "vit-l/14-336":
                model_id = "openai/clip-vit-large-patch14-336"
            elif model_name.lower().startswith("rn"):
                model_id = f"openai/clip-{model_name.lower()}"
            else:
                # Default to base model if unknown
                model_id = "openai/clip-vit-base-patch32"
                
        print(f"Using CLIP model: {model_id}")
        
        # Initialize with None values
        self.tokenizer: Optional[CLIPTokenizer] = None
        self.image_processor: Optional[CLIPImageProcessor] = None
        self.model: Optional[CLIPModel] = None
        
        try:
            # Initialize tokenizer and image processor separately
            self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
            self.image_processor = CLIPImageProcessor.from_pretrained(model_id)
            self.model = CLIPModel.from_pretrained(model_id)
            
            # Move model to device and set to eval mode
            if self.model is not None:
                # Use to() without reassignment but properly handling device
                self.model = self.model.to(DEVICE)  # type: ignore
                self.model.eval()
        except Exception as e:
            print(f"Error initializing CLIP model: {e}")
            # Don't reassign to None since we already initialized with None
            raise
        
        # Initialize clip interrogator if available
        self.interrogator = None
        if has_clip_interrogator and clip_interrogator_module:
            try:
                Config = getattr(clip_interrogator_module, "Config")
                Interrogator = getattr(clip_interrogator_module, "Interrogator")
                config = Config(clip_model_name=model_name)
                self.interrogator = Interrogator(config)
            except Exception as e:
                print(f"Warning: Could not initialize CLIP Interrogator: {e}")
    
    def encode_image(self, image_paths: List[Union[str, bytes, memoryview, bytearray]]) -> Mapping[str, Optional[torch.Tensor]]:
        """
        Encode images using CLIP model.
        Args:
            image_paths (List[Union[str, bytes, memoryview, bytearray]]): List of paths to images or image data
        Returns:
            Mapping[str, Optional[torch.Tensor]]: Dictionary with image features
        """
        # Create default return with None values
        default_return: Dict[str, Optional[torch.Tensor]] = {"image_features": None, "image_attention": None}
        
        if self.image_processor is None or self.model is None:
            return default_return
            
        try:
            # Handle both file paths and bytes/binary data
            images = []
            for img_path in image_paths:
                if isinstance(img_path, str):
                    # Handle file path
                    images.append(Image.open(img_path))
                else:
                    # Handle bytes-like object
                    images.append(Image.open(io.BytesIO(img_path)))
                    
            # Process images using the image processor
            inputs = self.image_processor(images=images, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            with torch.no_grad():
                # Get image features
                if hasattr(self.model, 'get_image_features'):
                    image_features = self.model.get_image_features(**inputs)
                else:
                    return default_return
                
                # Getting vision model outputs
                vision_outputs = None
                if hasattr(self.model, 'vision_model'):
                    vision_outputs = self.model.vision_model(**inputs)
                
                # Extract pooler_output
                pooler_output = None
                if vision_outputs is not None and hasattr(vision_outputs, "pooler_output"):
                    pooler_output = vision_outputs.pooler_output
                
            return {
                "image_features": image_features,
                "image_attention": pooler_output
            }
            
        except Exception as e:
            print(f"Error encoding image: {e}")
            return default_return
    
    def extract_image_tags(self, img_bytes: bytes, top_k: int = 5) -> List[str]:
        """Извлекает теги из изображения с помощью CLIP Interrogator."""
        if not self.interrogator:
            return []
            
        try:
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            # Get image description from CLIP Interrogator
            description = self.interrogator.interrogate(image)
            # Extract tags from description
            tags = re.findall(r'\b\w+\b', description.lower())
            stop_words = {'the', 'a', 'an', 'and', 'of', 'in', 'with', 'by'}
            filtered_tags = [tag for tag in tags if tag not in stop_words and len(tag) > 2]
            # Remove duplicates and limit to top_k
            unique_tags = list(dict.fromkeys(filtered_tags))
            return unique_tags[:top_k]
        except Exception as e:
            print(f"Error extracting image tags: {e}")
            return []
    
    def encode_text(self, texts: List[str]) -> Mapping[str, Optional[torch.Tensor]]:
        """
        Encode texts using CLIP model.
        Args:
            texts (List[str]): List of text inputs
        Returns:
            Mapping[str, Optional[torch.Tensor]]: Dictionary with text features
        """
        # Create default return with None values
        default_return: Dict[str, Optional[torch.Tensor]] = {"text_features": None, "text_attention": None}
        
        if self.tokenizer is None or self.model is None:
            return default_return
            
        try:
            # Process text using the tokenizer
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            with torch.no_grad():
                # Get text features
                if hasattr(self.model, 'get_text_features'):
                    text_features = self.model.get_text_features(**inputs)
                else:
                    return default_return
                
                # Getting text model outputs
                text_outputs = None
                if hasattr(self.model, 'text_model'):
                    text_outputs = self.model.text_model(**inputs)
                
                # Extract pooler_output
                pooler_output = None
                if text_outputs is not None and hasattr(text_outputs, "pooler_output"):
                    pooler_output = text_outputs.pooler_output
                
            return {
                "text_features": text_features,
                "text_attention": pooler_output
            }
            
        except Exception as e:
            print(f"Error encoding text: {e}")
            return default_return
    
    def compute_similarity(self, img_embedding: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        """Вычисляет сходство между изображением и текстом."""
        # Нормализация
        img_embedding = F.normalize(img_embedding, dim=-1)
        text_embedding = F.normalize(text_embedding, dim=-1)
        
        # Вычисление косинусного сходства
        return torch.sum(img_embedding * text_embedding, dim=-1)


class TextEncoder:
    """Базовый класс для текстовых энкодеров."""
    
    def encode(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Кодирует текст в embedding."""
        raise NotImplementedError("Subclasses must implement this method")


class AdvancedTextEncoder(TextEncoder):
    """Расширенный текстовый энкодер, использующий современные языковые модели."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Инициализирует энкодер с указанной моделью."""
        try:
            import sentence_transformers
            self.model = sentence_transformers.SentenceTransformer(model_name)
            self.model.to(DEVICE)
        except Exception as e:
            print(f"Error initializing AdvancedTextEncoder: {e}")
            self.model = None
    
    def encode(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Кодирует текст, используя предварительно обученную языковую модель."""
        try:
            if self.model is None:
                # Возвращаем нулевой тензор в случае ошибки
                if isinstance(text, list):
                    return torch.zeros((len(text), TEXT_EMBED_DIM), device=DEVICE)
                else:
                    return torch.zeros(TEXT_EMBED_DIM, device=DEVICE)
            
            # Преобразуем в список, если передана одна строка
            if isinstance(text, str):
                text_input = [text]
            else:
                text_input = text
                
            # Получаем эмбеддинги от модели
            with torch.no_grad():
                embeddings = self.model.encode(text_input, convert_to_tensor=True)
                # Преобразуем в тензор, если это еще не тензор
                if not isinstance(embeddings, torch.Tensor):
                    embeddings = torch.tensor(embeddings, device=DEVICE)
                elif embeddings.device != DEVICE:
                    embeddings = embeddings.to(DEVICE)
                
                # Возвращаем тензор соответствующей размерности
                if isinstance(text, str):
                    return embeddings[0] if embeddings.dim() > 1 else embeddings
                else:
                    return embeddings
                    
        except Exception as e:
            print(f"Error in TextEncoder.encode: {e}")
            if isinstance(text, list):
                return torch.zeros((len(text), TEXT_EMBED_DIM), device=DEVICE)
            else:
                return torch.zeros(TEXT_EMBED_DIM, device=DEVICE)
                
    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Кодирует батч текстов в векторные представления (для обратной совместимости)."""
        return self.encode(texts)


class NFTClassifier:
    """Классификатор NFT по категориям и стилям."""
    
    def __init__(self, text_encoder: AdvancedTextEncoder):
        self.text_encoder = text_encoder
        self._init_vectors()
        
    def _init_vectors(self):
        """Инициализирует векторные представления для категорий и стилей."""
        # Кодирование категорий
        self.category_embeddings = self.text_encoder.encode_batch(NFT_CATEGORIES)
        self.style_embeddings = self.text_encoder.encode_batch(NFT_STYLES)
        
        # Словари для маппинга индексов к названиям
        self.idx_to_category = {i: cat for i, cat in enumerate(NFT_CATEGORIES)}
        self.idx_to_style = {i: style for i, style in enumerate(NFT_STYLES)}
        
    def classify(self, text: str, top_k: int = 3) -> Dict[str, List[Tuple[str, float]]]:
        """
        Классифицирует NFT по категориям и стилям.
        
        Args:
            text: Описание NFT
            top_k: Количество топ результатов
            
        Returns:
            Словарь с топ категориями и стилями
        """
        # Получаем эмбеддинг текста
        text_embedding = self.text_encoder.encode(text)
        
        # Находим наиболее похожие категории
        category_scores = torch.matmul(self.category_embeddings, text_embedding)
        top_category_indices = torch.topk(category_scores, min(top_k, len(NFT_CATEGORIES))).indices.cpu().numpy()
        top_categories = [(self.idx_to_category[int(idx)], float(category_scores[idx])) 
                        for idx in top_category_indices]
        
        # Находим наиболее похожие стили
        style_scores = torch.matmul(self.style_embeddings, text_embedding)
        top_style_indices = torch.topk(style_scores, min(top_k, len(NFT_STYLES))).indices.cpu().numpy()
        top_styles = [(self.idx_to_style[int(idx)], float(style_scores[idx])) 
                     for idx in top_style_indices]
        
        return {
            "categories": top_categories,
            "styles": top_styles
        }


class AdvancedEncoder:
    """
    Продвинутый энкодер для NFT, объединяющий визуальную и текстовую модальности.
    Поддерживает CLIP для мультимодального кодирования и передовые текстовые модели.
    """
    
    def __init__(self):
        # Инициализация кодировщиков
        self.clip_encoder = CLIPEncoder(VISION_MODEL)
        self.text_encoder = AdvancedTextEncoder(TEXT_MODEL)
        
        # Классификатор для тегирования
        self.classifier = NFTClassifier(self.text_encoder)
    
    def encode_image(self, img_bytes: Union[bytes, str]) -> torch.Tensor:
        """Кодирует изображение, используя CLIP."""
        # Проверим, что процессор и модель инициализированы
        if getattr(self.clip_encoder, 'image_processor', None) is None or getattr(self.clip_encoder, 'model', None) is None:
            # Возвращаем пустой тензор нужной размерности
            return torch.zeros(IMAGE_EMBED_DIM, device=DEVICE)
            
        try:
            if isinstance(img_bytes, bytes):
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                if self.clip_encoder.image_processor is not None:
                    # Убираем параметр text=None, который вызывает предупреждение
                    inputs = self.clip_encoder.image_processor(images=[image], return_tensors="pt")
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                    
                    model = getattr(self.clip_encoder, 'model', None)
                    if model is not None:
                        with torch.no_grad():
                            image_features_func = getattr(model, "get_image_features", None)
                            if image_features_func is not None:
                                output = image_features_func(**inputs)
                                if output is not None:
                                    return output
            else:
                # В случае, если передан путь к изображению
                result = self.clip_encoder.encode_image([img_bytes])
                if result is not None and isinstance(result, dict):
                    features = result.get("image_features")
                    if features is not None and isinstance(features, torch.Tensor):
                        return features
            # Возвращаем пустой тензор если что-то пошло не так
            return torch.zeros(IMAGE_EMBED_DIM, device=DEVICE)
        except Exception as e:
            print(f"Error in encode_image: {e}")
            return torch.zeros(IMAGE_EMBED_DIM, device=DEVICE)
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Кодирует текст, используя AdvancedTextEncoder."""
        try:
            if self.text_encoder is None:
                # Возвращаем пустой тензор нужной размерности
                return torch.zeros(TEXT_EMBED_DIM, device=DEVICE)
            
            # Передаем строку напрямую
            text_features = self.text_encoder.encode(text)
            if text_features is not None and isinstance(text_features, torch.Tensor) and text_features.numel() > 0:
                return text_features
            else:
                return torch.zeros(TEXT_EMBED_DIM, device=DEVICE)
        except Exception as e:
            print(f"Error in encode_text: {e}")
            return torch.zeros(TEXT_EMBED_DIM, device=DEVICE)
    
    def encode(self, img_bytes: Union[bytes, str], text: str) -> np.ndarray:
        """
        Создает мультимодальное представление NFT.
        
        Объединяет визуальные и текстовые признаки с взвешиванием:
        - Изображение кодируется через CLIP
        - Текст кодируется через Sentence-Transformer
        - Признаки конкатенируются с весами
        
        Args:
            img_bytes: Байты изображения или путь к файлу
            text: Текстовое описание
            
        Returns:
            Объединенный вектор представления
        """
        # Получаем векторы представления
        img_embedding = self.encode_image(img_bytes)
        text_embedding = self.encode_text(text)
        
        # Убедимся, что у тензоров соответствующие размерности
        if len(img_embedding.shape) == 1:
            img_embedding = img_embedding.unsqueeze(0)  # добавляем размерность батча [D] -> [1, D]
        
        if len(text_embedding.shape) == 1:
            text_embedding = text_embedding.unsqueeze(0)  # добавляем размерность батча [D] -> [1, D]
            
        # Применяем веса
        weighted_img_embedding = img_embedding * SearchWeights.IMAGE_WEIGHT
        weighted_text_embedding = text_embedding * SearchWeights.TEXT_WEIGHT
        
        # Конкатенируем и преобразуем в numpy для совместимости с Faiss
        combined = torch.cat([weighted_img_embedding, weighted_text_embedding], dim=1)
        return combined.cpu().numpy().astype(np.float32)
    
    def compute_similarity(self, query_embedding: torch.Tensor, 
                         target_embedding: torch.Tensor) -> float:
        """
        Вычисляет косинусное сходство между двумя эмбеддингами.
        
        Args:
            query_embedding: Эмбеддинг запроса
            target_embedding: Эмбеддинг цели
            
        Returns:
            Косинусное сходство (от -1 до 1)
        """
        # Нормализация
        query_norm = F.normalize(query_embedding, dim=1)
        target_norm = F.normalize(target_embedding, dim=1)
        
        # Косинусное сходство
        return torch.matmul(query_norm, target_norm.T).item()
    
    def generate_tags(self, img_bytes: bytes, text: str, 
                     max_tags: int = 8) -> Dict[str, List[str]]:
        """
        Генерирует теги для NFT на основе изображения и текста.
        
        Args:
            img_bytes: Байты изображения
            text: Текстовое описание
            max_tags: Максимальное количество тегов
            
        Returns:
            Словарь с тегами разных типов
        """
        try:
            print(f"DEBUG: Extracting keywords from text...")
            # Получаем ключевые слова из текста
            keywords = extract_keywords(text, max_keywords=max_tags)
            print(f"DEBUG: Keywords extracted: {keywords}")
            
            print(f"DEBUG: Extracting image tags...")
            # Извлекаем теги из изображения, если доступно
            image_tags = []
            if hasattr(self, 'clip_encoder') and self.clip_encoder is not None:
                extracted_tags = self.clip_encoder.extract_image_tags(img_bytes, top_k=max_tags)
                if extracted_tags:
                    image_tags = extracted_tags
            print(f"DEBUG: Image tags extracted: {image_tags}")
            
            print(f"DEBUG: Classifying by categories and styles...")
            # Классифицируем по категориям и стилям
            classification = self.classifier.classify(text, top_k=3)
            
            # Формируем теги из категорий и стилей с высоким скором
            category_tags = [cat for cat, score in classification["categories"] if score > 0.3]
            style_tags = [style for style, score in classification["styles"] if score > 0.3]
            
            print(f"DEBUG: Categories: {category_tags}")
            print(f"DEBUG: Styles: {style_tags}")
            
            # Объединяем все теги
            all_tags = list(set(keywords + category_tags + style_tags + image_tags))[:max_tags]
            print(f"DEBUG: Combined tags: {all_tags}")
            
            return {
                "all": all_tags,
                "keywords": keywords,
                "categories": category_tags,
                "styles": style_tags,
                "image_tags": image_tags
            }
        except Exception as e:
            print(f"ERROR in generate_tags: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "all": [],
                "keywords": [],
                "categories": [],
                "styles": [],
                "image_tags": []
            }


# ---------------------------------------------------------------------------
# Singleton instance
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_encoder() -> AdvancedEncoder:
    """Создает и возвращает синглтон энкодера."""
    return AdvancedEncoder()

# Экземпляр для использования в других модулях
encoder = get_encoder()
