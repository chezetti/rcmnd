"""Wrapper around Faiss vector index + metadata persistence."""
from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
import torch
import faiss

from .config import FUSION_EMBED_DIM, INDEX_FILE, META_FILE, FAISS_INDEX_TYPE, FEEDBACK_FILE

# ---------------------------------------------------------------------------
# Feedback systems
# ---------------------------------------------------------------------------

class FeedbackSystem:
    """
    Система обратной связи для улучшения рекомендаций на основе действий пользователей.
    
    Поддерживает регистрацию кликов, добавление в избранное, и другие сигналы,
    которые используются для корректировки рейтингов рекомендаций в будущем.
    """
    
    def __init__(self, feedback_path: Path):
        """
        Инициализация системы обратной связи.
        
        Args:
            feedback_path: Путь к файлу для хранения данных обратной связи
        """
        self.feedback_path = feedback_path
        self.lock = threading.Lock()
        
        # Структура для хранения обратной связи
        self.feedback_data = {
            "clicks": {},  # UUID -> int (clicks count)
            "favorites": {},  # UUID -> int (favorites count)
            "purchases": {},  # UUID -> int (purchases count)
            "views": {},  # item_uuid -> count
            "favorites_users": {},  # item_uuid -> [user_id1, user_id2, ...]
            "purchases_users": {},  # item_uuid -> [user_id1, user_id2, ...]
            "user_preferences": {},  # user_id -> {uuid -> score}
            "item_similarity_boost": {}  # uuid -> {uuid -> boost}
        }
        
        # Загрузка существующих данных, если они есть
        self._load_feedback()
        
        # Убедимся, что все необходимые ключи существуют
        required_keys = [
            "clicks", "favorites", "purchases", "views",
            "favorites_users", "purchases_users",
            "user_preferences", "item_similarity_boost"
        ]
        for key in required_keys:
            if key not in self.feedback_data:
                self.feedback_data[key] = {}
    
    def _load_feedback(self):
        """Загрузка данных обратной связи из файла."""
        if os.path.exists(self.feedback_path):
            try:
                with open(self.feedback_path, "r", encoding="utf-8") as f:
                    self.feedback_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse feedback file {self.feedback_path}")
    
    def _save_feedback(self):
        """Сохранение данных обратной связи в файл."""
        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(self.feedback_path), exist_ok=True)
        
        with open(self.feedback_path, "w", encoding="utf-8") as f:
            json.dump(self.feedback_data, f, ensure_ascii=False, indent=2)
    
    def record_click(self, item_uuid: str, user_id: Optional[str] = None):
        """
        Регистрирует клик по элементу.
        
        Args:
            item_uuid: UUID элемента
            user_id: ID пользователя (опционально)
        """
        with self.lock:
            # Инициализируем словарь кликов, если он не существует
            if "clicks" not in self.feedback_data:
                self.feedback_data["clicks"] = {}
            
            # Инициализируем структуру для элемента
            if item_uuid not in self.feedback_data["clicks"]:
                # Инициализируем как словарь для хранения пользовательских кликов
                self.feedback_data["clicks"][item_uuid] = {}
            
            # Если структура неожиданного типа (не словарь), преобразуем ее
            if not isinstance(self.feedback_data["clicks"][item_uuid], dict):
                # Сохраняем предыдущее значение как общий счетчик
                old_count = self.feedback_data["clicks"][item_uuid] if isinstance(self.feedback_data["clicks"][item_uuid], int) else 0
                # Преобразуем в словарь для хранения пользовательских кликов
                self.feedback_data["clicks"][item_uuid] = {"_total": old_count}
            
            # Увеличиваем общий счетчик кликов
            if "_total" not in self.feedback_data["clicks"][item_uuid]:
                self.feedback_data["clicks"][item_uuid]["_total"] = 0
            self.feedback_data["clicks"][item_uuid]["_total"] += 1
            
            # Если указан пользователь, сохраняем его клик
            if user_id:
                # Сохраняем клик пользователя
                self.feedback_data["clicks"][item_uuid][user_id] = True
                
                # Обновляем предпочтения пользователя только если они уже существуют
                if user_id in self.feedback_data["user_preferences"]:
                    # Увеличиваем счетчик взаимодействия пользователя только для существующих записей
                    if item_uuid in self.feedback_data["user_preferences"][user_id]:
                        self.feedback_data["user_preferences"][user_id][item_uuid] += 0.1
            
            # Сохраняем данные
            self._save_feedback()
    
    def record_favorite(self, item_uuid: str, user_id: Optional[str] = None, value: float = 1.0):
        """
        Регистрирует добавление или удаление элемента из избранного.
        
        Args:
            item_uuid: UUID элемента
            user_id: ID пользователя (опционально)
            value: 1.0 для добавления в избранное, 0.0 для удаления из избранного
        """
        with self.lock:
            # Инициализируем структуры данных, если они отсутствуют
            if "favorites" not in self.feedback_data:
                self.feedback_data["favorites"] = {}
            if "favorites_users" not in self.feedback_data:
                self.feedback_data["favorites_users"] = {}
            if "user_preferences" not in self.feedback_data:
                self.feedback_data["user_preferences"] = {}
            
            # Сохраняем обратную связь от конкретного пользователя, если задан
            if user_id:
                # Инициализируем список пользователей для элемента, если его нет
                if item_uuid not in self.feedback_data["favorites_users"]:
                    self.feedback_data["favorites_users"][item_uuid] = []
                
                # Инициализируем предпочтения пользователя, если их нет
                if user_id not in self.feedback_data["user_preferences"]:
                    self.feedback_data["user_preferences"][user_id] = {}
                
                # Добавление в избранное (value > 0)
                if value > 0:
                    # Добавляем пользователя в список, если его там еще нет
                    if user_id not in self.feedback_data["favorites_users"][item_uuid]:
                        self.feedback_data["favorites_users"][item_uuid].append(user_id)
                    
                    # Увеличиваем счетчик избранного для элемента
                    if item_uuid not in self.feedback_data["favorites"]:
                        self.feedback_data["favorites"][item_uuid] = 0
                    self.feedback_data["favorites"][item_uuid] += 1
                    
                    # Обновляем предпочтения пользователя
                    self.feedback_data["user_preferences"][user_id][item_uuid] = 1.0
                    
                    # Debugs
                    print(f"DEBUG: Added item {item_uuid} to favorites for user {user_id}")
                    print(f"DEBUG: User preferences value: {self.feedback_data['user_preferences'][user_id][item_uuid]}")
                    print(f"DEBUG: favorites_users: {self.feedback_data['favorites_users'][item_uuid]}")
                    print(f"DEBUG: favorites count: {self.feedback_data['favorites'][item_uuid]}")
                
                # Удаление из избранного (value == 0)
                else:
                    # Удаляем пользователя из списка, если он там есть
                    if user_id in self.feedback_data["favorites_users"][item_uuid]:
                        self.feedback_data["favorites_users"][item_uuid].remove(user_id)
                    
                    # Уменьшаем счетчик избранного для элемента
                    if item_uuid in self.feedback_data["favorites"]:
                        self.feedback_data["favorites"][item_uuid] = max(0, self.feedback_data["favorites"][item_uuid] - 1)
                    
                    # Обновляем предпочтения пользователя
                    if item_uuid in self.feedback_data["user_preferences"][user_id]:
                        self.feedback_data["user_preferences"][user_id][item_uuid] = 0.0
                    
                    # Debugs
                    print(f"DEBUG: Removed item {item_uuid} from favorites for user {user_id}")
                    print(f"DEBUG: favorites_users after removal: {self.feedback_data['favorites_users'].get(item_uuid, [])}")
                    print(f"DEBUG: favorites count after removal: {self.feedback_data['favorites'].get(item_uuid, 0)}")
            else:
                # Если пользователь не указан, просто обновляем общий счетчик избранного
                if item_uuid not in self.feedback_data["favorites"]:
                    self.feedback_data["favorites"][item_uuid] = 0
                
                if value > 0:
                    self.feedback_data["favorites"][item_uuid] += 1
                else:
                    self.feedback_data["favorites"][item_uuid] = max(0, self.feedback_data["favorites"][item_uuid] - 1)
            
            # Сохраняем данные
            self._save_feedback()
    
    def record_purchase(self, item_uuid: str, user_id: Optional[str] = None):
        """
        Регистрирует покупку элемента.
        
        Args:
            item_uuid: UUID элемента
            user_id: ID пользователя (опционально)
        """
        with self.lock:
            if item_uuid not in self.feedback_data["purchases"]:
                self.feedback_data["purchases"][item_uuid] = 0
            
            self.feedback_data["purchases"][item_uuid] += 1
            
            # Сохраняем обратную связь от конкретного пользователя, если задан
            if user_id:
                # Инициализируем список пользователей для элемента, если его нет
                if item_uuid not in self.feedback_data["purchases_users"]:
                    self.feedback_data["purchases_users"][item_uuid] = []
                
                # Добавляем пользователя в список, если его там еще нет
                if user_id not in self.feedback_data["purchases_users"][item_uuid]:
                    self.feedback_data["purchases_users"][item_uuid].append(user_id)
                
                # Обновляем предпочтения пользователя только если они уже существуют
                if user_id in self.feedback_data["user_preferences"]:
                    # Увеличиваем счетчик взаимодействия пользователя только для существующих записей
                    if item_uuid in self.feedback_data["user_preferences"][user_id]:
                        self.feedback_data["user_preferences"][user_id][item_uuid] += 1.0
                    else:
                        # Для покупки создаем запись даже если её не было, так как это важное действие
                        self.feedback_data["user_preferences"][user_id][item_uuid] = 1.0
                else:
                    # Для покупки инициализируем предпочтения пользователя
                    self.feedback_data["user_preferences"][user_id] = {item_uuid: 1.0}
            
            # Сохраняем данные
            self._save_feedback()
    
    def record_itemitem_relation(self, source_uuid: str, target_uuid: str, relation_type: str, strength: float = 1.0):
        """
        Регистрирует связь между элементами для усиления item-item рекомендаций.
        
        Args:
            source_uuid: UUID исходного элемента
            target_uuid: UUID целевого элемента
            relation_type: Тип связи (например, "similar", "complementary")
            strength: Сила связи (от 0 до 1)
        """
        with self.lock:
            if source_uuid not in self.feedback_data["item_similarity_boost"]:
                self.feedback_data["item_similarity_boost"][source_uuid] = {}
            
            self.feedback_data["item_similarity_boost"][source_uuid][target_uuid] = {
                "type": relation_type,
                "strength": strength
            }
            
            # Сохраняем данные
            self._save_feedback()
    
    def get_item_boost(self, item_uuid: str) -> float:
        """
        Возвращает коэффициент усиления для элемента на основе его популярности.
        
        Args:
            item_uuid: UUID элемента
        
        Returns:
            float: Коэффициент усиления (от 0 до 0.5)
        """
        # Получаем данные о кликах, избранном и покупках
        clicks_data = self.feedback_data["clicks"].get(item_uuid, 0)
        favorites_data = self.feedback_data["favorites"].get(item_uuid, 0)
        purchases_data = self.feedback_data["purchases"].get(item_uuid, 0)
        
        # Обрабатываем клики - могут быть словарем или числом
        clicks = 0
        if isinstance(clicks_data, dict):
            clicks = len(clicks_data)  # Количество пользователей, кликнувших на элемент
        else:
            clicks = clicks_data  # Прямое значение счетчика
        
        # Обрабатываем избранное - может быть словарем или числом
        favorites = 0
        if isinstance(favorites_data, dict):
            favorites = len(favorites_data)  # Количество пользователей, добавивших в избранное
        else:
            favorites = favorites_data  # Прямое значение счетчика
        
        # Обрабатываем покупки - могут быть словарем или числом
        purchases = 0
        if isinstance(purchases_data, dict):
            purchases = len(purchases_data)  # Количество пользователей, купивших элемент
        else:
            purchases = purchases_data  # Прямое значение счетчика
        
        # Вычисляем общую популярность (с весами)
        popularity = clicks * 0.01 + favorites * 0.2 + purchases * 0.5
        
        # Нормализуем коэффициент усиления (до 0.5 максимум)
        return min(popularity / 10.0, 0.5)
    
    def get_item_relations(self, item_uuid: str) -> Dict[str, float]:
        """
        Возвращает связанные элементы и их веса для данного элемента.
        
        Args:
            item_uuid: UUID элемента
        
        Returns:
            Dict[str, float]: Маппинг UUID -> вес
        """
        relations = {}
        
        if item_uuid in self.feedback_data["item_similarity_boost"]:
            for related_uuid, data in self.feedback_data["item_similarity_boost"][item_uuid].items():
                relations[related_uuid] = data["strength"]
        
        return relations
    
    def get_user_preferences(self, user_id: str) -> Dict[str, float]:
        """
        Возвращает предпочтения пользователя (веса элементов).
        
        Args:
            user_id: ID пользователя
        
        Returns:
            Dict[str, float]: Маппинг UUID -> вес
        """
        return self.feedback_data["user_preferences"].get(user_id, {})

    def get_user_feedback(self, user_id: str) -> Dict[str, Dict[str, float]]:
        """
        Получает данные обратной связи для конкретного пользователя.
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Dict: Словарь с данными обратной связи пользователя
        """
        with self.lock:
            result = {}
            
            # Проверяем наличие пользователя в данных
            if user_id not in self.feedback_data["user_preferences"]:
                return {}
            
            # Создаем категории обратной связи для пользователя
            result["clicks"] = {}
            result["favorites"] = {}
            result["purchases"] = {}
            
            # Собираем клики
            for item_uuid, item_data in self.feedback_data["user_preferences"][user_id].items():
                if item_uuid in self.feedback_data["clicks"]:
                    result["clicks"][item_uuid] = item_data
                    
            # Собираем избранное
            for item_uuid in self.feedback_data["favorites"]:
                if user_id in self.feedback_data["favorites_users"].get(item_uuid, []):
                    result["favorites"][item_uuid] = 1.0
                    
            # Собираем покупки
            for item_uuid in self.feedback_data["purchases"]:
                if user_id in self.feedback_data["purchases_users"].get(item_uuid, []):
                    result["purchases"][item_uuid] = 1.0
                    
            return result

    def record_view(self, item_uuid: str, user_id: Optional[str] = None, value: float = 0.5):
        """
        Регистрирует просмотр элемента.
        
        Args:
            item_uuid: UUID элемента
            user_id: ID пользователя (опционально)
            value: Значение обратной связи (от 0 до 1), по умолчанию 0.5
        """
        with self.lock:
            # Проверяем и инициализируем словарь просмотров, если он не существует
            if "views" not in self.feedback_data:
                self.feedback_data["views"] = {}
            
            # Инициализируем счетчик просмотров для элемента
            if item_uuid not in self.feedback_data["views"]:
                self.feedback_data["views"][item_uuid] = {}
            
            # Если указан пользователь, сохраняем данные о его просмотре
            if user_id:
                # Проверяем, существует ли структура для сохранения данных о пользователе
                if isinstance(self.feedback_data["views"][item_uuid], dict):
                    # Увеличиваем счетчик просмотров пользователя для этого элемента
                    self.feedback_data["views"][item_uuid][user_id] = value
                else:
                    # Если структура не соответствует ожидаемой, создаем новую
                    self.feedback_data["views"][item_uuid] = {user_id: value}
                
                # Обновляем предпочтения пользователя только если элемент уже есть в предпочтениях
                # или значение просмотра достаточно высокое
                if user_id not in self.feedback_data["user_preferences"]:
                    self.feedback_data["user_preferences"][user_id] = {}
                
                # Обновляем значение только если элемент уже существует в предпочтениях
                # или значение просмотра достаточно высокое (больше 0.4)
                if item_uuid in self.feedback_data["user_preferences"][user_id] or value > 0.4:
                    current_pref = self.feedback_data["user_preferences"][user_id].get(item_uuid, 0)
                    # Увеличиваем значение только если оно уже существует
                    if current_pref > 0:
                        self.feedback_data["user_preferences"][user_id][item_uuid] = min(current_pref + 0.05, 1.0)
            
            # Сохраняем данные
            self._save_feedback()


# ---------------------------------------------------------------------------
# Vector index
# ---------------------------------------------------------------------------

class VectorIndex:
    """
    Класс, оборачивающий Faiss векторный индекс с персистентностью метаданных.
    
    Позволяет добавлять векторы с метаданными, искать похожие и извлекать метаданные
    по UUID. Также поддерживает удаление элементов и обновление метаданных.
    """

    def __init__(self, dim: int, index_path: Path, meta_path: Path, feedback_path: Path = FEEDBACK_FILE):
        """
        Инициализация индекса векторов.
        
        Args:
            dim: Размерность векторов
            index_path: Путь к файлу индекса Faiss
            meta_path: Путь к файлу метаданных
            feedback_path: Путь к файлу обратной связи
        """
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        self.lock = threading.Lock()
        
        # Набор для отслеживания уникальности контента по MD5 хешам
        self.content_hashes: Set[str] = set()
        self.uuid_to_idx: Dict[str, int] = {}  # Маппинг UUID -> индекс в Faiss
        self.idx_to_uuid: Dict[int, str] = {}  # Маппинг индекс в Faiss -> UUID
        self.metadata: Dict[str, Dict[str, Any]] = {}  # Метаданные по UUID
        
        # Система обратной связи
        self.feedback = FeedbackSystem(feedback_path)
        
        # Загрузка или создание индекса
        if os.path.exists(index_path):
            self.index = faiss.read_index(str(index_path))
            self._load_metadata()
        else:
            # Создаем индекс с указанным в конфигурации типом
            if FAISS_INDEX_TYPE == "Flat":
                self.index = faiss.IndexFlatIP(dim)  # Индекс на основе скалярного произведения
            elif FAISS_INDEX_TYPE == "IVF100,Flat":
                quantizer = faiss.IndexFlatIP(dim)
                self.index = faiss.IndexIVFFlat(quantizer, dim, 100, faiss.METRIC_INNER_PRODUCT)
                # Для IVF индекса требуется обучение - это будет сделано при добавлении первых векторов
            elif FAISS_INDEX_TYPE == "HNSW":
                self.index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            elif FAISS_INDEX_TYPE == "PQ":
                self.index = faiss.IndexPQ(dim, 16, 8, faiss.METRIC_INNER_PRODUCT)
            else:
                # По умолчанию используем плоский индекс
                self.index = faiss.IndexFlatIP(dim)

    def _load_metadata(self):
        """Загрузка метаданных из файла."""
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.metadata = data.get("metadata", {})
                self.uuid_to_idx = {k: int(v) for k, v in data.get("uuid_to_idx", {}).items()}
                self.idx_to_uuid = {int(k): v for k, v in data.get("idx_to_uuid", {}).items()}
                self.content_hashes = set(data.get("content_hashes", []))

    def _save_metadata(self):
        """Сохранение метаданных в файл."""
        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(self.meta_path), exist_ok=True)
        
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": self.metadata,
                "uuid_to_idx": self.uuid_to_idx,
                "idx_to_uuid": self.idx_to_uuid,
                "content_hashes": list(self.content_hashes)
            }, f, ensure_ascii=False, indent=2)

    def _compute_content_hash(self, vector, metadata) -> str:
        """
        Вычисляет хеш контента для дедупликации.
        
        Args:
            vector: Векторное представление
            metadata: Метаданные элемента
        
        Returns:
            str: Хеш контента
        """
        # Для упрощения используем хеш от описания и имени
        content = f"{metadata.get('description', '')}{metadata.get('name', '')}"
        return str(hash(content))
        
    def _train_ivf_index_if_needed(self, vectors: np.ndarray):
        """
        Обучает IVF индекс, если это необходимо.
        
        Args:
            vectors: Векторы для обучения
        """
        print("Training IVF index...")
        self.index.train(vectors)

    def add(self, vector: np.ndarray, metadata: Dict[str, Any]) -> str:
        """
        Добавляет вектор с метаданными в индекс.
        
        Args:
            vector: Векторное представление (размерности self.dim)
            metadata: Метаданные для сохранения
        
        Returns:
            str: UUID добавленного элемента или существующего, если контент дублируется
        """
        # Проверка размерности вектора
        if isinstance(vector, torch.Tensor):
            vector = vector.cpu().numpy()
            
        if vector.shape[-1] != self.dim:
            raise ValueError(f"Vector dimension mismatch: expected {self.dim}, got {vector.shape[-1]}")
        
        # Убедимся, что вектор имеет правильную размерность [1, dim]
        if len(vector.shape) == 1:
            vector = vector.reshape(1, -1)
        
        # Вычисляем хеш контента для дедупликации
        content_hash = self._compute_content_hash(vector, metadata)
        
        # Если такой контент уже существует, возвращаем его UUID
        for item_uuid, meta in self.metadata.items():
            if meta.get("content_hash") == content_hash and not meta.get("deleted", False):
                return item_uuid
        
        # Генерируем UUID для нового элемента
        item_uuid = str(uuid.uuid4())
        
        # Нормализуем вектор для использования косинусного сходства
        norm_vector = vector / np.linalg.norm(vector, axis=1, keepdims=True)
        norm_vector = norm_vector.astype(np.float32)
        
        # Обучаем IVF индекс, если нужно
        if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
            self._train_ivf_index_if_needed(norm_vector)
        
        # Добавляем вектор в индекс
        idx = self.index.ntotal
        self.index.add(norm_vector)
        
        # Сохраняем метаданные и маппинги
        metadata["content_hash"] = content_hash
        self.metadata[item_uuid] = metadata
        self.uuid_to_idx[item_uuid] = idx
        self.idx_to_uuid[idx] = item_uuid
        self.content_hashes.add(content_hash)
        
        # Сохраняем индекс и метаданные
        faiss.write_index(self.index, str(self.index_path))
        self._save_metadata()
        
        return item_uuid

    def delete(self, item_uuid: str) -> bool:
        """
        Удаляет элемент из индекса по UUID.
        
        Args:
            item_uuid: UUID элемента для удаления
        
        Returns:
            bool: True, если элемент был найден и удален, иначе False
        """
        if item_uuid not in self.metadata:
            return False
        
        # Удаляем хеш контента из множества
        content_hash = self.metadata[item_uuid].get("content_hash")
        if content_hash:
            self.content_hashes.discard(content_hash)
        
        # Получаем индекс в Faiss
        idx = self.uuid_to_idx.get(item_uuid)
        if idx is not None:
            # Для упрощения реализации - просто помечаем как удаленный в метаданных
            # Физическое удаление из Faiss требует перестроения индекса, что может быть затратно
            self.metadata[item_uuid]["deleted"] = True
            
            # Обновляем маппинги
            del self.uuid_to_idx[item_uuid]
            del self.idx_to_uuid[idx]
            
            # Сохраняем метаданные
            self._save_metadata()
            return True
            
        return False

    def search(self, query_vector: np.ndarray, top_k: int = 5, 
              user_id: Optional[str] = None, boost_popular: bool = True, 
              diversify: bool = True) -> List[Dict[str, Any]]:
        """
        Поиск ближайших векторов в индексе с учетом обратной связи.
        
        Args:
            query_vector: Запрос (размерности self.dim)
            top_k: Количество результатов
            user_id: ID пользователя для персонализации
            boost_popular: Усиливать ли популярные элементы
            diversify: Увеличивать ли разнообразие результатов
        
        Returns:
            List[Dict]: Список результатов поиска с метаданными и оценкой схожести
        """
        if isinstance(query_vector, torch.Tensor):
            query_vector = query_vector.cpu().numpy()
            
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        if query_vector.shape[1] != self.dim:
            raise ValueError(f"Query vector dimension mismatch: expected {self.dim}, got {query_vector.shape[1]}")
        
        # Нормализуем вектор запроса
        query_vector = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)
        query_vector = query_vector.astype(np.float32)
        
        # Для IVF индекса требуется выполнить nprobe
        if isinstance(self.index, faiss.IndexIVFFlat):
            self.index.nprobe = 10  # Устанавливаем количество ячеек для поиска
        
        # Выполняем поиск в индексе (берем больше для последующей фильтрации)
        k = min(top_k * 3, self.index.ntotal)
        distances, indices = self.index.search(query_vector, k)
        
        # Преобразуем расстояния в оценки сходства (от 0 до 1)
        # Для Inner Product, наши векторы нормализованы, так что это косинусное сходство
        scores = distances  # Уже представляет собой косинусное сходство
        
        # Формируем список результатов
        results = []
        seen_uuids = set()  # Для отслеживания дубликатов
        
        # Получаем предпочтения пользователя, если указан ID
        user_preferences = {}
        if user_id:
            user_preferences = self.feedback.get_user_preferences(user_id)
        
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # Пропускаем недействительные результаты
                continue
                
            uuid_str = self.idx_to_uuid.get(int(idx))
            if uuid_str and uuid_str in self.metadata and uuid_str not in seen_uuids:
                meta = self.metadata[uuid_str].copy()
                
                # Пропускаем удаленные элементы
                if meta.get("deleted", False):
                    continue
                    
                # Добавляем базовую информацию
                meta["uuid"] = uuid_str
                meta["score"] = float(score)  # Исходная оценка сходства
                
                # Применяем буст на основе популярности, если включено
                if boost_popular:
                    popularity_boost = self.feedback.get_item_boost(uuid_str)
                    meta["score"] += popularity_boost
                    meta["popularity_boost"] = popularity_boost
                
                # Применяем предпочтения пользователя, если есть
                if uuid_str in user_preferences:
                    user_boost = user_preferences[uuid_str] * 0.2  # Максимум +0.4 (при предпочтении 2.0)
                    meta["score"] += user_boost
                    meta["user_boost"] = user_boost
                
                results.append(meta)
                seen_uuids.add(uuid_str)
                
                if len(results) >= top_k * 2:  # Берем с запасом для диверсификации
                    break
        
        # Сортируем по скору
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Применяем диверсификацию, если включено
        if diversify and len(results) > top_k:
            diversified_results = [results[0]]  # Первый результат всегда берем
            
            # Находим разнообразные элементы
            for _ in range(top_k - 1):
                # Вычисляем "расстояние" до уже выбранных элементов на основе тегов
                max_div_idx = -1
                max_div_score = -1
                
                for i, item in enumerate(results):
                    if item in diversified_results:
                        continue
                    
                    # Рассчитываем разнообразие на основе тегов (если есть)
                    diversity_score = 0
                    item_tags = set(item.get("tags", []))
                    
                    if item_tags:
                        overlap_sum = 0
                        for selected in diversified_results:
                            selected_tags = set(selected.get("tags", []))
                            if selected_tags:
                                # Вычисляем метрику Жаккара (меньше - более разнообразно)
                                overlap = len(item_tags.intersection(selected_tags)) / len(item_tags.union(selected_tags))
                                overlap_sum += overlap
                        
                        avg_overlap = overlap_sum / len(diversified_results) if diversified_results else 0
                        diversity_score = (1 - avg_overlap) * 0.3  # Бонус за разнообразие (до +0.3)
                    
                    # Комбинированная оценка: релевантность + разнообразие
                    combined_score = item["score"] + diversity_score
                    if combined_score > max_div_score:
                        max_div_score = combined_score
                        max_div_idx = i
                
                if max_div_idx >= 0:
                    results[max_div_idx]["diversity_boost"] = max_div_score - results[max_div_idx]["score"]
                    results[max_div_idx]["score"] = max_div_score
                    diversified_results.append(results[max_div_idx])
            
            results = diversified_results
        
        # Финальная обрезка до top_k и сортировка
        results = sorted(results[:top_k], key=lambda x: x["score"], reverse=True)
        
        return results

    def get(self, item_uuid: str) -> Optional[Dict[str, Any]]:
        """
        Получает метаданные по UUID.
        
        Args:
            item_uuid: UUID элемента
        
        Returns:
            Dict или None: Метаданные элемента или None, если не найден
        """
        if item_uuid in self.metadata:
            meta = self.metadata[item_uuid].copy()
            # Не возвращаем удаленные элементы
            if meta.get("deleted", False):
                return None
            return meta
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получает статистику индекса.
        
        Returns:
            Dict: Статистика индекса (размер, размерность и т.д.)
        """
        # Считаем количество неудаленных элементов
        active_count = sum(1 for meta in self.metadata.values() if not meta.get("deleted", False))
        
        # Собираем статистику обратной связи
        # Считаем общее количество кликов с учетом новой структуры
        total_clicks = 0
        for item_uuid, click_data in self.feedback.feedback_data["clicks"].items():
            if isinstance(click_data, dict):
                # Новая структура: используем _total или считаем количество пользователей
                if "_total" in click_data:
                    total_clicks += click_data["_total"]
                else:
                    # Подсчитываем уникальных пользователей, исключая служебные ключи
                    total_clicks += sum(1 for k in click_data.keys() if not k.startswith("_"))
            elif isinstance(click_data, int):
                # Старая структура: просто число
                total_clicks += click_data
        
        # Считаем общее количество просмотров
        total_views = 0
        for item_uuid, view_data in self.feedback.feedback_data["views"].items():
            if isinstance(view_data, dict):
                total_views += len(view_data)  # Количество уникальных пользователей
        
        feedback_stats = {
            "total_clicks": total_clicks,
            "total_views": total_views,
            "total_favorites": sum(self.feedback.feedback_data["favorites"].values()),
            "total_purchases": sum(self.feedback.feedback_data["purchases"].values()),
            "users_with_preferences": len(self.feedback.feedback_data["user_preferences"])
        }
        
        # Подсчитываем общее количество взаимодействий (клики + избранное + покупки)
        total_interactions = feedback_stats["total_clicks"] + feedback_stats["total_favorites"] + feedback_stats["total_purchases"] + feedback_stats["total_views"]
        feedback_stats["total_interactions"] = total_interactions
        
        return {
            "total_items": self.index.ntotal,
            "active_items": active_count,
            "dimension": self.dim,
            "index_type": FAISS_INDEX_TYPE,
            "feedback": feedback_stats
        }

    def record_feedback(self, item_uuid: str, feedback_type: str, user_id: Optional[str] = None, value: float = 1.0):
        """
        Записывает обратную связь для элемента.
        
        Args:
            item_uuid: UUID элемента
            feedback_type: Тип обратной связи ("click", "favorite", "purchase", "view")
            user_id: ID пользователя (опционально)
            value: Значение обратной связи (для числовых оценок)
        """
        if item_uuid not in self.metadata or self.metadata[item_uuid].get("deleted", False):
            return
        
        if feedback_type == "click":
            self.feedback.record_click(item_uuid, user_id)
        elif feedback_type == "favorite":
            self.feedback.record_favorite(item_uuid, user_id, value)
        elif feedback_type == "purchase":
            self.feedback.record_purchase(item_uuid, user_id)
        elif feedback_type == "view":
            self.feedback.record_view(item_uuid, user_id, value)
        elif feedback_type == "relation":
            # В этом случае value должен быть UUID связанного элемента
            if isinstance(value, str) and value in self.metadata:
                self.feedback.record_itemitem_relation(item_uuid, value, "similar", 0.7)


# Singleton instance
vector_store = VectorIndex(FUSION_EMBED_DIM, INDEX_FILE, META_FILE, FEEDBACK_FILE)
