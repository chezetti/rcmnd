"""Pydantic request / response DTOs."""
from __future__ import annotations

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator, EmailStr


class RecommendResponseItem(BaseModel):
    """Элемент результата рекомендации."""
    uuid: str = Field(..., description="Unique item identifier")
    score: float = Field(..., description="Similarity (inner‑product in [0,1])")
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    styles: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    popularity_boost: Optional[float] = None
    user_boost: Optional[float] = None
    diversity_boost: Optional[float] = None
    
    class Config:
        schema_extra = {
            "example": {
                "uuid": "550e8400-e29b-41d4-a716-446655440000",
                "score": 0.92,
                "name": "Cosmic Voyager #42",
                "description": "Rare digital artwork exploring cosmic themes",
                "tags": ["space", "abstract", "cosmic"],
                "styles": ["abstract", "futuristic"],
                "categories": ["art", "collectible"],
                "popularity_boost": 0.1,
                "diversity_boost": 0.05
            }
        }


class RecommendResponse(BaseModel):
    """Ответ API с рекомендациями."""
    results: List[RecommendResponseItem]


class IndexStatsResponse(BaseModel):
    """Статистика индекса векторов."""
    total_items: int = Field(..., description="Общее количество элементов в индексе")
    active_items: int = Field(..., description="Количество активных элементов")
    dimension: int = Field(..., description="Размерность векторов в индексе")
    index_type: str = Field(..., description="Тип индекса FAISS")
    feedback: Optional[Dict[str, int]] = Field(None, description="Статистика обратной связи")


class FeedbackType(str, Enum):
    """Типы обратной связи, которые можно оставить для NFT."""
    CLICK = "click"
    VIEW = "view"
    FAVORITE = "favorite"
    PURCHASE = "purchase"
    RELATION = "relation"


class FeedbackRequest(BaseModel):
    """Запрос на сохранение обратной связи по NFT."""
    feedback_type: FeedbackType = Field(..., description="Тип обратной связи")
    item_uuid: str = Field(..., description="UUID элемента")
    user_id: Optional[str] = Field(None, description="ID пользователя (для персонализации)")
    value: Optional[float] = Field(1.0, description="Значение обратной связи (от 0 до 1)")
    related_uuid: Optional[str] = Field(None, description="UUID связанного элемента (для связей)")
    
    @validator('value')
    def validate_value(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError('Значение обратной связи должно быть от 0 до 1')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "feedback_type": "click",
                "item_uuid": "550e8400-e29b-41d4-a716-446655440000",
                "user_id": "user123",
                "value": 1.0
            }
        }


class FeedbackResponse(BaseModel):
    """Ответ на запрос обратной связи."""
    status: str = Field(..., description="Статус операции")
    message: str = Field(..., description="Сообщение о результате")


class ItemResponse(BaseModel):
    """Ответ с метаданными NFT."""
    uuid: str = Field(..., description="Unique item identifier")
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    styles: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    attributes: Optional[Dict[str, Any]] = None
    creation_date: Optional[str] = None
    is_favorite: Optional[bool] = Field(None, description="Флаг, указывающий добавлен ли айтем в избранное текущего пользователя")
    
    class Config:
        schema_extra = {
            "example": {
                "uuid": "550e8400-e29b-41d4-a716-446655440000",
                "name": "Cosmic Voyager #42",
                "description": "Rare digital artwork exploring cosmic themes",
                "tags": ["space", "abstract", "cosmic"],
                "styles": ["abstract", "futuristic"],
                "categories": ["art", "collectible"],
                "creation_date": "2023-05-20",
                "attributes": {
                    "rarity": "legendary",
                    "edition": "limited",
                    "creator": "ai"
                }
            }
        }


class ExploreFilter(BaseModel):
    """Фильтры для исследования коллекции NFT."""
    categories: Optional[List[str]] = Field(None, description="Список категорий для фильтрации")
    styles: Optional[List[str]] = Field(None, description="Список стилей для фильтрации")
    tags: Optional[List[str]] = Field(None, description="Список тегов для фильтрации")
    min_score: Optional[float] = Field(None, description="Минимальная оценка релевантности")
    created_after: Optional[datetime] = Field(None, description="Созданы после даты")
    created_before: Optional[datetime] = Field(None, description="Созданы до даты")
    
    class Config:
        schema_extra = {
            "example": {
                "categories": ["art", "collectible"],
                "styles": ["abstract"],
                "tags": ["space", "cosmic"],
                "min_score": 0.7,
                "created_after": "2023-01-01T00:00:00"
            }
        }


class ExploreResponse(BaseModel):
    """Ответ API для исследования коллекции NFT."""
    results: List[ItemResponse]
    total: int = Field(..., description="Общее количество результатов")
    page: int = Field(1, description="Текущая страница")
    pages: int = Field(..., description="Общее количество страниц")
    
    class Config:
        schema_extra = {
            "example": {
                "results": [],
                "total": 120,
                "page": 1,
                "pages": 6
            }
        }


class ItemMetadata(BaseModel):
    """Метаданные для добавления NFT."""
    name: str = Field(..., description="Имя NFT")
    description: str = Field(..., description="Описание NFT")
    tags: Optional[List[str]] = Field(None, description="Список тегов")
    attributes: Optional[Dict[str, Any]] = Field(None, description="Дополнительные атрибуты")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Cosmic Voyager #42",
                "description": "Rare digital artwork exploring cosmic themes",
                "tags": ["space", "abstract", "cosmic"],
                "attributes": {
                    "rarity": "legendary",
                    "edition": "limited"
                }
            }
        }


# Authentication schemas
class UserRole(str, Enum):
    """Роли пользователей в системе."""
    USER = "user"
    CREATOR = "creator"
    ADMIN = "admin"


class UserBase(BaseModel):
    """Базовая модель данных пользователя."""
    username: str = Field(..., min_length=3, max_length=50, description="Имя пользователя")
    email: EmailStr = Field(..., description="Адрес электронной почты")
    full_name: Optional[str] = Field(None, description="Полное имя пользователя")


class UserCreate(UserBase):
    """Данные для создания нового пользователя."""
    password: str = Field(..., min_length=8, description="Пароль пользователя")
    
    class Config:
        schema_extra = {
            "example": {
                "username": "johndoe",
                "email": "john.doe@example.com",
                "full_name": "John Doe",
                "password": "securepassword123"
            }
        }


class UserLogin(BaseModel):
    """Данные для входа пользователя."""
    username: str = Field(..., description="Имя пользователя или адрес электронной почты")
    password: str = Field(..., description="Пароль пользователя")
    
    class Config:
        schema_extra = {
            "example": {
                "username": "johndoe",
                "password": "securepassword123"
            }
        }


class UserResponse(UserBase):
    """Модель ответа с данными пользователя."""
    id: str = Field(..., description="Уникальный идентификатор пользователя")
    role: UserRole = Field(default=UserRole.USER, description="Роль пользователя")
    created_at: datetime = Field(..., description="Дата создания аккаунта")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "username": "johndoe",
                "email": "john.doe@example.com",
                "full_name": "John Doe",
                "role": "user",
                "created_at": "2023-05-20T12:34:56"
            }
        }


class Token(BaseModel):
    """Модель токена доступа."""
    access_token: str = Field(..., description="Токен доступа JWT")
    token_type: str = Field(default="bearer", description="Тип токена")
    expires_in: int = Field(..., description="Срок действия токена в секундах")
    user: UserResponse = Field(..., description="Информация о пользователе")
    
    class Config:
        schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 3600,
                "user": {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "username": "johndoe",
                    "email": "john.doe@example.com",
                    "full_name": "John Doe",
                    "role": "user",
                    "created_at": "2023-05-20T12:34:56"
                }
            }
        }


class TokenData(BaseModel):
    """Данные, извлекаемые из токена."""
    sub: str = Field(..., description="Идентификатор пользователя")
    exp: datetime = Field(..., description="Время истечения срока действия токена")
    role: UserRole = Field(..., description="Роль пользователя")


class ChangePasswordRequest(BaseModel):
    """Запрос на изменение пароля."""
    current_password: str = Field(..., description="Текущий пароль")
    new_password: str = Field(..., min_length=8, description="Новый пароль")
    
    class Config:
        schema_extra = {
            "example": {
                "current_password": "oldpassword123",
                "new_password": "newpassword456"
            }
        }

