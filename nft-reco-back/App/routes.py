"""HTTP endpoints grouped in a FastAPI router."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Query, Path, Body, Depends, Request, status
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import torch
import numpy as np
from datetime import datetime, timedelta

from .encoders import encoder
from .index import vector_store
from .schemas import (
    RecommendResponse, IndexStatsResponse, FeedbackRequest, 
    FeedbackResponse, ExploreFilter, ExploreResponse, ItemResponse,
    UserCreate, UserLogin, UserResponse, Token, ChangePasswordRequest
)
from .config import (
    SearchMode, ADVANCED_SEARCH_CONFIG, FUSION_EMBED_DIM, 
    IMAGE_EMBED_DIM, TEXT_EMBED_DIM, JWT_EXPIRATION_MINUTES,
    REQUIRE_AUTH_FOR_WRITE, REQUIRE_AUTH_FOR_FEEDBACK
)
from .auth import (
    user_manager, create_access_token, get_current_user, 
    get_current_user_from_bearer, check_rate_limit, get_current_user_optional
)

router = APIRouter()


# Authentication routes
@router.post("/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED, summary="Register a new user")
async def register(user_data: UserCreate):
    """
    Регистрирует нового пользователя в системе.
    
    - **username**: Уникальное имя пользователя
    - **email**: Адрес электронной почты
    - **full_name**: Полное имя пользователя (опционально)
    - **password**: Пароль (минимум 8 символов)
    
    Возвращает информацию о созданном пользователе.
    """
    try:
        user = user_manager.create_user(user_data)
        return user
    except HTTPException as exc:
        raise exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(exc)}"
        )


@router.post("/auth/login", response_model=Token, summary="Login and get access token")
async def login(request: Request, login_data: UserLogin):
    """
    Аутентифицирует пользователя и выдает JWT токен.
    
    - **username**: Имя пользователя или адрес электронной почты
    - **password**: Пароль пользователя
    
    Возвращает токен доступа JWT и информацию о пользователе.
    """
    # Get client IP for rate limiting
    client_ip = request.client.host if request.client else "unknown"
    
    # Check rate limiting
    if not check_rate_limit(login_data.username, client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Please try again later."
        )
    
    # Authenticate user
    user = user_manager.authenticate_user(login_data.username, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Generate JWT token
    access_token = create_access_token(user)
    
    # Return token and user info
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=JWT_EXPIRATION_MINUTES * 60,  # Convert to seconds
        user=user
    )


@router.get("/auth/me", response_model=UserResponse, summary="Get current user info")
async def get_me(current_user: UserResponse = Depends(get_current_user)):
    """
    Возвращает информацию о текущем аутентифицированном пользователе.
    
    Требуется действительный JWT токен в заголовке Authorization.
    """
    return current_user


@router.post("/auth/change-password", response_model=dict, summary="Change user password")
async def change_password(
    request: ChangePasswordRequest,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Изменяет пароль текущего пользователя.
    
    - **current_password**: Текущий пароль
    - **new_password**: Новый пароль (минимум 8 символов)
    
    Требуется действительный JWT токен в заголовке Authorization.
    """
    success = user_manager.change_password(
        current_user.id, 
        request.current_password, 
        request.new_password
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    return {"status": "success", "message": "Password changed successfully"}


@router.post("/items", summary="Index a new NFT item")
async def add_item(
    image: UploadFile = File(..., description="NFT image file (jpeg/png)"),
    description: str = Form(..., description="Textual description / attributes"),
    name: str = Form("", description="Human‑readable name"),
    tags: Optional[str] = Form(None, description="Comma-separated tags (optional)"),
    current_user: Optional[UserResponse] = Depends(get_current_user) if REQUIRE_AUTH_FOR_WRITE else None,
):
    """
    Индексирует новый NFT элемент для последующего поиска.
    
    - **image**: Изображение NFT (jpeg/png)
    - **description**: Текстовое описание NFT
    - **name**: Человекочитаемое имя (опционально)
    - **tags**: Теги через запятую (опционально)
    
    Возвращает уникальный идентификатор NFT в системе.
    Выполняется дедупликация - если NFT с таким же именем и описанием
    уже существует, возвращается его ID.
    """
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Unsupported image type")

    img_bytes = await image.read()
    try:
        print(f"DEBUG: Processing image: {image.filename}, size: {len(img_bytes)}")
        print(f"DEBUG: Description: {description[:50]}...")
        
        # Получаем эмбеддинг из энкодера
        print(f"DEBUG: Calling encoder.encode...")
        embedding = encoder.encode(img_bytes, description)
        print(f"DEBUG: Got embedding with shape: {embedding.shape}")
        
        # Генерируем теги из контента, если не предоставлены
        user_tags = []
        if tags:
            user_tags = [tag.strip().lower() for tag in tags.split(",") if tag.strip()]
        
        print(f"DEBUG: Generating tags...")
        # Получаем автоматические теги из изображения и текста
        generated_tags = encoder.generate_tags(img_bytes, description)
        
        # Объединяем пользовательские и автоматические теги
        all_tags = list(set(user_tags + generated_tags["all"]))
        
        # Подготавливаем метаданные
        metadata = {
            "name": name, 
            "description": description,
            "tags": all_tags,
            "categories": generated_tags["categories"],
            "styles": generated_tags["styles"],
            "created_by": current_user.id if current_user else None,
            "created_at": datetime.now().isoformat()
        }
        
        print(f"DEBUG: Adding to vector store...")
        # Добавляем в индекс
        item_uuid = vector_store.add(embedding, metadata)
        print(f"DEBUG: Added with UUID: {item_uuid}")
        
        return {"uuid": item_uuid, "tags": all_tags}
        
    except Exception as exc:
        print(f"ERROR in add_item: {type(exc).__name__}: {str(exc)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/recommend", response_model=RecommendResponse, summary="Get similar NFTs")
async def recommend(
    image: UploadFile = File(default=None, media_type="image/*", description="Query image"),
    description: Optional[str] = Form(default=None, description="Query textual description"),
    top_k: int = Form(default=5, ge=1, le=20, description="Number of results"),
    user_id: Optional[str] = Form(default=None, description="User ID for personalization"),
    search_mode: Optional[str] = Form(default=None, description="Search mode (visual, textual, balanced)"),
    diversify: bool = Form(default=True, description="Diversify results"),
    current_user: Optional[UserResponse] = Depends(get_current_user_optional),
):
    """
    Поиск похожих NFT на основе изображения и текстового описания.
    
    - **image**: Изображение для поиска похожих NFT (опционально)
    - **description**: Текстовое описание для поиска (опционально)
    - **top_k**: Количество результатов (от 1 до 20)
    - **user_id**: ID пользователя для персонализации результатов (опционально)
    - **search_mode**: Режим поиска (visual, textual, balanced)
    - **diversify**: Диверсифицировать результаты
    
    Результаты сортируются по убыванию схожести и содержат
    оценку релевантности (score) и метаданные NFT.
    """
    if image is None and not description:
        raise HTTPException(
            status_code=400, 
            detail="Either image or description must be provided"
        )
    
    if image and image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(
            status_code=415, 
            detail=f"Unsupported image type: {image.content_type}. Supported types: image/jpeg, image/png, image/webp"
        )
    
    # Используем ID пользователя из токена, если доступен и не указан явно
    if current_user and not user_id:
        user_id = current_user.id
    
    # Определяем режим поиска
    current_mode = search_mode or ADVANCED_SEARCH_CONFIG["default_mode"]
    if search_mode and search_mode not in {mode.value for mode in SearchMode}:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid search mode. Available modes: {', '.join(mode.value for mode in SearchMode)}"
        )
    
    try:
        # Получаем изображение, если предоставлено
        img_bytes = None
        if image:
            img_bytes = await image.read()
        
        # Получаем эмбеддинг в зависимости от режима
        if current_mode == SearchMode.VISUAL.value and img_bytes:
            # Только визуальный поиск
            img_embedding = encoder.encode_image(img_bytes)

            # Убедимся, что эмбеддинг имеет правильную размерность
            if len(img_embedding.shape) == 1:
                img_embedding = img_embedding.unsqueeze(0)
                
            # Если размерность не соответствует ожидаемой, дополняем нулями
            if img_embedding.shape[1] != FUSION_EMBED_DIM:
                if img_embedding.shape[1] == IMAGE_EMBED_DIM:
                    # Дополняем нулевым текстовым эмбеддингом
                    text_embedding = torch.zeros((1, TEXT_EMBED_DIM), device=img_embedding.device)
                    combined = torch.cat([img_embedding, text_embedding], dim=1)
                    embedding = combined.cpu().numpy()
                else:
                    # Непредвиденная размерность, создаем нулевой вектор нужной размерности
                    print(f"Warning: Unexpected image embedding dimension: {img_embedding.shape[1]}, expected {IMAGE_EMBED_DIM}")
                    embedding = np.zeros((1, FUSION_EMBED_DIM), dtype=np.float32)
            else:
                embedding = img_embedding.cpu().numpy()
                
        elif current_mode == SearchMode.TEXTUAL.value and description:
            # Только текстовый поиск
            if description is None:
                raise HTTPException(status_code=400, detail="Description is required for textual search mode")
            
            # Создаем нулевой вектор размерности FUSION_EMBED_DIM
            # Получаем текстовый эмбеддинг и конкатенируем с нулями для изображения
            text_embedding = encoder.encode_text(description)
            
            # Убедимся, что эмбеддинг имеет правильную размерность
            if len(text_embedding.shape) == 1:
                text_embedding = text_embedding.unsqueeze(0)
                
            # Создаем нулевой вектор для изображения
            img_embedding = torch.zeros((1, IMAGE_EMBED_DIM), device=text_embedding.device)
            
            # Конкатенируем
            combined = torch.cat([img_embedding, text_embedding], dim=1)
            embedding = combined.cpu().numpy()
        else:
            # Комбинированный поиск (по умолчанию)
            if img_bytes and description:
                embedding = encoder.encode(img_bytes, description)
            elif img_bytes:
                img_embedding = encoder.encode_image(img_bytes)
                # Убедимся, что эмбеддинг имеет правильную размерность
                if len(img_embedding.shape) == 1:
                    img_embedding = img_embedding.unsqueeze(0)
                    
                # Если размерность не соответствует ожидаемой, дополняем нулями
                if img_embedding.shape[1] != FUSION_EMBED_DIM:
                    if img_embedding.shape[1] == IMAGE_EMBED_DIM:
                        # Дополняем нулевым текстовым эмбеддингом
                        text_embedding = torch.zeros((1, TEXT_EMBED_DIM), device=img_embedding.device)
                        combined = torch.cat([img_embedding, text_embedding], dim=1)
                        embedding = combined.cpu().numpy()
                    else:
                        # Непредвиденная размерность, создаем нулевой вектор нужной размерности
                        print(f"Warning: Unexpected image embedding dimension: {img_embedding.shape[1]}, expected {IMAGE_EMBED_DIM}")
                        embedding = np.zeros((1, FUSION_EMBED_DIM), dtype=np.float32)
                else:
                    embedding = img_embedding.cpu().numpy()
            elif description:
                if description is None:
                    raise HTTPException(status_code=400, detail="Either image or description must be provided")
                    
                # Создаем нулевой вектор размерности FUSION_EMBED_DIM
                # Получаем текстовый эмбеддинг и конкатенируем с нулями для изображения
                text_embedding = encoder.encode_text(description)
                
                # Убедимся, что эмбеддинг имеет правильную размерность
                if len(text_embedding.shape) == 1:
                    text_embedding = text_embedding.unsqueeze(0)
                    
                # Создаем нулевой вектор для изображения
                img_embedding = torch.zeros((1, IMAGE_EMBED_DIM), device=text_embedding.device)
                
                # Конкатенируем
                combined = torch.cat([img_embedding, text_embedding], dim=1)
                embedding = combined.cpu().numpy()
            else:
                raise HTTPException(status_code=400, detail="Either image or description must be provided")
        
        # Поиск с учетом персонализации и диверсификации
        results = vector_store.search(
            embedding, 
            top_k=top_k, 
            user_id=user_id,
            boost_popular=True,
            diversify=diversify
        )
        
        # Если пользователь аутентифицирован, автоматически записываем событие просмотра
        if current_user:
            try:
                for hit in results:
                    vector_store.record_feedback(
                        hit["uuid"],
                        "view",
                        current_user.id,
                        hit["score"]  # Используем score как значение
                    )
            except Exception as e:
                print(f"Warning: Failed to record automatic feedback: {str(e)}")
        
        # Удаляем служебные поля из результатов
        for hit in results:
            if "content_hash" in hit:
                del hit["content_hash"]
        
        return {"results": results}
    
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/items/{item_uuid}", response_model=ItemResponse, summary="Fetch stored item metadata")
async def get_item(
    item_uuid: str,
    current_user: Optional[UserResponse] = Depends(get_current_user_optional),
):
    """
    Получает метаданные NFT по его UUID.
    
    - **item_uuid**: Уникальный идентификатор NFT
    
    Возвращает сохраненные метаданные NFT (имя, описание и т.д.).
    """
    meta = vector_store.get(item_uuid)
    if meta is None:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Удаляем служебные поля
    if "content_hash" in meta:
        del meta["content_hash"]
    
    # Добавляем UUID в результат
    meta["uuid"] = item_uuid
    
    # Проверяем, есть ли элемент в избранном у пользователя
    meta["is_favorite"] = False
    if current_user:
        try:
            # Используем универсальную функцию для проверки избранного
            feedback_data = vector_store.feedback.feedback_data
            meta["is_favorite"] = check_if_favorite(item_uuid, current_user.id, feedback_data)
            print(f"DEBUG: Item {item_uuid} is_favorite for user {current_user.id}: {meta['is_favorite']}")
        except Exception as e:
            print(f"Warning: Failed to check if item is in favorites: {str(e)}")
    
    # Если пользователь аутентифицирован, записываем событие просмотра
    if current_user:
        try:
            vector_store.record_feedback(
                item_uuid,
                "view",
                current_user.id,
                1.0
            )
        except Exception as e:
            print(f"Warning: Failed to record automatic feedback: {str(e)}")
    
    return meta


@router.post("/feedback", response_model=FeedbackResponse, summary="Submit user feedback")
async def submit_feedback(
    feedback: FeedbackRequest,
    current_user: Optional[UserResponse] = Depends(get_current_user) if REQUIRE_AUTH_FOR_FEEDBACK else None,
):
    """
    Отправляет обратную связь по NFT элементу.
    
    - **feedback_type**: Тип обратной связи (click, view, favorite, purchase, relation)
    - **item_uuid**: UUID элемента
    - **user_id**: ID пользователя (опционально)
    - **value**: Значение обратной связи от 0 до 1 (опционально, по умолчанию 1.0)
    - **related_uuid**: UUID связанного элемента (для связей)
    
    Возвращает статус операции и сообщение.
    """
    print(f"DEBUG: Received feedback request: {feedback}")
    
    # If authentication is required, use the authenticated user's ID
    if REQUIRE_AUTH_FOR_FEEDBACK and current_user:
        feedback.user_id = current_user.id
    elif current_user and not feedback.user_id:
        # Если пользователь аутентифицирован, но user_id не указан
        feedback.user_id = current_user.id
    
    print(f"DEBUG: Processing feedback with user_id: {feedback.user_id}, type: {feedback.feedback_type.value}, item: {feedback.item_uuid}")
        
    # Проверяем, существует ли элемент
    if vector_store.get(feedback.item_uuid) is None:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Записываем обратную связь
    try:
        # Используем значение value или дефолтное значение 1.0
        feedback_value = 1.0
        if feedback.value is not None:
            feedback_value = float(feedback.value)
        
        print(f"DEBUG: Before recording feedback, checking current feedback data")
        feedback_data = vector_store.feedback.feedback_data
        print(f"DEBUG: Current feedback data keys: {feedback_data.keys()}")
        
        # Особая проверка для лайков (favorite)
        is_favorite_before = False
        if feedback.feedback_type.value == "favorite" and feedback.user_id:
            # Проверяем user_preferences
            user_preferences = feedback_data.get("user_preferences", {}).get(feedback.user_id, {})
            if feedback.item_uuid in user_preferences:
                is_favorite_before = user_preferences[feedback.item_uuid] > 0
                print(f"DEBUG: is_favorite_before (from user_preferences): {is_favorite_before}")
            
            # Проверяем favorites_users
            favorites_users = feedback_data.get("favorites_users", {}).get(feedback.item_uuid, [])
            if feedback.user_id in favorites_users:
                is_favorite_before = True
                print(f"DEBUG: is_favorite_before (from favorites_users): {is_favorite_before}")
            
            print(f"DEBUG: Current is_favorite for item {feedback.item_uuid} and user {feedback.user_id}: {is_favorite_before}")
            
        # Базовая проверка избранного
        if "favorites" in feedback_data:
            print(f"DEBUG: Current favorites count: {len(feedback_data['favorites'])}")
            if feedback.item_uuid in feedback_data['favorites']:
                print(f"DEBUG: Item {feedback.item_uuid} already in favorites: {feedback_data['favorites'][feedback.item_uuid]}")
        
        if "user_preferences" in feedback_data:
            print(f"DEBUG: Current user_preferences count: {len(feedback_data['user_preferences'])}")
            if feedback.user_id in feedback_data['user_preferences']:
                print(f"DEBUG: User {feedback.user_id} has preferences: {len(feedback_data['user_preferences'][feedback.user_id])} items")
                if feedback.item_uuid in feedback_data['user_preferences'][feedback.user_id]:
                    print(f"DEBUG: Item {feedback.item_uuid} already in user preferences with value: {feedback_data['user_preferences'][feedback.user_id][feedback.item_uuid]}")
            
        print(f"DEBUG: Recording feedback: {feedback.item_uuid}, {feedback.feedback_type.value}, {feedback.user_id}, {feedback_value}")
        vector_store.record_feedback(
            feedback.item_uuid, 
            feedback.feedback_type.value, 
            feedback.user_id, 
            feedback_value
        )
        
        # Проверяем, как изменились данные после сохранения
        print(f"DEBUG: After recording feedback, checking updated feedback data")
        updated_feedback = vector_store.feedback.feedback_data
        
        # Особая проверка для лайков (favorite) после обновления
        is_favorite_after = False
        if feedback.feedback_type.value == "favorite" and feedback.user_id:
            # Проверяем user_preferences после обновления
            user_preferences = updated_feedback.get("user_preferences", {}).get(feedback.user_id, {})
            if feedback.item_uuid in user_preferences:
                is_favorite_after = user_preferences[feedback.item_uuid] > 0
                print(f"DEBUG: is_favorite_after (from user_preferences): {is_favorite_after}, value={user_preferences[feedback.item_uuid]}")
            
            # Проверяем favorites_users после обновления
            favorites_users = updated_feedback.get("favorites_users", {}).get(feedback.item_uuid, [])
            if feedback.user_id in favorites_users:
                is_favorite_after = True
                print(f"DEBUG: is_favorite_after (from favorites_users): {is_favorite_after}")
            
            print(f"DEBUG: After update, is_favorite for item {feedback.item_uuid} and user {feedback.user_id}: {is_favorite_after}")
            print(f"DEBUG: Value changed: {is_favorite_before} -> {is_favorite_after}")
            
        # Общая проверка обновления данных
        if "favorites" in updated_feedback:
            print(f"DEBUG: Updated favorites count: {len(updated_feedback['favorites'])}")
            if feedback.item_uuid in updated_feedback['favorites']:
                print(f"DEBUG: Item {feedback.item_uuid} in favorites after update: {updated_feedback['favorites'][feedback.item_uuid]}")
        
        if "user_preferences" in updated_feedback:
            print(f"DEBUG: Updated user_preferences count: {len(updated_feedback['user_preferences'])}")
            if feedback.user_id in updated_feedback['user_preferences']:
                print(f"DEBUG: User {feedback.user_id} has updated preferences: {len(updated_feedback['user_preferences'][feedback.user_id])} items")
                if feedback.item_uuid in updated_feedback['user_preferences'][feedback.user_id]:
                    print(f"DEBUG: Item {feedback.item_uuid} in user preferences after update with value: {updated_feedback['user_preferences'][feedback.user_id][feedback.item_uuid]}")
        
        return {"status": "success", "message": "Feedback recorded"}
    except Exception as e:
        print(f"ERROR in submit_feedback: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/explore", response_model=ExploreResponse, summary="Explore/browse NFTs")
async def explore_nfts(
    category: Optional[str] = Query(None, description="Filter by category"),
    style: Optional[str] = Query(None, description="Filter by style"),
    tags: Optional[str] = Query(None, description="Filter by comma-separated tags"),
    limit: int = Query(20, ge=1, le=100, description="Number of results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    sort_by: Optional[str] = Query("popularity", description="Sort by field (popularity, newest)"),
    current_user: Optional[UserResponse] = Depends(get_current_user_optional),
):
    """
    Просмотр и исследование NFT с фильтрацией и сортировкой.
    
    - **category**: Фильтр по категории (опционально)
    - **style**: Фильтр по стилю (опционально)
    - **tags**: Фильтр по тегам через запятую (опционально)
    - **limit**: Количество результатов (от 1 до 100)
    - **offset**: Смещение для пагинации
    - **sort_by**: Поле для сортировки (popularity, newest)
    
    Возвращает список NFT с метаданными, отсортированный согласно параметрам.
    """
    try:
        print("DEBUG: Starting explore_nfts method")
        
        # Получаем все доступные элементы без фильтрации
        items = []
        results = []
        
        # Безопасный доступ к индексу
        try:
            # Безопасный доступ к индексу
            from .index import vector_store as vs
            
            # Попытка доступа к metadata через атрибут
            metadata = getattr(vs, "metadata", None)
            if metadata and isinstance(metadata, dict):
                for uuid, item_data in metadata.items():
                    if not item_data.get("deleted", False):
                        item = dict(item_data)
                        item["uuid"] = uuid
                        
                        # Убедимся, что необходимые поля существуют
                        if "categories" not in item:
                            item["categories"] = []
                        if "styles" not in item:
                            item["styles"] = []
                        if "tags" not in item:
                            item["tags"] = []
                        if "score" not in item:
                            item["score"] = 1.0
                            
                        items.append(item)
            else:
                # Альтернативный метод через api
                uuid_list = vs.get_all_uuids() if hasattr(vs, 'get_all_uuids') else []
                for uuid in uuid_list:
                    try:
                        item = vs.get(uuid)
                        if item and not item.get("deleted", False):
                            item["uuid"] = uuid
                            
                            # Убедимся, что необходимые поля существуют
                            if "categories" not in item:
                                item["categories"] = []
                            if "styles" not in item:
                                item["styles"] = []
                            if "tags" not in item:
                                item["tags"] = []
                            if "score" not in item:
                                item["score"] = 1.0
                                
                            items.append(item)
                    except Exception:
                        continue
                        
            if not items:
                print("WARNING: No items found in index, returning sample data")
                # Если элементы не найдены, возвращаем тестовые данные
                sample_data = [
                    {
                        "uuid": f"sample-{i}",
                        "name": f"Sample NFT {i}",
                        "description": "This is a sample NFT for testing",
                        "categories": ["art", "collectible"],
                        "styles": ["digital", "abstract"],
                        "tags": ["sample", "test", "demo"],
                        "score": 1.0,
                        "created_at": datetime.now().isoformat()
                    } for i in range(1, 6)
                ]
                items.extend(sample_data)
                
        except Exception as e:
            print(f"ERROR accessing vector store: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Возвращаем тестовые данные в случае ошибки
            sample_data = [
                {
                    "uuid": f"sample-{i}",
                    "name": f"Sample NFT {i}",
                    "description": "This is a sample NFT for testing",
                    "categories": ["art", "collectible"],
                    "styles": ["digital", "abstract"],
                    "tags": ["sample", "test", "demo"],
                    "score": 1.0,
                    "created_at": datetime.now().isoformat()
                } for i in range(1, 6)
            ]
            items = sample_data
        
        print(f"DEBUG: Got {len(items)} items from index")
        
        # Фильтруем элементы согласно заданным фильтрам
        for item in items:
            # Применяем фильтры
            if category and category.lower() != "all" and category not in item.get("categories", []):
                continue
                
            if style and style.lower() != "all" and style not in item.get("styles", []):
                continue
                
            if tags and tags.lower() != "all":
                tag_list = [tag.strip().lower() for tag in tags.split(",")]
                item_tags = [tag.lower() for tag in item.get("tags", [])]
                if not any(tag in item_tags for tag in tag_list):
                    continue
            
            # Добавляем элемент в результаты после фильтрации
            results.append(item)
        
        print(f"DEBUG: After filtering: {len(results)} items remain")
        
        # Сортировка результатов
        if sort_by == "newest":
            results.sort(key=lambda x: x.get("created_at", "2023-01-01T00:00:00"), reverse=True)
        else:  # Сортировка по популярности по умолчанию
            results.sort(key=lambda x: x.get("popularity", 0) if "popularity" in x else 0, reverse=True)
        
        # Пагинация
        total = len(results)
        pages = (total + limit - 1) // limit if limit > 0 else 1
        page = offset // limit + 1 if limit > 0 else 1
        
        # Применяем пагинацию
        paginated_results = results[offset:offset+limit]
        
        # Если пользователь аутентифицирован, добавим информацию о избранных элементах
        if current_user:
            try:
                # Получаем данные о избранном для пользователя
                feedback_data = vector_store.feedback.feedback_data
                
                print(f"DEBUG: In explore_nfts, checking favorites for user {current_user.id}")
                
                # Проверяем каждый элемент на наличие в избранном, используя общую функцию
                for item in paginated_results:
                    item_uuid = item["uuid"]
                    item["is_favorite"] = check_if_favorite(item_uuid, current_user.id, feedback_data)
                    
                    # Записываем событие просмотра
                    vector_store.record_feedback(
                        item_uuid,
                        "view",
                        current_user.id,
                        0.5  # Меньшее значение для обзора, чем для прямого просмотра
                    )
            except Exception as e:
                print(f"Warning: Failed to record automatic feedback: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            # Если пользователь не аутентифицирован, просто устанавливаем is_favorite в False
            for item in paginated_results:
                item["is_favorite"] = False
        
        print(f"DEBUG: Returning {len(paginated_results)} items (page {page}/{pages}, total: {total})")
        
        return {
            "results": paginated_results,
            "total": total,
            "page": page,
            "pages": pages
        }
    except Exception as e:
        print(f"ERROR in explore_nfts: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # В случае ошибки возвращаем тестовые данные
        sample_data = [
            {
                "uuid": f"sample-{i}",
                "name": f"Sample NFT {i}",
                "description": "This is a sample NFT for testing",
                "categories": ["art", "collectible"],
                "styles": ["digital", "abstract"],
                "tags": ["sample", "test", "demo"],
                "score": 1.0,
                "created_at": datetime.now().isoformat()
            } for i in range(1, 6)
        ]
        
        return {
            "results": sample_data,
            "total": len(sample_data),
            "page": 1,
            "pages": 1
        }


@router.get("/health", summary="Liveness probe")
async def health():
    """
    Проверка работоспособности API.
    Возвращает статус 'ok', если сервис работает корректно.
    """
    return {"status": "ok"}


@router.get("/stats", response_model=IndexStatsResponse, summary="Get index statistics")
async def stats(current_user: Optional[UserResponse] = Depends(get_current_user_optional)):
    """
    Получение статистики индекса.
    
    Возвращает информацию о количестве элементов в индексе, 
    размерности векторов и другие статистические данные.
    
    Для авторизованных пользователей возвращает персонализированную статистику.
    """
    stats_data = vector_store.get_stats()
    
    # Если пользователь авторизован, возвращаем только его статистику
    if current_user:
        user_id = current_user.id
        
        # Получаем данные обратной связи пользователя
        feedback_data = vector_store.feedback.feedback_data
        
        # Получаем данные о избранном для пользователя - используем множество для исключения дублирования
        user_favorites = set()
        
        # Собираем избранные элементы из user_preferences
        user_preferences = feedback_data.get("user_preferences", {}).get(user_id, {})
        for item_uuid, value in user_preferences.items():
            if value > 0 and not vector_store.metadata.get(item_uuid, {}).get("deleted", False):
                user_favorites.add(item_uuid)
        
        # Добавляем элементы из favorites_users
        favorites_users = feedback_data.get("favorites_users", {})
        for item_uuid, users in favorites_users.items():
            if user_id in users and not vector_store.metadata.get(item_uuid, {}).get("deleted", False):
                user_favorites.add(item_uuid)
        
        # Правильный подсчет избранных (без дублирования)
        user_favorites_count = len(user_favorites)
        
        # Счетчик взаимодействий - правильный подход
        user_interactions = 0
        
        # Создаем множество всех уникальных элементов, с которыми взаимодействовал пользователь
        interaction_items = set()
        
        # Считаем клики - используем данные из clicks
        clicks_count = 0
        clicks_data = feedback_data.get("clicks", {})
        for item_uuid, items in clicks_data.items():
            if not vector_store.metadata.get(item_uuid, {}).get("deleted", False):
                # Проверяем новую структуру данных (словарь)
                if isinstance(items, dict) and user_id in items:
                    clicks_count += 1
                    interaction_items.add(item_uuid)
                # Проверяем старую структуру (список)
                elif isinstance(items, list) and any(record.get("user_id") == user_id for record in items):
                    clicks_count += 1
                    interaction_items.add(item_uuid)
        
        # Добавляем количество просмотров
        views_count = 0
        views_data = feedback_data.get("views", {})
        for item_uuid, items in views_data.items():
            if isinstance(items, dict) and user_id in items and not vector_store.metadata.get(item_uuid, {}).get("deleted", False):
                views_count += 1
                interaction_items.add(item_uuid)
            elif isinstance(items, list) and any(record.get("user_id") == user_id for record in items) and not vector_store.metadata.get(item_uuid, {}).get("deleted", False):
                views_count += 1
                interaction_items.add(item_uuid)
        
        # Общее количество взаимодействий - считаем все уникальные взаимодействия между пользователем и элементами
        # Удаляем этот дублирующий код, так как interaction_items уже инициализировано выше
        # interaction_items = set()
        
        # Добавляем элементы с кликами - Эта секция дублирует код выше, поэтому удаляем
        # for item_uuid, items in clicks_data.items():
        #    if not vector_store.metadata.get(item_uuid, {}).get("deleted", False):
        #        if isinstance(items, dict) and user_id in items:
        #            interaction_items.add(item_uuid)
        #        elif isinstance(items, list) and any(record.get("user_id") == user_id for record in items):
        #            interaction_items.add(item_uuid)
        
        # Добавляем элементы с просмотрами - Эта секция дублирует код выше, поэтому удаляем
        # for item_uuid, items in views_data.items():
        #    if not vector_store.metadata.get(item_uuid, {}).get("deleted", False):
        #        if isinstance(items, dict) and user_id in items:
        #            interaction_items.add(item_uuid)
        #        elif isinstance(items, list) and any(record.get("user_id") == user_id for record in items):
        #            interaction_items.add(item_uuid)
        
        # Добавляем избранные элементы
        interaction_items.update(user_favorites)
        
        # Общее количество уникальных взаимодействий
        user_interactions = len(interaction_items)
        
        print(f"DEBUG: User {user_id} statistics: favorites={user_favorites_count}, clicks={clicks_count}, views={views_count}, interaction_items={len(interaction_items)}")
        
        # Формируем статистику для конкретного пользователя
        user_stats = {
            "total_items": stats_data["active_items"],
            "active_items": stats_data["active_items"],
            "dimension": stats_data["dimension"],
            "index_type": stats_data["index_type"],
            "feedback": {
                "total_clicks": clicks_count,
                "total_views": views_count,
                "total_favorites": user_favorites_count,
                "total_purchases": 0,  # Пока нет функционала покупок
                "users_with_preferences": 1 if user_favorites_count > 0 else 0,
                "total_interactions": user_interactions
            }
        }
        return user_stats
    
    return stats_data


@router.get("/stats/all", response_model=IndexStatsResponse, summary="Get all index statistics")
async def all_stats():
    """
    Получение полной статистики индекса по всем пользователям.
    
    Возвращает информацию о количестве элементов в индексе, 
    общую статистику по всем пользователям.
    
    Не требует авторизации.
    """
    return vector_store.get_stats()


@router.delete("/items/{item_uuid}", summary="Delete an NFT item")
async def delete_item(
    item_uuid: str,
    current_user: UserResponse = Depends(get_current_user) if REQUIRE_AUTH_FOR_WRITE else None
):
    """
    Удаляет NFT элемент из системы по его UUID.
    
    - **item_uuid**: Уникальный идентификатор NFT
    
    Возвращает подтверждение удаления.
    """
    # Проверяем, существует ли элемент
    item = vector_store.get(item_uuid)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Проверяем права доступа: только владелец или админ может удалить элемент
    if current_user and REQUIRE_AUTH_FOR_WRITE:
        if item.get("created_by") and item["created_by"] != current_user.id and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to delete this item"
            )
    
    if not vector_store.delete(item_uuid):
        raise HTTPException(status_code=404, detail="Item not found")
    
    return {"message": f"Item {item_uuid} successfully deleted"}


@router.get("/trending", response_model=ExploreResponse, summary="Get trending NFTs")
async def trending_nfts(
    limit: int = Query(20, ge=1, le=100, description="Number of results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    current_user: Optional[UserResponse] = Depends(get_current_user_optional),
):
    """
    Получение популярных NFT на основе взаимодействий пользователей.
    
    Возвращает список NFT, отсортированных по количеству взаимодействий (просмотры, избранное).
    
    - **limit**: Максимальное количество результатов
    - **offset**: Смещение для пагинации
    """
    try:
        # Получение всех элементов
        all_items = []
        for uuid, meta in vector_store.metadata.items():
            if not meta.get('deleted', False):
                item_data = meta.copy()
                item_data['uuid'] = uuid
                
                # Расчет показателя популярности
                clicks = vector_store.feedback.feedback_data["clicks"].get(uuid, 0)
                favorites = vector_store.feedback.feedback_data["favorites"].get(uuid, 0)
                
                # Формула популярности: клики + избранное*3 (избранное имеет больший вес)
                popularity_score = clicks + favorites * 3
                item_data['popularity_score'] = popularity_score
                
                all_items.append(item_data)
        
        # Сортировка по популярности
        all_items.sort(key=lambda x: x.get('popularity_score', 0), reverse=True)
        
        # Вычисление пагинации
        total = len(all_items)
        pages = (total + limit - 1) // limit if total > 0 else 0
        page = offset // limit + 1 if limit > 0 else 1
        
        # Применение пагинации
        paginated_items = all_items[offset:offset+limit]
        
        # Удаляем служебные поля
        for item in paginated_items:
            if 'content_hash' in item:
                del item['content_hash']
            if 'popularity_score' in item:
                del item['popularity_score']
        
        # Если пользователь аутентифицирован, добавим информацию о избранных элементах
        if current_user:
            try:
                # Получаем данные о избранном для пользователя
                feedback_data = vector_store.feedback.feedback_data
                
                # Проверяем каждый элемент на наличие в избранном, используя общую функцию
                for item in paginated_items:
                    item_uuid = item["uuid"]
                    item["is_favorite"] = check_if_favorite(item_uuid, current_user.id, feedback_data)
                    
                    # Записываем событие просмотра с меньшим весом
                    vector_store.record_feedback(
                        item_uuid,
                        "view",
                        current_user.id,
                        0.3  # Меньший вес для обзора популярных элементов
                    )
            except Exception as e:
                print(f"Warning: Failed to check favorites: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            # Если пользователь не аутентифицирован, просто устанавливаем is_favorite в False
            for item in paginated_items:
                item["is_favorite"] = False
        
        return {
            "results": paginated_items,
            "total": total,
            "page": page,
            "pages": pages
        }
    except Exception as e:
        print(f"ERROR in trending_nfts: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # В случае ошибки возвращаем пустой результат
        return {
            "results": [],
            "total": 0,
            "page": 1,
            "pages": 0
        }


@router.get("/user/favorites", response_model=ExploreResponse, summary="Get user's favorite items")
async def get_user_favorites(
    limit: int = Query(20, ge=1, le=100, description="Number of results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    current_user: Optional[UserResponse] = Depends(get_current_user_optional),
):
    """
    Получает список избранных элементов пользователя.
    
    - **limit**: Количество результатов (от 1 до 100)
    - **offset**: Смещение для пагинации
    
    Если пользователь не аутентифицирован, используется ID из параметра запроса.
    Возвращает список NFT с метаданными, добавленных в избранное пользователем.
    """
    try:
        # Получаем ID пользователя
        user_id = None
        if current_user:
            user_id = current_user.id
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required to access favorites"
            )
            
        # Получаем данные об избранном из системы обратной связи
        try:
            # Собираем все избранные элементы пользователя
            user_favorites = set()
            feedback_data = vector_store.feedback.feedback_data
            
            # Проверяем user_preferences
            user_preferences = feedback_data.get("user_preferences", {}).get(user_id, {})
            for item_uuid, value in user_preferences.items():
                if value > 0 and not vector_store.metadata.get(item_uuid, {}).get("deleted", False):
                    user_favorites.add(item_uuid)
            
            # Проверяем избранные элементы через favorites_users
            favorites_users = feedback_data.get("favorites_users", {})
            for item_uuid, users in favorites_users.items():
                if user_id in users and not vector_store.metadata.get(item_uuid, {}).get("deleted", False):
                    user_favorites.add(item_uuid)
            
            print(f"DEBUG: Found {len(user_favorites)} favorites for user {user_id}")
            
            # Преобразуем в список для пагинации
            user_favorites_list = list(user_favorites)
            
            # Применяем пагинацию
            paginated_favorites = user_favorites_list[offset:offset + limit]
            
            # Загружаем полные данные для каждого избранного элемента
            results = []
            for fav_uuid in paginated_favorites:
                item_data = vector_store.get(fav_uuid)
                if item_data and not item_data.get("deleted", False):
                    item_data["uuid"] = fav_uuid
                    # Поскольку это избранные элементы, устанавливаем is_favorite = True
                    item_data["is_favorite"] = True
                    results.append(item_data)
            
            # Возвращаем в формате ExploreResponse
            return {
                "results": results,
                "total": len(user_favorites),
                "page": offset // limit + 1 if limit > 0 else 1,
                "pages": (len(user_favorites) + limit - 1) // limit if limit > 0 else 1
            }
            
        except Exception as e:
            print(f"ERROR getting user favorites: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # В случае ошибки возвращаем пустой список
            return {
                "results": [],
                "total": 0,
                "page": 1,
                "pages": 1
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving favorites: {str(e)}"
        )


@router.get("/users/{user_id}/favorites", response_model=ExploreResponse, summary="Get favorite items by user ID")
async def get_user_favorites_by_id(
    user_id: str = Path(..., description="User ID"),
    limit: int = Query(20, ge=1, le=100, description="Number of results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    current_user: Optional[UserResponse] = Depends(get_current_user_optional),
):
    """
    Получает список избранных элементов для конкретного пользователя по ID.
    
    - **user_id**: ID пользователя
    - **limit**: Количество результатов (от 1 до 100)
    - **offset**: Смещение для пагинации
    
    Если требуется получить избранное для текущего пользователя, используйте /user/favorites.
    Возвращает список NFT с метаданными, добавленных в избранное указанным пользователем.
    """
    try:
        # Получаем данные об избранном из системы обратной связи
        try:
            # Собираем все избранные элементы пользователя
            user_favorites = set()
            feedback_data = vector_store.feedback.feedback_data
            
            # Проверяем user_preferences
            user_preferences = feedback_data.get("user_preferences", {}).get(user_id, {})
            for item_uuid, value in user_preferences.items():
                if value > 0 and not vector_store.metadata.get(item_uuid, {}).get("deleted", False):
                    user_favorites.add(item_uuid)
            
            # Проверяем избранные элементы через favorites_users
            favorites_users = feedback_data.get("favorites_users", {})
            for item_uuid, users in favorites_users.items():
                if user_id in users and not vector_store.metadata.get(item_uuid, {}).get("deleted", False):
                    user_favorites.add(item_uuid)
            
            print(f"DEBUG: Found {len(user_favorites)} favorites for user {user_id}")
            
            # Преобразуем в список для пагинации
            user_favorites_list = list(user_favorites)
            
            # Применяем пагинацию
            paginated_favorites = user_favorites_list[offset:offset + limit]
            
            # Загружаем полные данные для каждого избранного элемента
            results = []
            for fav_uuid in paginated_favorites:
                item_data = vector_store.get(fav_uuid)
                if item_data and not item_data.get("deleted", False):
                    item_data["uuid"] = fav_uuid
                    # Поскольку это избранные элементы, устанавливаем is_favorite = True для всех
                    item_data["is_favorite"] = True
                    # Если текущий пользователь смотрит на чужое избранное, is_favorite будет true только
                    # если элемент есть и у него в избранном
                    if current_user and current_user.id != user_id:
                        current_user_favs = set()
                        current_user_prefs = feedback_data.get("user_preferences", {}).get(current_user.id, {})
                        
                        # Проверяем через user_preferences
                        for item_id, value in current_user_prefs.items():
                            if value > 0:
                                current_user_favs.add(item_id)
                        
                        # Проверяем через favorites_users
                        for item_id, users in favorites_users.items():
                            if current_user.id in users:
                                current_user_favs.add(item_id)
                                
                        if fav_uuid not in current_user_favs:
                            item_data["is_favorite"] = False
                    
                    results.append(item_data)
            
            # Возвращаем в формате ExploreResponse
            return {
                "results": results,
                "total": len(user_favorites),
                "page": offset // limit + 1 if limit > 0 else 1,
                "pages": (len(user_favorites) + limit - 1) // limit if limit > 0 else 1
            }
            
        except Exception as e:
            print(f"ERROR getting user favorites: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # В случае ошибки возвращаем пустой список
            return {
                "results": [],
                "total": 0,
                "page": 1,
                "pages": 1
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving favorites: {str(e)}"
        )


@router.get("/items/{item_uuid}/favorite", summary="Check if item is in user's favorites")
async def check_item_favorite(
    item_uuid: str,
    current_user: Optional[UserResponse] = Depends(get_current_user_optional),
):
    """
    Проверяет, добавлен ли элемент в избранное пользователя.
    
    - **item_uuid**: Уникальный идентификатор NFT
    
    Возвращает статус избранного для текущего пользователя.
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required to check favorite status"
        )
    
    # Проверяем существование элемента
    meta = vector_store.get(item_uuid)
    if meta is None:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Проверяем, есть ли элемент в избранном у пользователя
    is_favorite = False
    try:
        print(f"DEBUG: Checking favorite status for item {item_uuid} and user {current_user.id}")
        feedback_data = vector_store.feedback.feedback_data
        
        # Единый метод проверки избранного, который должен использоваться во всех эндпоинтах
        is_favorite = check_if_favorite(item_uuid, current_user.id, feedback_data)
        
        print(f"DEBUG: Final is_favorite value: {is_favorite}")
    except Exception as e:
        print(f"WARNING: Failed to check if item is in favorites: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return {"is_favorite": is_favorite, "item_uuid": item_uuid, "user_id": current_user.id}

# Вспомогательная функция для проверки, находится ли элемент в избранном у пользователя
# Позволяет унифицировать логику проверки во всех эндпоинтах
def check_if_favorite(item_uuid: str, user_id: str, feedback_data: dict) -> bool:
    """
    Проверяет, находится ли элемент в избранном у пользователя.
    
    Args:
        item_uuid: UUID элемента
        user_id: ID пользователя
        feedback_data: Данные обратной связи
        
    Returns:
        bool: True если элемент в избранном, False в противном случае
    """
    # Проверяем в user_preferences
    user_preferences = feedback_data.get("user_preferences", {}).get(user_id, {})
    if item_uuid in user_preferences and user_preferences[item_uuid] > 0:
        return True
    
    # Проверяем в favorites_users
    favorites_users = feedback_data.get("favorites_users", {})
    if item_uuid in favorites_users and user_id in favorites_users[item_uuid]:
        return True
    
    # Проверяем в favorites
    favorites = feedback_data.get("favorites", {})
    if item_uuid in favorites:
        favor_data = favorites[item_uuid]
        if isinstance(favor_data, list):
            if any(record.get("user_id") == user_id for record in favor_data):
                return True
        elif isinstance(favor_data, dict) and user_id in favor_data:
            return True
            
    return False


@router.get("/recommendations", response_model=ExploreResponse, summary="Get personalized recommendations")
async def get_recommendations(
    limit: int = Query(20, ge=1, le=100, description="Number of results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    user: UserResponse = Depends(get_current_user),
):
    """
    Получение персонализированных рекомендаций для авторизованного пользователя.
    
    - **limit**: Количество результатов (от 1 до 100)
    - **offset**: Смещение для пагинации
    
    Возвращает список NFT, рекомендованных пользователю на основе его предпочтений.
    """
    
    # Инициализируем список для хранения результатов
    all_results = []
    
    try:
        # Получаем идентификатор пользователя
        user_id = user.id
        
        # Получаем список рекомендаций
        recommendation_results = recommender.get_recommendations(user_id, limit=limit * 5)  # Запрашиваем больше для фильтрации
        
        if recommendation_results:
            # Фильтруем только активные элементы
            active_results = []
            for item_uuid in recommendation_results:
                item_data = vector_store.get(item_uuid)
                if item_data and not item_data.get("deleted", False):
                    item_data["uuid"] = item_uuid
                    
                    # Проверяем, находится ли элемент в избранном у пользователя
                    feedback_data = vector_store.feedback.feedback_data
                    is_favorite = False
                    
                    # Проверяем через user_preferences
                    user_preferences = feedback_data.get("user_preferences", {}).get(user_id, {})
                    if item_uuid in user_preferences and user_preferences[item_uuid] > 0:
                        is_favorite = True
                    
                    # Проверяем через favorites_users
                    favorites_users = feedback_data.get("favorites_users", {})
                    if item_uuid in favorites_users and user_id in favorites_users[item_uuid]:
                        is_favorite = True
                    
                    item_data["is_favorite"] = is_favorite
                    active_results.append(item_data)
            
            # Применяем пагинацию
            all_results = active_results[offset:offset + limit]
        
        # Если рекомендаций недостаточно, дополняем из общего списка, отсортированного по популярности
        if len(all_results) < limit:
            # Получаем элементы, отсортированные по популярности
            remaining_count = limit - len(all_results)
            popular_items = vector_store.get_most_popular(offset + len(all_results), remaining_count)
            
            # Добавляем к результатам, если их еще нет в списке
            existing_uuids = {item.get("uuid") for item in all_results}
            
            for popular_item in popular_items:
                item_uuid = popular_item.get("uuid")
                if item_uuid and item_uuid not in existing_uuids and not popular_item.get("deleted", False):
                    # Проверяем, находится ли элемент в избранном у пользователя
                    feedback_data = vector_store.feedback.feedback_data
                    is_favorite = False
                    
                    # Проверяем через user_preferences
                    user_preferences = feedback_data.get("user_preferences", {}).get(user_id, {})
                    if item_uuid in user_preferences and user_preferences[item_uuid] > 0:
                        is_favorite = True
                    
                    # Проверяем через favorites_users
                    favorites_users = feedback_data.get("favorites_users", {})
                    if item_uuid in favorites_users and user_id in favorites_users[item_uuid]:
                        is_favorite = True
                    
                    popular_item["is_favorite"] = is_favorite
                    all_results.append(popular_item)
                    existing_uuids.add(item_uuid)
        
        # Получаем общее количество рекомендаций
        total_items = vector_store.count()
        
        # Формируем ответ с пагинацией
        return {
            "results": all_results,
            "total": total_items,
            "page": offset // limit + 1 if limit > 0 else 1,
            "pages": (total_items + limit - 1) // limit if limit > 0 else 1
        }
        
    except Exception as e:
        print(f"Error getting recommendations: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Если произошла ошибка, возвращаем пустой список
        return {
            "results": all_results,
            "total": 0,
            "page": 1,
            "pages": 1
        }
