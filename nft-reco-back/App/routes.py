"""HTTP endpoints grouped in a FastAPI router."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Query, Path, Body
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import torch
import numpy as np

from .encoders import encoder
from .index import vector_store
from .schemas import (
    RecommendResponse, IndexStatsResponse, FeedbackRequest, 
    FeedbackResponse, ExploreFilter, ExploreResponse, ItemResponse
)
from .config import SearchMode, ADVANCED_SEARCH_CONFIG, FUSION_EMBED_DIM, IMAGE_EMBED_DIM, TEXT_EMBED_DIM

router = APIRouter()


@router.post("/items", summary="Index a new NFT item")
async def add_item(
    image: UploadFile = File(..., description="NFT image file (jpeg/png)"),
    description: str = Form(..., description="Textual description / attributes"),
    name: str = Form("", description="Human‑readable name"),
    tags: Optional[str] = Form(None, description="Comma-separated tags (optional)"),
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
            "styles": generated_tags["styles"]
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
        
        # Удаляем служебные поля из результатов
        for hit in results:
            if "content_hash" in hit:
                del hit["content_hash"]
        
        return {"results": results}
    
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/items/{item_uuid}", response_model=ItemResponse, summary="Fetch stored item metadata")
async def get_item(item_uuid: str):
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
    
    return meta


@router.post("/feedback", response_model=FeedbackResponse, summary="Submit user feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Отправляет обратную связь от пользователя для улучшения рекомендаций.
    
    - **feedback_type**: Тип обратной связи (click, favorite, purchase)
    - **user_id**: ID пользователя (опционально)
    - **item_uuid**: UUID элемента
    - **related_uuid**: UUID связанного элемента (опционально)
    - **value**: Значение обратной связи (опционально)
    
    Возвращает статус успешного сохранения обратной связи.
    """
    # Проверяем, существует ли элемент
    if vector_store.get(feedback.item_uuid) is None:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Записываем обратную связь
    try:
        # Используем значение value или дефолтное значение 1.0
        feedback_value = 1.0
        if feedback.value is not None:
            feedback_value = float(feedback.value)
            
        vector_store.record_feedback(
            feedback.item_uuid, 
            feedback.feedback_type.value, 
            feedback.user_id, 
            feedback_value
        )
        return {"status": "success", "message": "Feedback recorded"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/explore", response_model=ExploreResponse, summary="Explore/browse NFTs")
async def explore_nfts(
    category: Optional[str] = Query(None, description="Filter by category"),
    style: Optional[str] = Query(None, description="Filter by style"),
    tags: Optional[str] = Query(None, description="Filter by comma-separated tags"),
    limit: int = Query(20, ge=1, le=100, description="Number of results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    sort_by: Optional[str] = Query("popularity", description="Sort by field (popularity, newest)"),
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
        
        # Получаем статистику индекса для проверки
        from .index import vector_store as vs
        stats = vs.get_stats()
        print(f"DEBUG: Vector store stats: {stats}")
        
        # Получаем все активные UUID и создаем результаты на основе get(uuid)
        items = []
        results = []
        
        print("DEBUG: Accessing index internal state")
        # Пытаемся получить все UUID напрямую
        try:
            uuid_list = list(vs.idx_to_uuid.values())
            print(f"DEBUG: Found {len(uuid_list)} UUIDs in index")
            
            # Получаем метаданные для каждого UUID
            for uuid in uuid_list:
                item = vs.get(uuid)
                if item and not item.get("deleted", False):
                    # Добавляем UUID в метаданные
                    item["uuid"] = uuid
                    items.append(item)
        except Exception as e:
            print(f"ERROR accessing idx_to_uuid: {str(e)}")
            # Пытаемся использовать альтернативный метод
            try:
                # Альтернативный метод - напрямую обратиться к metadata
                metadata = getattr(vs, "metadata", {})
                print(f"DEBUG: Found {len(metadata)} items in metadata")
                
                for uuid, item_data in metadata.items():
                    if not item_data.get("deleted", False):
                        item = dict(item_data)
                        item["uuid"] = uuid
                        items.append(item)
            except Exception as e2:
                print(f"ERROR accessing metadata: {str(e2)}")
                # Возвращаем пустой результат в случае ошибки
                return {
                    "results": [],
                    "total": 0,
                    "page": offset // limit + 1 if limit > 0 else 1,
                    "pages": 0
                }
        
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
            
            # Добавляем дополнительные поля если их нет
            if "score" not in item:
                item["score"] = 1.0
                
            # Добавляем элемент в результаты после фильтрации
            results.append(item)
        
        print(f"DEBUG: After filtering: {len(results)} items remain")
        
        # Сортировка результатов
        if sort_by == "newest":
            results.sort(key=lambda x: x.get("created_at", "2023-01-01T00:00:00"), reverse=True)
        else:  # Сортировка по популярности по умолчанию
            results.sort(key=lambda x: x.get("popularity", 0), reverse=True)
        
        # Пагинация
        total = len(results)
        pages = (total + limit - 1) // limit if limit > 0 else 1
        page = offset // limit + 1 if limit > 0 else 1
        
        # Применяем пагинацию
        paginated_results = results[offset:offset+limit]
        
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
        # В случае ошибки возвращаем пустой результат
        return {
            "results": [],
            "total": 0,
            "page": offset // limit + 1 if limit > 0 else 1,
            "pages": 0
        }


@router.get("/health", summary="Liveness probe")
async def health():
    """
    Проверка работоспособности API.
    Возвращает статус 'ok', если сервис работает корректно.
    """
    return {"status": "ok"}


@router.get("/stats", response_model=IndexStatsResponse, summary="Get index statistics")
async def stats():
    """
    Получение статистики индекса.
    
    Возвращает информацию о количестве элементов в индексе, 
    размерности векторов и другие статистические данные.
    """
    return vector_store.get_stats()


@router.delete("/items/{item_uuid}", summary="Delete an NFT item")
async def delete_item(item_uuid: str):
    """
    Удаляет NFT элемент из системы по его UUID.
    
    - **item_uuid**: Уникальный идентификатор NFT
    
    Возвращает подтверждение удаления.
    """
    if not vector_store.delete(item_uuid):
        raise HTTPException(status_code=404, detail="Item not found")
    
    return {"message": f"Item {item_uuid} successfully deleted"}
