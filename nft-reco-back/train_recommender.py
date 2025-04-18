#!/usr/bin/env python
"""
Скрипт для обучения и тестирования рекомендательной системы на 
сгенерированном наборе данных из NFT изображений.
"""
import os
import json
import argparse
import random
import time
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import sys

# Добавляем путь к корневой директории проекта
sys.path.append('.')

# Импортируем нашу рекомендательную систему
from App.index import VectorIndex, FeedbackSystem
from App.config import FUSION_EMBED_DIM, FAISS_INDEX_TYPE
from tests.metrics_utils import MetricsCollector, TimingContext

# Константы
DEFAULT_DATASET_DIR = "synthetic_dataset"
RESULTS_DIR = "recommender_results"
INDEX_FILE = "recommender_index.faiss"
META_FILE = "recommender_meta.json"
FEEDBACK_FILE = "recommender_feedback.json"

def load_dataset(dataset_dir: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Загружает тренировочные и тестовые данные
    
    Args:
        dataset_dir: Директория с набором данных
        
    Returns:
        Кортеж (тренировочные данные, тестовые данные)
    """
    base_dir = Path(dataset_dir)
    
    # Загружаем тренировочные данные
    with open(base_dir / "train" / "metadata.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    
    # Загружаем тестовые данные
    with open(base_dir / "test" / "metadata.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    print(f"Загружено {len(train_data)} тренировочных и {len(test_data)} тестовых образцов")
    return train_data, test_data

def train_recommender(train_data: List[Dict[str, Any]], 
                      index_file: str, 
                      meta_file: str, 
                      feedback_file: str) -> VectorIndex:
    """
    Обучает рекомендательную систему на предоставленных данных
    
    Args:
        train_data: Список метаданных для обучения
        index_file: Путь для сохранения векторного индекса
        meta_file: Путь для сохранения метаданных
        feedback_file: Путь для сохранения данных обратной связи
        
    Returns:
        Обученный экземпляр VectorIndex
    """
    # Инициализируем систему
    vector_index = VectorIndex(FUSION_EMBED_DIM, Path(index_file), Path(meta_file), Path(feedback_file))
    
    # Добавляем данные в индекс
    print("Добавление данных в векторный индекс...")
    for item in tqdm(train_data):
        # Преобразуем эмбеддинг обратно в numpy-массив
        embedding = np.array(item["combined_embedding"])
        
        # Создаем метаданные для индекса
        metadata = {
            "name": item["name"],
            "description": item["description"],
            "categories": item["categories"],
            "styles": item["styles"],
            "tags": item["tags"],
            "image_path": item["image_path"],
            "attributes": item["attributes"]
        }
        
        # Добавляем в индекс
        vector_index.add(embedding, metadata)
    
    # Симуляция обратной связи - создаем "популярные" элементы
    print("Симуляция обратной связи пользователей...")
    for _ in range(300):  # Симулируем 300 взаимодействий
        # Случайно выбираем элемент
        item = random.choice(train_data)
        item_uuid = item["uuid"]
        
        # Случайно выбираем тип обратной связи
        feedback_type = random.choice(["click", "favorite", "purchase"])
        
        # Записываем обратную связь
        if feedback_type == "click":
            vector_index.record_feedback(item_uuid, "click")
        elif feedback_type == "favorite":
            vector_index.record_feedback(item_uuid, "favorite")
        else:
            vector_index.record_feedback(item_uuid, "purchase")
    
    # Симуляция персонализированных предпочтений пользователей
    users = [f"user_{i}" for i in range(10)]  # Создаем 10 виртуальных пользователей
    
    for user_id in users:
        # Выбираем 5-10 случайных элементов для каждого пользователя
        user_items = random.sample(train_data, random.randint(5, 10))
        
        for item in user_items:
            # Случайно выбираем тип обратной связи
            feedback_type = random.choice(["click", "favorite", "purchase"])
            
            # Записываем обратную связь
            vector_index.record_feedback(item["uuid"], feedback_type, user_id)
    
    # Создаем связи между элементами
    for i in range(50):  # Создаем 50 связей между элементами
        item1 = random.choice(train_data)
        item2 = random.choice(train_data)
        
        # Записываем связь
        if item1["uuid"] != item2["uuid"]:
            vector_index.record_feedback(item1["uuid"], "relation", value=item2["uuid"])
    
    return vector_index

def evaluate_recommender(vector_index: VectorIndex, 
                         test_data: List[Dict[str, Any]], 
                         metrics: MetricsCollector) -> Dict[str, float]:
    """
    Оценивает качество рекомендательной системы на тестовых данных
    
    Args:
        vector_index: Обученный экземпляр VectorIndex
        test_data: Тестовые данные
        metrics: Коллектор метрик
        
    Returns:
        Словарь с метриками качества
    """
    results = {}
    
    # Метрики для оценки рекомендаций
    relevant_hits = 0
    total_queries = 0
    avg_precision = 0
    category_hits = 0
    style_hits = 0
    tag_hits = 0
    
    print("Тестирование рекомендательной системы...")
    
    # Проходим по тестовым данным
    for query_item in tqdm(test_data):
        total_queries += 1
        
        # Получаем эмбеддинг запроса
        query_vector = np.array(query_item["combined_embedding"])
        
        # Имитируем запрос к рекомендательной системе
        with TimingContext(metrics, "search_time"):
            recommendations = vector_index.search(
                query_vector, 
                top_k=10
            )
        
        # Добавляем результаты поиска в метрики
        metrics.add_search_result(
            query={"id": query_item["uuid"], "name": query_item["name"]},
            results=recommendations,
            timing_ms=metrics.metrics["timing"]["search_time"][-1]
        )
        
        # Проверяем релевантность результатов
        query_categories = set(query_item["categories"])
        query_styles = set(query_item["styles"])
        query_tags = set(query_item["tags"])
        
        # Считаем попадания для каждого результата
        for rank, result in enumerate(recommendations):
            result_categories = set(result.get("categories", []))
            result_styles = set(result.get("styles", []))
            result_tags = set(result.get("tags", []))
            
            # Считаем пересечения
            cat_overlap = len(query_categories.intersection(result_categories))
            style_overlap = len(query_styles.intersection(result_styles))
            tag_overlap = len(query_tags.intersection(result_tags))
            
            # Учитываем попадания
            if cat_overlap > 0:
                category_hits += 1
            if style_overlap > 0:
                style_hits += 1
            if tag_overlap > 0:
                tag_hits += 1
            
            # Считаем общую релевантность
            if cat_overlap > 0 or style_overlap > 0 or tag_overlap > 0:
                relevant_hits += 1
                # Precision@k для текущего ранга
                avg_precision += 1.0 / (rank + 1)
    
    # Рассчитываем итоговые метрики
    results["relevance_ratio"] = relevant_hits / (total_queries * 10) if total_queries > 0 else 0
    results["category_match_ratio"] = category_hits / (total_queries * 10) if total_queries > 0 else 0
    results["style_match_ratio"] = style_hits / (total_queries * 10) if total_queries > 0 else 0
    results["tag_match_ratio"] = tag_hits / (total_queries * 10) if total_queries > 0 else 0
    results["mean_avg_precision"] = avg_precision / total_queries if total_queries > 0 else 0
    
    # Добавляем метрики в коллектор
    for name, value in results.items():
        metrics.add_accuracy_metric(name, value)
    
    return results

def test_personalization(vector_index: VectorIndex, 
                          test_data: List[Dict[str, Any]], 
                          metrics: MetricsCollector) -> Dict[str, float]:
    """
    Тестирует персонализацию рекомендаций
    
    Args:
        vector_index: Обученный экземпляр VectorIndex
        test_data: Тестовые данные
        metrics: Коллектор метрик
        
    Returns:
        Словарь с метриками качества персонализации
    """
    results = {}
    
    print("Тестирование персонализации рекомендаций...")
    
    # Создаем двух тестовых пользователей с разными предпочтениями
    user1_id = "test_user_1"
    user2_id = "test_user_2"
    
    # Разделяем тестовые данные по категориям
    category_groups = {}
    for item in test_data:
        for category in item["categories"]:
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(item)
    
    # Выбираем две разные категории для пользователей
    available_categories = list(category_groups.keys())
    if len(available_categories) >= 2:
        user1_category = available_categories[0]
        user2_category = available_categories[-1]
        
        # Создаем историю взаимодействий для пользователя 1
        for i in range(5):
            if i < len(category_groups[user1_category]):
                item = category_groups[user1_category][i]
                vector_index.record_feedback(item["uuid"], "favorite", user1_id)
        
        # Создаем историю взаимодействий для пользователя 2
        for i in range(5):
            if i < len(category_groups[user2_category]):
                item = category_groups[user2_category][i]
                vector_index.record_feedback(item["uuid"], "favorite", user2_id)
        
        # Выбираем нейтральный тестовый элемент (не из выбранных категорий)
        neutral_items = [item for item in test_data 
                         if user1_category not in item["categories"] and 
                            user2_category not in item["categories"]]
        
        if neutral_items:
            test_item = random.choice(neutral_items)
            query_vector = np.array(test_item["combined_embedding"])
            
            # Получаем рекомендации для пользователя 1
            recommendations_user1 = vector_index.search(
                query_vector, 
                top_k=10,
                user_id=user1_id
            )
            
            # Получаем рекомендации для пользователя 2
            recommendations_user2 = vector_index.search(
                query_vector, 
                top_k=10,
                user_id=user2_id
            )
            
            # Получаем рекомендации без персонализации
            recommendations_anon = vector_index.search(
                query_vector, 
                top_k=10
            )
            
            # Считаем категории в рекомендациях
            user1_cat_count = sum(1 for rec in recommendations_user1 if user1_category in rec.get("categories", []))
            user2_cat_count = sum(1 for rec in recommendations_user2 if user2_category in rec.get("categories", []))
            
            # Расчет показателя различия в рекомендациях
            common_items = len(set(r["uuid"] for r in recommendations_user1) & 
                             set(r["uuid"] for r in recommendations_user2))
            personalization_diff = 1.0 - (common_items / 10) if common_items > 0 else 1.0
            
            # Сохраняем результаты
            results["user1_category_bias"] = user1_cat_count / 10
            results["user2_category_bias"] = user2_cat_count / 10
            results["personalization_difference"] = personalization_diff
            
            # Добавляем метрики в коллектор
            for name, value in results.items():
                metrics.add_accuracy_metric(name, value)
    
    return results

def plot_results(metrics: MetricsCollector, output_dir: str):
    """
    Создает графики для визуализации результатов
    
    Args:
        metrics: Коллектор метрик
        output_dir: Директория для сохранения графиков
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Создаем графики на основе данных метрик
    metrics.visualize_search_performance()
    metrics.visualize_tag_distribution()
    metrics.visualize_category_style_distribution()
    metrics.visualize_timing_metrics()
    
    # Экспортируем все графики в указанную директорию
    metrics.create_summary_visualizations()
    
    # Создаем сводный график метрик качества
    plt.figure(figsize=(12, 8))
    metrics_data = {k: v for k, v in metrics.metrics["accuracy"].items() 
                   if k not in ["user1_category_bias", "user2_category_bias", "personalization_difference"]}
    
    keys = list(metrics_data.keys())
    values = list(metrics_data.values())
    
    plt.bar(keys, values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Метрики качества рекомендательной системы")
    plt.ylabel("Значение")
    plt.tight_layout()
    plt.savefig(output_path / "quality_metrics.png", dpi=300)
    
    # Сохраняем метрики в JSON-файл
    with open(output_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics.metrics, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Обучение и тестирование рекомендательной системы")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_DIR, 
                       help="Директория с набором данных")
    parser.add_argument("--output", type=str, default=RESULTS_DIR,
                       help="Директория для сохранения результатов")
    parser.add_argument("--index_file", type=str, default=INDEX_FILE,
                       help="Имя файла для векторного индекса")
    parser.add_argument("--meta_file", type=str, default=META_FILE,
                       help="Имя файла для метаданных")
    parser.add_argument("--feedback_file", type=str, default=FEEDBACK_FILE,
                       help="Имя файла для данных обратной связи")
    args = parser.parse_args()
    
    # Создаем директорию для результатов
    os.makedirs(args.output, exist_ok=True)
    
    # Полные пути к файлам
    index_file = os.path.join(args.output, args.index_file)
    meta_file = os.path.join(args.output, args.meta_file)
    feedback_file = os.path.join(args.output, args.feedback_file)
    
    # Инициализируем коллектор метрик
    metrics = MetricsCollector("recommender_evaluation")
    
    # Загружаем данные
    train_data, test_data = load_dataset(args.dataset)
    
    # Засекаем время обучения
    start_time = time.time()
    
    # Обучаем рекомендательную систему
    vector_index = train_recommender(train_data, index_file, meta_file, feedback_file)
    
    # Записываем время обучения
    training_time = (time.time() - start_time) * 1000  # в миллисекундах
    metrics.add_timing("training_time", training_time)
    
    # Оцениваем качество рекомендаций
    evaluation_results = evaluate_recommender(vector_index, test_data, metrics)
    
    # Тестируем персонализацию
    personalization_results = test_personalization(vector_index, test_data, metrics)
    
    # Выводим результаты
    print("\nРезультаты оценки:")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nРезультаты персонализации:")
    for metric, value in personalization_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Создаем графики
    plot_results(metrics, args.output)
    
    print(f"\nРезультаты сохранены в директории: {args.output}")

if __name__ == "__main__":
    main() 