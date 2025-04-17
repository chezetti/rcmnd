#!/usr/bin/env python
"""
Скрипт для тестирования производительности рекомендательной системы
и сравнения различных параметров индекса.
"""
import os
import json
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import sys
import random
from typing import Dict, List, Any, Tuple

# Добавляем путь к корневой директории проекта
sys.path.append('.')

# Импортируем нашу рекомендательную систему
from App.index import VectorIndex, FeedbackSystem
from App.config import FUSION_EMBED_DIM, FAISS_INDEX_TYPE
from tests.metrics_utils import MetricsCollector, TimingContext

# Константы
DEFAULT_DATASET_DIR = "synthetic_dataset"
RESULTS_DIR = "benchmark_results"
INDEX_FILE_TEMPLATE = "recommender_index_{}.faiss"
META_FILE_TEMPLATE = "recommender_meta_{}.json"
FEEDBACK_FILE_TEMPLATE = "recommender_feedback_{}.json"

def load_test_data(dataset_dir: str) -> List[Dict[str, Any]]:
    """
    Загружает тестовые данные для бенчмаркинга
    
    Args:
        dataset_dir: Директория с набором данных
        
    Returns:
        Список тестовых данных
    """
    base_dir = Path(dataset_dir)
    
    # Загружаем тестовые данные
    with open(base_dir / "test" / "metadata.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    print(f"Загружено {len(test_data)} тестовых образцов")
    return test_data

def load_train_data(dataset_dir: str) -> List[Dict[str, Any]]:
    """
    Загружает тренировочные данные для бенчмаркинга
    
    Args:
        dataset_dir: Директория с набором данных
        
    Returns:
        Список тренировочных данных
    """
    base_dir = Path(dataset_dir)
    
    # Загружаем тренировочные данные
    with open(base_dir / "train" / "metadata.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    
    print(f"Загружено {len(train_data)} тренировочных образцов")
    return train_data

def train_recommender(train_data: List[Dict[str, Any]], 
                      index_type: str,
                      index_file: Path, 
                      meta_file: Path, 
                      feedback_file: Path) -> Tuple[VectorIndex, float]:
    """
    Обучает рекомендательную систему с определенным типом индекса
    
    Args:
        train_data: Тренировочные данные
        index_type: Тип индекса Faiss
        index_file: Путь к файлу индекса
        meta_file: Путь к файлу метаданных
        feedback_file: Путь к файлу обратной связи
        
    Returns:
        Кортеж (обученный индекс, время обучения в мс)
    """
    start_time = time.time()
    
    # Временно меняем тип индекса в переменной окружения
    old_index_type = os.environ.get("FAISS_INDEX_TYPE", FAISS_INDEX_TYPE)
    os.environ["FAISS_INDEX_TYPE"] = index_type
    
    # Создаем экземпляр индекса
    vector_index = VectorIndex(
        FUSION_EMBED_DIM, 
        index_file, 
        meta_file, 
        feedback_file
    )
    
    # Восстанавливаем оригинальный тип индекса
    os.environ["FAISS_INDEX_TYPE"] = old_index_type
    
    # Добавляем данные в индекс
    print(f"Добавление данных в индекс {index_type}...")
    for item in tqdm(train_data):
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
    
    # Считаем время обучения
    training_time = (time.time() - start_time) * 1000  # в миллисекундах
    
    return vector_index, training_time

def benchmark_search(vector_index: VectorIndex, 
                    test_data: List[Dict[str, Any]], 
                    metrics: MetricsCollector,
                    label: str,
                    num_queries: int = 100,
                    k_values: List[int] = [1, 5, 10, 20, 50]) -> Dict[str, Dict[int, float]]:
    """
    Тестирует скорость поиска индекса с различными параметрами
    
    Args:
        vector_index: Векторный индекс
        test_data: Тестовые данные
        metrics: Коллектор метрик
        label: Метка для данного бенчмарка
        num_queries: Количество запросов для теста
        k_values: Значения k для тестирования
        
    Returns:
        Словарь с результатами бенчмарка
    """
    results = {
        "search_times": {},
        "accuracy": {}
    }
    
    # Выбираем случайные запросы из тестовых данных
    query_items = random.sample(test_data, min(num_queries, len(test_data)))
    
    # Тестируем скорость поиска для разных значений k
    for k in k_values:
        search_times = []
        relevance_score = 0
        
        print(f"Тестирование поиска с k={k}...")
        
        for query_item in tqdm(query_items):
            query_vector = np.array(query_item["combined_embedding"])
            
            # Замеряем время поиска
            start_time = time.time()
            recommendations = vector_index.search(query_vector, top_k=k)
            search_time = (time.time() - start_time) * 1000  # в миллисекундах
            
            search_times.append(search_time)
            
            # Оцениваем релевантность (упрощенно)
            query_categories = set(query_item["categories"])
            query_styles = set(query_item["styles"])
            query_tags = set(query_item["tags"])
            
            for result in recommendations:
                result_categories = set(result.get("categories", []))
                result_styles = set(result.get("styles", []))
                result_tags = set(result.get("tags", []))
                
                # Считаем пересечения
                cat_overlap = len(query_categories.intersection(result_categories))
                style_overlap = len(query_styles.intersection(result_styles))
                tag_overlap = len(query_tags.intersection(result_tags))
                
                if cat_overlap > 0 or style_overlap > 0 or tag_overlap > 0:
                    relevance_score += 1
        
        # Сохраняем результаты
        avg_search_time = sum(search_times) / len(search_times) if search_times else 0
        avg_relevance = relevance_score / (len(query_items) * k) if query_items else 0
        
        results["search_times"][k] = avg_search_time
        results["accuracy"][k] = avg_relevance
        
        # Добавляем метрики в коллектор
        metrics.add_performance_metric(f"{label}_search_time_k{k}", avg_search_time)
        metrics.add_accuracy_metric(f"{label}_accuracy_k{k}", avg_relevance)
    
    return results

def plot_benchmark_results(results: Dict[str, Dict[str, Dict[int, float]]], 
                          output_dir: str,
                          training_times: Dict[str, float],
                          metrics: MetricsCollector):
    """
    Создает графики результатов бенчмарка
    
    Args:
        results: Словарь с результатами бенчмарка
        output_dir: Директория для сохранения графиков
        training_times: Словарь с временем обучения для каждого индекса
        metrics: Коллектор метрик
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Настройка стиля графиков
    plt.style.use('ggplot')
    sns.set(style="whitegrid")
    
    # 1. График времени обучения
    plt.figure(figsize=(10, 6))
    index_types = list(training_times.keys())
    times = list(training_times.values())
    
    bars = plt.bar(index_types, times)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom')
    
    plt.title('Время обучения для разных типов индексов')
    plt.ylabel('Время (мс)')
    plt.xlabel('Тип индекса')
    plt.tight_layout()
    plt.savefig(output_path / 'training_times.png', dpi=300)
    
    # 2. График времени поиска
    plt.figure(figsize=(12, 8))
    
    for index_type, result in results.items():
        k_values = list(result["search_times"].keys())
        search_times = list(result["search_times"].values())
        plt.plot(k_values, search_times, marker='o', label=index_type)
    
    plt.title('Среднее время поиска при разных k')
    plt.xlabel('Количество результатов (k)')
    plt.ylabel('Время (мс)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path / 'search_times.png', dpi=300)
    
    # 3. График релевантности
    plt.figure(figsize=(12, 8))
    
    for index_type, result in results.items():
        k_values = list(result["accuracy"].keys())
        accuracy = list(result["accuracy"].values())
        plt.plot(k_values, accuracy, marker='o', label=index_type)
    
    plt.title('Средняя релевантность при разных k')
    plt.xlabel('Количество результатов (k)')
    plt.ylabel('Релевантность')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path / 'accuracy.png', dpi=300)
    
    # 4. График соотношения скорость/качество
    plt.figure(figsize=(12, 8))
    
    for index_type, result in results.items():
        search_times = list(result["search_times"].values())
        accuracy = list(result["accuracy"].values())
        plt.scatter(search_times, accuracy, label=index_type, s=100)
    
    plt.title('Соотношение время поиска / релевантность')
    plt.xlabel('Время поиска (мс)')
    plt.ylabel('Релевантность')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path / 'speed_vs_accuracy.png', dpi=300)
    
    # Сохраняем все метрики
    metrics.create_summary_visualizations()
    
    # Сохраняем сырые данные бенчмарка
    benchmark_data = {
        "training_times": training_times,
        "search_results": results
    }
    
    with open(output_path / 'benchmark_data.json', 'w', encoding='utf-8') as f:
        json.dump(benchmark_data, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Бенчмаркинг рекомендательной системы")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_DIR, 
                       help="Директория с набором данных")
    parser.add_argument("--output", type=str, default=RESULTS_DIR,
                       help="Директория для сохранения результатов")
    parser.add_argument("--num_queries", type=int, default=100,
                       help="Количество запросов для тестирования")
    args = parser.parse_args()
    
    # Создаем директорию для результатов
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Загружаем данные
    train_data = load_train_data(args.dataset)
    test_data = load_test_data(args.dataset)
    
    # Инициализируем коллектор метрик
    metrics = MetricsCollector("recommender_benchmark")
    
    # Типы индексов для тестирования
    index_types = [
        "Flat",       # Простой точный поиск
        "IVF100,Flat" # Иерархический поиск с 100 разделами
    ]
    
    # Словарь для хранения результатов
    results = {}
    training_times = {}
    
    # Тестируем каждый тип индекса
    for index_type in index_types:
        print(f"\nТестирование индекса типа {index_type}")
        
        # Формируем пути к файлам
        safe_name = index_type.replace(",", "_").lower()
        index_file = output_dir / INDEX_FILE_TEMPLATE.format(safe_name)
        meta_file = output_dir / META_FILE_TEMPLATE.format(safe_name)
        feedback_file = output_dir / FEEDBACK_FILE_TEMPLATE.format(safe_name)
        
        # Обучаем индекс
        vector_index, training_time = train_recommender(
            train_data,
            index_type,
            index_file,
            meta_file,
            feedback_file
        )
        
        training_times[index_type] = training_time
        
        # Добавляем время обучения в метрики
        metrics.add_timing(f"training_{safe_name}", training_time)
        
        # Тестируем скорость поиска
        benchmark_result = benchmark_search(
            vector_index,
            test_data,
            metrics,
            safe_name,
            num_queries=args.num_queries
        )
        
        # Сохраняем результаты
        results[index_type] = benchmark_result
    
    # Создаем визуализации результатов
    plot_benchmark_results(results, args.output, training_times, metrics)
    
    print(f"\nРезультаты бенчмарка сохранены в директории: {args.output}")

if __name__ == "__main__":
    main() 