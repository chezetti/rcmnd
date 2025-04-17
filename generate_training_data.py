#!/usr/bin/env python
"""
Скрипт для генерации синтетического набора данных из 1000 NFT-изображений
и их метаданных для обучения и тестирования рекомендательной системы.
"""
import os
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
import sys

# Добавляем путь к корневой директории проекта, чтобы импортировать модули
sys.path.append('.')

# Импортируем функции для генерации тестовых данных
from tests.test_data_generator import generate_random_nft_image, generate_nft_metadata
from App.encoders import AdvancedEncoder

def generate_dataset(count: int, output_dir: str, split_ratio: float = 0.8):
    """
    Генерирует набор данных из NFT изображений и метаданных
    
    Args:
        count: Количество изображений для создания
        output_dir: Директория для сохранения набора данных
        split_ratio: Соотношение данных для обучения/тестирования (по умолчанию 80/20)
    """
    # Создаем директории для тренировочных и тестовых данных
    base_dir = Path(output_dir)
    train_dir = base_dir / "train"
    test_dir = base_dir / "test"
    
    images_dir = base_dir / "images"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Инициализируем энкодер для извлечения эмбеддингов
    print("Инициализация энкодера...")
    encoder = AdvancedEncoder()
    
    # Списки для хранения метаданных
    all_metadata = []
    train_metadata = []
    test_metadata = []
    
    # Вычисляем количество тренировочных и тестовых образцов
    train_count = int(count * split_ratio)
    test_count = count - train_count
    
    print(f"Генерация {count} NFT изображений ({train_count} для обучения, {test_count} для тестирования)...")
    
    # Генерируем изображения и метаданные
    for i in tqdm(range(count)):
        seed = i + 1000  # Используем отличные от тестовых сиды
        
        # Генерируем изображение
        image_filename = f"nft_{seed:04d}.png"
        image_path = images_dir / image_filename
        
        try:
            img_data = generate_random_nft_image(seed, str(image_path))
            
            # Генерируем метаданные
            metadata = generate_nft_metadata(seed, img_data)
            
            # Добавляем путь к изображению
            metadata["image_path"] = str(image_path)
            
            # Получаем эмбеддинги
            image_embedding = encoder.encode_image(img_data).cpu().numpy()
            text_embedding = encoder.encode_text(metadata["description"]).cpu().numpy()
            combined_embedding = encoder.encode(img_data, metadata["description"])
            
            # Сохраняем эмбеддинги
            metadata["image_embedding"] = image_embedding.tolist()
            metadata["text_embedding"] = text_embedding.tolist()
            metadata["combined_embedding"] = combined_embedding.tolist()
            
            # Добавляем метаданные в соответствующие списки
            all_metadata.append(metadata)
            
        except Exception as e:
            print(f"Ошибка при генерации NFT {seed}: {e}")
            continue
    
    # Перемешиваем данные и разделяем на тренировочные и тестовые
    random.shuffle(all_metadata)
    train_metadata = all_metadata[:train_count]
    test_metadata = all_metadata[train_count:]
    
    # Сохраняем метаданные
    with open(train_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(train_metadata, f, ensure_ascii=False, indent=2)
    
    with open(test_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(test_metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Набор данных создан: всего {len(all_metadata)} NFT")
    print(f"Тренировочные данные: {len(train_metadata)}")
    print(f"Тестовые данные: {len(test_metadata)}")
    
    return train_metadata, test_metadata

def main():
    parser = argparse.ArgumentParser(description="Генерация набора данных из NFT изображений")
    parser.add_argument("--count", type=int, default=1000, help="Количество изображений для создания")
    parser.add_argument("--output", type=str, default="synthetic_dataset", help="Директория для сохранения набора данных")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Соотношение данных для обучения/тестирования")
    args = parser.parse_args()
    
    generate_dataset(args.count, args.output, args.split_ratio)

if __name__ == "__main__":
    main() 