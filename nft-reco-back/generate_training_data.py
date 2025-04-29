#!/usr/bin/env python
"""
Скрипт для генерации синтетического набора данных из NFT-изображений
и их метаданных для обучения и тестирования рекомендательной системы.
Оптимизирован для максимальной производительности.
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
import concurrent.futures
import multiprocessing
import traceback
import time
import signal
import gc
import threading
import queue
import psutil

# Добавляем путь к корневой директории проекта, чтобы импортировать модули
sys.path.append('.')

# Импортируем функции для генерации тестовых данных
from tests.test_data_generator import generate_random_nft_image, generate_nft_metadata
from App.encoders import AdvancedEncoder

# Глобальные кэши
_ENCODER_CACHE = {}
_BATCH_SIZE = 10  # Размер пакета для одновременной обработки
_IMAGE_CACHE = {}  # Кэш для изображений
_RESULT_QUEUE = None
_STOP_EVENT = None

def set_process_priority():
    """Устанавливает приоритет процесса на высокий"""
    try:
        process = psutil.Process(os.getpid())
        if sys.platform == 'win32':
            process.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            process.nice(-10)  # Более высокий приоритет для Unix-подобных систем
    except Exception:
        pass  # Игнорируем ошибки при настройке приоритета

def init_worker():
    """Инициализация рабочего процесса"""
    # Устанавливаем seed для каждого процесса
    worker_id = multiprocessing.current_process().name
    random.seed(int(time.time()) + hash(worker_id) % 10000)
    
    # Отключаем многопоточность внутри процессов для PyTorch
    torch.set_num_threads(1)
    
    # Устанавливаем высокий приоритет процессов
    set_process_priority()
    
    # Принудительно используем CPU для предотвращения конфликтов GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Предварительно инициализируем энкодер для процесса
    pid = os.getpid()
    if pid not in _ENCODER_CACHE:
        _ENCODER_CACHE[pid] = AdvancedEncoder()

def prefetch_data(seeds, images_dir, result_queue, stop_event):
    """Предварительная загрузка данных в отдельном потоке"""
    for seed in seeds:
        if stop_event.is_set():
            break
        
        image_filename = f"nft_{seed:04d}.png"
        image_path = Path(images_dir) / image_filename
        
        # Если изображение уже существует, загружаем его заранее
        if image_path.exists():
            try:
                with open(image_path, "rb") as f:
                    img_data = f.read()
                result_queue.put((seed, img_data))
            except Exception:
                pass  # Игнорируем ошибки при предзагрузке

def generate_single_nft(args):
    """
    Генерирует одно NFT изображение и его метаданные
    
    Args:
        args: Кортеж (seed, images_dir)
        
    Returns:
        Метаданные NFT или None в случае ошибки
    """
    global _RESULT_QUEUE, _STOP_EVENT
    
    seed, images_dir = args
    pid = os.getpid()
    
    try:
        # Генерируем изображение
        image_filename = f"nft_{seed:04d}.png"
        image_path = Path(images_dir) / image_filename
        
        # Проверяем, есть ли изображение в кэше предзагрузки
        img_data = None
        if _RESULT_QUEUE is not None:
            try:
                # Проверяем очередь без блокировки
                while not _RESULT_QUEUE.empty():
                    cached_seed, cached_data = _RESULT_QUEUE.get_nowait()
                    if cached_seed == seed:
                        img_data = cached_data
                        break
            except queue.Empty:
                pass
        
        # Если изображения нет в кэше, генерируем или загружаем его
        if img_data is None:
            if not image_path.exists():
                img_data = generate_random_nft_image(seed, str(image_path))
            else:
                with open(image_path, "rb") as f:
                    img_data = f.read()
        
        # Генерируем метаданные
        metadata = generate_nft_metadata(seed, img_data)
        
        # Добавляем путь к изображению
        metadata["image_path"] = str(image_path)
        
        # Используем кэшированный энкодер для процесса или создаем новый
        if pid not in _ENCODER_CACHE:
            _ENCODER_CACHE[pid] = AdvancedEncoder()
        
        local_encoder = _ENCODER_CACHE[pid]
        
        # Получаем эмбеддинги
        image_embedding = local_encoder.encode_image(img_data).cpu().numpy()
        text_embedding = local_encoder.encode_text(metadata["description"]).cpu().numpy()
        combined_embedding = local_encoder.encode(img_data, metadata["description"])
        
        # Сохраняем эмбеддинги
        metadata["image_embedding"] = image_embedding.tolist()
        metadata["text_embedding"] = text_embedding.tolist()
        metadata["combined_embedding"] = combined_embedding.tolist()
        
        # Очищаем память
        del img_data
        
        return metadata
    
    except Exception as e:
        print(f"Ошибка при генерации NFT {seed}: {e}")
        return None

def process_batch(batch, executor):
    """Обрабатывает пакет заданий и возвращает результаты"""
    futures = [executor.submit(generate_single_nft, args) for args in batch]
    results = []
    
    for future in concurrent.futures.as_completed(futures):
        try:
            metadata = future.result()
            if metadata is not None:
                results.append(metadata)
        except Exception as e:
            print(f"Ошибка при выполнении задания: {e}")
    
    return results

def generate_dataset(count: int, output_dir: str, split_ratio: float = 0.8, workers: int | None = None):
    """
    Генерирует набор данных из NFT изображений и метаданных
    
    Args:
        count: Количество изображений для создания
        output_dir: Директория для сохранения набора данных
        split_ratio: Соотношение данных для обучения/тестирования (по умолчанию 80/20)
        workers: Количество рабочих процессов
    """
    global _RESULT_QUEUE, _STOP_EVENT
    
    # Создаем директории для тренировочных и тестовых данных
    base_dir = Path(output_dir)
    train_dir = base_dir / "train"
    test_dir = base_dir / "test"
    
    images_dir = base_dir / "images"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Определяем оптимальное количество рабочих процессов
    available_cpus = multiprocessing.cpu_count()
    if workers is None:
        # По умолчанию используем 75% доступных ядер для лучшей загрузки CPU
        workers = max(1, int(available_cpus * 0.75))
    else:
        # Ограничиваем максимальное количество процессов доступными ядрами
        workers = min(workers, available_cpus)
    
    print(f"\nПараллельная генерация {count} NFT изображений используя {workers} процессов...")
    print(f"Размер пакетной обработки: {_BATCH_SIZE}")
    
    # Подготавливаем параметры для каждого задания
    seeds = [i + 1000 for i in range(count)]
    tasks = [(seed, images_dir) for seed in seeds]
    
    # Создаем очередь для результатов предзагрузки
    _RESULT_QUEUE = multiprocessing.Manager().Queue()
    _STOP_EVENT = multiprocessing.Manager().Event()
    
    # Запускаем поток для предзагрузки изображений
    prefetch_thread = threading.Thread(
        target=prefetch_data, 
        args=(seeds, images_dir, _RESULT_QUEUE, _STOP_EVENT)
    )
    prefetch_thread.daemon = True
    prefetch_thread.start()
    
    # Списки для хранения метаданных
    all_metadata = []
    
    # Блокируем сигналы CTRL+C во время выполнения
    original_sigint_handler = None
    if hasattr(signal, 'SIGINT'):
        try:
            original_sigint_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, lambda sig, frame: None)
        except:
            pass
    
    start_time = time.time()
    completed = 0
    
    try:
        # Создаем ProcessPoolExecutor с настраиваемым числом рабочих процессов
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, 
            initializer=init_worker
        ) as executor:
            # Разбиваем задания на пакеты для оптимизации памяти и параллелизма
            for i in range(0, len(tasks), _BATCH_SIZE):
                batch = tasks[i:i+_BATCH_SIZE]
                batch_results = process_batch(batch, executor)
                all_metadata.extend(batch_results)
                
                # Обновляем счетчик завершенных заданий
                completed += len(batch_results)
                
                # Вычисляем прогресс и оценку времени
                elapsed = time.time() - start_time
                progress = completed / count
                estimated_total = elapsed / progress if progress > 0 else 0
                remaining = estimated_total - elapsed
                
                # Отображаем прогресс
                print(f"\rПрогресс: {completed}/{count} ({progress:.1%}) | "
                      f"Прошло: {elapsed:.1f}с | Осталось: {remaining:.1f}с | "
                      f"Скорость: {completed/elapsed:.1f} NFT/с", end="", flush=True)
                
                # Запускаем сборку мусора после каждой партии
                gc.collect()
    
    finally:
        # Останавливаем поток предзагрузки
        _STOP_EVENT.set()
        prefetch_thread.join(timeout=1.0)
        
        # Восстанавливаем обработчик сигналов
        if original_sigint_handler:
            try:
                signal.signal(signal.SIGINT, original_sigint_handler)
            except:
                pass
        
        print()  # Новая строка после индикатора прогресса
    
    # Вычисляем количество тренировочных и тестовых образцов
    train_count = int(len(all_metadata) * split_ratio)
    
    # Перемешиваем данные и разделяем на тренировочные и тестовые
    print("\nПерешивание и разделение данных...")
    random.shuffle(all_metadata)
    train_metadata = all_metadata[:train_count]
    test_metadata = all_metadata[train_count:]
    
    # Сохраняем метаданные
    print("Сохранение метаданных...")
    
    # Оптимизированное сохранение в JSON (быстрее для больших файлов)
    with open(train_dir / "metadata.json", "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, metadata in enumerate(train_metadata):
            json_str = json.dumps(metadata, ensure_ascii=False)
            f.write(json_str)
            if i < len(train_metadata) - 1:
                f.write(",\n")
        f.write("\n]")
    
    with open(test_dir / "metadata.json", "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, metadata in enumerate(test_metadata):
            json_str = json.dumps(metadata, ensure_ascii=False)
            f.write(json_str)
            if i < len(test_metadata) - 1:
                f.write(",\n")
        f.write("\n]")
    
    print(f"\nНабор данных создан: всего {len(all_metadata)} NFT из {count} запрошенных")
    print(f"Тренировочные данные: {len(train_metadata)}")
    print(f"Тестовые данные: {len(test_metadata)}")
    
    # Вычисляем итоговую производительность
    elapsed_time = time.time() - start_time
    nft_per_second = len(all_metadata) / elapsed_time if elapsed_time > 0 else 0
    
    print(f"Время выполнения: {elapsed_time:.2f} секунд ({elapsed_time/60:.2f} минут)")
    print(f"Производительность: {nft_per_second:.2f} NFT/с")
    
    return train_metadata, test_metadata

def main():
    parser = argparse.ArgumentParser(description="Генерация набора данных из NFT изображений")
    parser.add_argument("--count", type=int, default=1000, help="Количество изображений для создания")
    parser.add_argument("--output", type=str, default="synthetic_dataset", help="Директория для сохранения набора данных")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Соотношение данных для обучения/тестирования")
    parser.add_argument("--workers", type=int, default=None, help="Количество рабочих процессов (по умолчанию: 75% ядер CPU)")
    parser.add_argument("--batch_size", type=int, default=10, help="Размер пакета для обработки")
    args = parser.parse_args()
    
    # Устанавливаем глобальный размер пакета
    global _BATCH_SIZE
    _BATCH_SIZE = args.batch_size
    
    # Устанавливаем приоритет основного процесса
    set_process_priority()
    
    # Выводим информацию о системе
    cpu_count = multiprocessing.cpu_count()
    total_memory = psutil.virtual_memory().total / (1024**3)  # Гигабайты
    
    print(f"=== Системная информация ===")
    print(f"Доступно ядер CPU: {cpu_count}")
    print(f"Оперативная память: {total_memory:.1f} ГБ")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Доступно GPU: {torch.cuda.device_count()}")
        print(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} ГБ")
    else:
        print("GPU не доступно, будет использоваться CPU")
    
    # Запускаем генерацию набора данных
    start_time = time.time()
    generate_dataset(args.count, args.output, args.split_ratio, args.workers)
    elapsed_time = time.time() - start_time
    
    print(f"\n=== Итоговые результаты ===")
    print(f"Общее время выполнения: {elapsed_time:.2f} секунд ({elapsed_time/60:.2f} минут)")
    print(f"Среднее время на один NFT: {elapsed_time/args.count*1000:.2f} мс")

if __name__ == "__main__":
    # Устанавливаем приоритет планировщика Python для улучшения производительности
    try:
        # Эти функции доступны только в Unix-системах
        if sys.platform != 'win32' and hasattr(os, "sched_setscheduler"):
            import sched
            os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
    except (AttributeError, ImportError, PermissionError):
        pass
    
    main() 