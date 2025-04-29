#!/usr/bin/env python
"""
Скрипт для автоматического обновления рекомендательной системы NFT.
Выполняет полный цикл: генерация данных, обучение модели и обновление базы.

Использование:
python update_model.py --count 1000 --backup --preserve_feedback
"""
import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np

def create_backup(backup_dir):
    """Создает резервные копии файлов базы данных"""
    print("Создание резервной копии базы данных...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = Path(backup_dir) / f"backup_{timestamp}"
    backup_path.mkdir(parents=True, exist_ok=True)
    
    # Проверяем наличие файлов
    persist_dir = Path("persist")
    if persist_dir.exists():
        files_to_backup = ["faiss.index", "metadata.json", "feedback.json"]
        for file in files_to_backup:
            src_file = persist_dir / file
            if src_file.exists():
                shutil.copy2(src_file, backup_path / file)
    
    print(f"Резервная копия создана: {backup_path}")
    return backup_path

def merge_feedback(new_feedback_path, existing_feedback_path):
    """Объединяет данные обратной связи, сохраняя предпочтения пользователей"""
    print("Объединение данных обратной связи...")
    
    # Загрузить новые данные обратной связи
    with open(new_feedback_path, "r", encoding="utf-8") as f:
        new_feedback = json.load(f)

    # Загрузить существующие данные обратной связи
    with open(existing_feedback_path, "r", encoding="utf-8") as f:
        existing_feedback = json.load(f)

    # Объединить данные обратной связи, сохраняя предпочтения пользователей
    keys_to_preserve = [
        "user_preferences", "clicks", "favorites", "purchases", 
        "views", "favorites_users", "purchases_users", "item_similarity_boost"
    ]
    
    for key in keys_to_preserve:
        if key in existing_feedback:
            if key not in new_feedback:
                new_feedback[key] = {}
            
            # Для словарей пользователь->предпочтения объединяем их
            if key in ["user_preferences"]:
                for user_id, preferences in existing_feedback[key].items():
                    if user_id not in new_feedback[key]:
                        new_feedback[key][user_id] = preferences
            # Для простых счетчиков сохраняем максимальные значения
            elif isinstance(existing_feedback[key], dict):
                for item_id, value in existing_feedback[key].items():
                    if item_id not in new_feedback[key]:
                        new_feedback[key][item_id] = value

    return new_feedback

def run_command(cmd, description):
    """Запускает команду и выводит ее результат"""
    print(f"\n{'='*80}\n{description}\n{'='*80}")
    
    # Запускаем команду без перехвата вывода, чтобы видеть его в реальном времени
    result = subprocess.run(cmd, shell=True, stdout=None, stderr=None)
    
    if result.returncode != 0:
        print(f"Ошибка при выполнении команды.")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Обновление рекомендательной системы NFT")
    
    # Аргументы для генерации данных
    parser.add_argument("--count", type=int, default=1000, help="Количество синтетических NFT (по умолчанию: 1000)")
    parser.add_argument("--dataset", type=str, default="synthetic_dataset", help="Директория для сохранения сгенерированных данных")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Соотношение тренировочных и тестовых данных")
    parser.add_argument("--workers", type=int, default=None, help="Количество рабочих процессов для генерации (по умолчанию: CPU_COUNT - 1)")
    parser.add_argument("--batch_size", type=int, default=10, help="Размер пакета для обработки данных (кол-во NFT в пакете)")
    
    # Аргументы для обучения модели
    parser.add_argument("--output", type=str, default="recommender_results", help="Директория для сохранения результатов обучения")
    parser.add_argument("--index_file", type=str, default="recommender_index.faiss", help="Имя файла для векторного индекса")
    parser.add_argument("--meta_file", type=str, default="recommender_meta.json", help="Имя файла для метаданных")
    parser.add_argument("--feedback_file", type=str, default="recommender_feedback.json", help="Имя файла для данных обратной связи")
    
    # Дополнительные опции
    parser.add_argument("--backup", action="store_true", help="Создать резервную копию текущей базы данных")
    parser.add_argument("--backup_dir", type=str, default="backups", help="Директория для резервных копий")
    parser.add_argument("--preserve_feedback", action="store_true", help="Сохранить существующие данные обратной связи")
    parser.add_argument("--skip_generation", action="store_true", help="Пропустить генерацию синтетических данных")
    parser.add_argument("--skip_training", action="store_true", help="Пропустить обучение модели")
    
    args = parser.parse_args()
    
    # Создаем директорию persist, если она не существует
    os.makedirs("persist", exist_ok=True)
    
    # Создаем резервную копию базы данных, если указан флаг --backup
    if args.backup:
        backup_path = create_backup(args.backup_dir)
    
    # Шаг 1: Генерация синтетических данных
    if not args.skip_generation:
        workers_arg = f"--workers {args.workers}" if args.workers is not None else ""
        gen_cmd = f"python generate_training_data.py --count {args.count} --output {args.dataset} --split_ratio {args.split_ratio} {workers_arg} --batch_size {args.batch_size}"
        if not run_command(gen_cmd, "Генерация синтетических данных"):
            print("Ошибка при генерации данных. Завершение работы.")
            sys.exit(1)
    
    # Шаг 2: Обучение модели
    if not args.skip_training:
        train_cmd = (f"python train_recommender.py --dataset {args.dataset} --output {args.output} "
                    f"--index_file {args.index_file} --meta_file {args.meta_file} --feedback_file {args.feedback_file}")
        if not run_command(train_cmd, "Обучение рекомендательной системы"):
            print("Ошибка при обучении модели. Завершение работы.")
            sys.exit(1)
    
    # Шаг 3: Обновление базы данных
    print("\n" + "="*80 + "\nОбновление базы данных\n" + "="*80)
    
    # Пути к файлам
    source_index = Path(args.output) / args.index_file
    source_meta = Path(args.output) / args.meta_file
    source_feedback = Path(args.output) / args.feedback_file
    
    target_index = Path("persist/faiss.index")
    target_meta = Path("persist/metadata.json")
    target_feedback = Path("persist/feedback.json")
    
    # Проверяем существование файлов источника
    # Если пропущено обучение и файлы не существуют, создаем пустые файлы
    if args.skip_training:
        # Убедимся, что директория результатов существует
        os.makedirs(args.output, exist_ok=True)
        
        # Проверяем каждый файл и создаем его, если он не существует
        if not source_index.exists():
            print(f"Создание пустого файла индекса: {source_index}")
            # Создаем индекс с нулевыми векторами
            import faiss
            from App.config import FAISS_INDEX_TYPE, FUSION_EMBED_DIM
            
            # Создаем индекс в соответствии с типом из конфигурации
            if FAISS_INDEX_TYPE == "Flat":
                empty_index = faiss.IndexFlatIP(FUSION_EMBED_DIM)
            elif FAISS_INDEX_TYPE == "IVF100,Flat":
                quantizer = faiss.IndexFlatIP(FUSION_EMBED_DIM)
                empty_index = faiss.IndexIVFFlat(quantizer, FUSION_EMBED_DIM, 100, faiss.METRIC_INNER_PRODUCT)
                # Для IVF индекса нужен хотя бы один вектор для обучения
                random_vectors = np.random.random((1, FUSION_EMBED_DIM)).astype('float32')
                empty_index.train(random_vectors)
            elif FAISS_INDEX_TYPE == "HNSW":
                empty_index = faiss.IndexHNSWFlat(FUSION_EMBED_DIM, 32, faiss.METRIC_INNER_PRODUCT)
            elif FAISS_INDEX_TYPE == "PQ":
                empty_index = faiss.IndexPQ(FUSION_EMBED_DIM, 16, 8, faiss.METRIC_INNER_PRODUCT)
            else:
                # По умолчанию используем плоский индекс
                empty_index = faiss.IndexFlatIP(FUSION_EMBED_DIM)
            
            faiss.write_index(empty_index, str(source_index))
        
        if not source_meta.exists():
            print(f"Создание пустого файла метаданных: {source_meta}")
            with open(source_meta, "w", encoding="utf-8") as f:
                json.dump({}, f)
        
        if not source_feedback.exists():
            print(f"Создание пустого файла обратной связи: {source_feedback}")
            empty_feedback = {
                "clicks": {},
                "favorites": {},
                "purchases": {},
                "views": {},
                "favorites_users": {},
                "purchases_users": {},
                "user_preferences": {},
                "item_similarity_boost": {}
            }
            with open(source_feedback, "w", encoding="utf-8") as f:
                json.dump(empty_feedback, f, ensure_ascii=False, indent=2)
    
    # Копирование файлов
    print(f"Копирование индекса из {source_index} в {target_index}")
    shutil.copy2(source_index, target_index)
    
    print(f"Копирование метаданных из {source_meta} в {target_meta}")
    shutil.copy2(source_meta, target_meta)
    
    # Обработка данных обратной связи
    if args.preserve_feedback and target_feedback.exists():
        print("Сохранение существующих данных обратной связи...")
        new_feedback = merge_feedback(source_feedback, target_feedback)
        
        # Сохранение объединенных данных
        with open(target_feedback, "w", encoding="utf-8") as f:
            json.dump(new_feedback, f, ensure_ascii=False, indent=2)
    else:
        print(f"Копирование данных обратной связи из {source_feedback} в {target_feedback}")
        shutil.copy2(source_feedback, target_feedback)
    
    print("\n" + "="*80)
    print("Обновление базы данных завершено успешно!")
    print("="*80)
    print("\nДля запуска API выполните команду: uvicorn main:app --reload")

if __name__ == "__main__":
    main()