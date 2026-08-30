import shutil
from pathlib import Path

def gather_photos(source_folder, destination_folder):
    src = Path(source_folder)
    dest = Path(destination_folder)

    # Проверяем, существует ли исходная папка
    if not src.exists():
        print(f"⚠️ Папка '{source_folder}' не найдена, пропускаем...")
        return

    # Создаем папку назначения, если её еще нет
    dest.mkdir(parents=True, exist_ok=True)

    # Расширения файлов, которые считаем фотографиями
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
    
    counter = 1
    
    print(f"\n📁 Начинаем копирование: из '{source_folder}' в '{destination_folder}'")

    # rglob('*') рекурсивно ищет все файлы во всех подпапках
    for file_path in src.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            new_file_name = file_path.name
            dest_path = dest / new_file_name

            # Если файл с таким именем уже есть, добавляем к имени счетчик
            while dest_path.exists():
                new_file_name = f"{file_path.stem}_{counter}{file_path.suffix}"
                dest_path = dest / new_file_name
                counter += 1

            # Копируем файл
            shutil.copy2(file_path, dest_path)
            print(f"  Скопировано: {file_path.name} -> {dest_path.name}")

if __name__ == '__main__':
    # 1. Шаг первый: копируем фото из wiki в real
    gather_photos('./data/processed/wiki', './data/processed2/real')
    
    # 2. Шаг второй: копируем фото из трех папок в общую папку fake
    fake_sources = ['./data/processed/inpainting', './data/processed/insight', './data/processed/text2img']
    
    for folder in fake_sources:
        gather_photos(folder, './data/processed2/fake')
        
    print("\n✅ Все задачи успешно завершены!")