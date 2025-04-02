import os
import shutil
import random

def split_data(base_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    dest_dir = base_dir  # Nowe foldery powstaną w tym samym miejscu
    
    for split in ['Training', 'Validation', 'Test']:
        os.makedirs(os.path.join(dest_dir, split), exist_ok=True)
        for i in range(10):
            os.makedirs(os.path.join(dest_dir, split, str(i)), exist_ok=True)
    
    for class_num in range(10):
        class_path = os.path.join(base_dir, str(class_num))
        if not os.path.exists(class_path):
            print(f"Folder {class_path} nie istnieje, pomijam...")
            continue
        
        images = os.listdir(class_path)
        random.shuffle(images)
        
        train_split = int(len(images) * train_ratio)
        val_split = int(len(images) * val_ratio)
        
        train_files = images[:train_split]
        val_files = images[train_split:train_split + val_split]
        test_files = images[train_split + val_split:]
        
        for file in train_files:
            shutil.copy(os.path.join(class_path, file), os.path.join(dest_dir, 'Training', str(class_num), file))
        for file in val_files:
            shutil.copy(os.path.join(class_path, file), os.path.join(dest_dir, 'Validation', str(class_num), file))
        for file in test_files:
            shutil.copy(os.path.join(class_path, file), os.path.join(dest_dir, 'Test', str(class_num), file))
    
    print("Podział danych zakończony!")

# Przykład użycia
current_directory = os.path.dirname(os.path.abspath(__file__))  # Pobiera ścieżkę do folderu, w którym znajduje się skrypt
split_data(current_directory)
