import os
import shutil
import random

def split_data(base_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, max_classes=46):
    dest_dir = base_dir  # Nowe foldery powstaną w tym samym miejscu

    for class_num in range(max_classes):
        class_path = os.path.join(base_dir, str(class_num))
        if not os.path.isdir(class_path):
            print(f"⚠️ Folder {class_path} nie istnieje, pomijam...")
            continue

        images = [
            f for f in os.listdir(class_path)
            if os.path.isfile(os.path.join(class_path, f)) and f.lower().endswith(".ppm")
        ]

        if not images:
            print(f"⚠️ Brak plików .ppm w folderze {class_path}, pomijam...")
            continue

        random.shuffle(images)

        train_split = int(len(images) * train_ratio)
        val_split = int(len(images) * val_ratio)

        train_files = images[:train_split]
        val_files = images[train_split:train_split + val_split]
        test_files = images[train_split + val_split:]

        for split_name, file_list in zip(['Training', 'Validation', 'Test'], [train_files, val_files, test_files]):
            split_path = os.path.join(dest_dir, split_name, str(class_num))
            os.makedirs(split_path, exist_ok=True)
            for file in file_list:
                shutil.copy(os.path.join(class_path, file), os.path.join(split_path, file))

    print("✅ Podział danych zakończony!")

# Przykład użycia
if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))  # lub os.getcwd() jeśli uruchamiasz z terminala
    split_data(current_directory)
