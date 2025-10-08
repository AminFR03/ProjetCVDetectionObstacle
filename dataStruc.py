import os
import zipfile
import requests
from pathlib import Path

# URL du jeu de données (par exemple, depuis Kaggle ou Mendeley)
dataset_url = "https://www.kaggle.com/datasets/dataclusterlabs/potholes-or-cracks-on-road-image-dataset?utm_source=chatgpt.com"  # Remplacez par l'URL du jeu de données réel

# Répertoire de destination
dataset_dir = Path("C:/Users/aminf/Desktop/ProjetCVgi2/ProjetCVDetectionObstacle")
images_dir = dataset_dir / "images"
labels_dir = dataset_dir / "labels"

# Créer la structure de répertoires
for subdir in ["train", "val"]:
    (images_dir / subdir).mkdir(parents=True, exist_ok=True)
    (labels_dir / subdir).mkdir(parents=True, exist_ok=True)

# Télécharger le jeu de données (remplacez cette partie par le téléchargement réel)
response = requests.get(dataset_url)
zip_path = dataset_dir / "dataset.zip"
with open(zip_path, "wb") as f:
    f.write(response.content)

# Extraire le fichier ZIP
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(dataset_dir)

# Déplacer les images et annotations dans les répertoires appropriés
# (Adaptez cette partie en fonction de la structure du jeu de données téléchargé)
for img_file in dataset_dir.glob("*.jpg"):
    img_name = img_file.stem
    if "train" in img_name:
        img_file.rename(images_dir / "train" / img_file.name)
        label_file = dataset_dir / f"{img_name}.txt"
        label_file.rename(labels_dir / "train" / label_file.name)
    elif "val" in img_name:
        img_file.rename(images_dir / "val" / img_file.name)
        label_file = dataset_dir / f"{img_name}.txt"
        label_file.rename(labels_dir / "val" / label_file.name)

# Créer le fichier data.yaml
data_yaml = dataset_dir / "data.yaml"
with open(data_yaml, "w") as f:
    f.write("""
train: images/train
val: images/val

nc: 1
names: ['speedbump']
""")

print(f"Jeu de données prêt à l'emploi dans {dataset_dir}")