import json
import imageio.v2 as imageio
import numpy as np
import napari
import os

# ŚCIEŻKI
plik_png = '/Users/rozam/Downloads/md_dataset/images/0239-1.png'      # plik PNG
plik_json = '/Users/rozam/Downloads/md_dataset/test_annotations.json'     # plik z adnotacjami COCO
image_id_do_sprawdzenia = 1  # ID obrazu w pliku z adnotacjami
image = imageio.imread(plik_png)

# Wczytanie JSON-a
with open(plik_json, 'r') as f:
    coco_data = json.load(f)

# Szukanie adnotacji dla wybranego obrazu
shapes = []
for ann in coco_data["annotations"]:
    if ann["image_id"] == image_id_do_sprawdzenia:
        for seg in ann["segmentation"]:
            coords = np.array(seg).reshape(-1, 2)  # (x, y)
            coords = coords[:, [1, 0]]  # zamiana na (y, x) dla Napari
            shapes.append(coords)

# Napari
viewer = napari.Viewer()
viewer.add_image(image, name='Obraz')

if shapes:
    viewer.add_shapes(
        shapes,
        shape_type='polygon',
        edge_color='cyan',
        face_color='transparent',
        name='Adnotacje'
    )

napari.run()
