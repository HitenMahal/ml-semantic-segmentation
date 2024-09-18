import json
import requests
import cv2
import pickle
import base64
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

url = 'http://127.0.0.1:8000'

size = (2000,2000,3)

type_to_color = {
    "gravel": [131, 72, 13],
    "dirt": [74, 46, 7],
    "paved-area": [15, 3, 1],
    "rocks": [168, 84, 29],
    "water": [3, 157, 252],
    
    "pool": [3, 227, 252],
    "vegetation": [15, 163, 44],
    "tree": [14, 48, 20],
    "grass": [30, 179, 56],
    "roof": [71, 32, 5],
    
    "wall": [99, 21, 4],
    "door": [199, 130, 52],
    "obstacle":  [214, 51, 15],
    "dog": [156, 154, 53],
    "person": [245, 241, 5],

    "fence": [112, 15, 101],
    "fence-pole": [41, 11, 37],
    "car": [148, 81, 237],
    "bicycle": [46, 12, 92],
    "window": [234, 235, 230],
}

def main():
    filename = "drone-park-crop.jpg"
    files = {"file": (filename, open(filename, "rb"))}

    image = cv2.imread(filename)
    image = cv2.resize(image, size[:2])
    
    response = requests.post(f"{url}/analyze", files=files)

    resp = {}
    load = json.loads(response.content)

    load = pickle.loads(base64.b64decode(load))
    print(load.keys())

    mask = np.zeros(size, dtype='uint8')
    for k, v in load.items():
        print(k,v)
        resp[k] = cv2.cvtColor(np.array(v), cv2.COLOR_RGB2BGR)
        mask = combine_masks(mask, resp[k], type_to_color[k])

    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axarr[1].imshow(mask)
    f.legend(handles=[
        mpatches.Patch(color=[v/255 for v in type_to_color[k]], label=k) for k in load.keys()
    ], loc='outside center right')
    f.set_size_inches(12,6)
    plt.show()

def combine_masks(image, mask, mask_color):
    color = np.full(size, mask_color, dtype='uint8')
    return np.where(mask, color, image)

def tint_image(image, mask, tint_color):
    color = np.full(size, tint_color, dtype='uint8')
    masked_img = np.where(mask, color, image)
    return cv2.addWeighted(image, 0.8, masked_img, 0.2, 0)

if __name__ == "__main__":
    main()