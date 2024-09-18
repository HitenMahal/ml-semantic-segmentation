from fastapi import FastAPI, UploadFile, File, Response
from transformers import pipeline
from PIL import Image
import numpy as np
import cv2
import pickle
import json
import base64
import time

size = (2000,2000)

app = FastAPI()
pipe = pipeline("image-segmentation", model="Thalirajesh/Aerial-Drone-Image-Segmentation")
# for i in pipe.model.parameters():
#     print(i.shape)

@app.post("/analyze")
def analyze(file: UploadFile = File(...)):
    contents = file.file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, size)

    print(img.shape)
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).show()
    result = pipe(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    print(len(result))
    print(result)

    output = {}
    for i in result:
        output[i['label']] = i['mask']

    print(output.keys())

    imdata = pickle.dumps(output)
    jstr = json.dumps(base64.b64encode(imdata).decode('ascii'))
    return Response(content=jstr, media_type="image/png")