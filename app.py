# 1. Library imports
import json
from numpy import int64
import uvicorn
from fastapi import FastAPI, UploadFile
from keras.preprocessing import image
from fastapi.middleware.cors import CORSMiddleware
from mnist_2_test import predict_image
from PIL import Image
import base64
import logging
from io import BytesIO
# 2. Create the app object
app = FastAPI()
logging.basicConfig(level=logging.DEBUG)

# 3. Index route, opens automatically on http://127.0.0.1:8000
origins = [
    "http://localhost",
    "http://localhost:3000",  # Update with your React app's URL
    "http://localhost:3000/upload"
    # Add more origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST","GET"],
    allow_headers=["*"],
)


@app.get('/')
def index():
    print('On Index server')
    return {'message': 'Hello, World'}

@app.get('/{name}')
def get_name(name: str):
    return {'Welcome': f'{name}'}


@app.post('/upload')
async def predict(imageData:dict):

    print('On Server', imageData)
    # Decode base64 image data
    image_data = imageData['image'].split(',')[1]  # Extract the base64-encoded image data
    print('Image Data afet split',image_data)
    image_bytes = base64.b64decode(image_data)

    # Create PIL Image object from the decoded bytes
    image_stream = BytesIO(image_bytes)
    image = Image.open(image_stream)
    
    image.save("img.png") 


    print('Before entering model')
    output = predict_image(image)
    print('Exitted from Model!', type(output))
    return {'output':output.item()}

    

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run("app:app", host="192.168.137.190", port=8000, reload=True)

# uvicorn app:app --reload
