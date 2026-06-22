FROM --platform=linux/amd64 python:3.13-slim

WORKDIR /app

# libgl1/libglib2.0-0: OpenCV. libegl1/libgles2: MediaPipe FaceLandmarker (Tasks API)
# necesita libEGL.so.1 y libGLESv2.so.2, que NO vienen en la imagen slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 9000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
