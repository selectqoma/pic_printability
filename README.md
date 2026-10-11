# Pic Printability

A Streamlit experiment that scores faces in photos using landmarks and emotion detection. It combines Dlib, NNabla and a CNN trained on FER2013.

![Demo](https://github.com/selectqoma/pic_printability/assets/85152770/db9dbe0a-fdc3-4d1a-b278-95dd819ba6a6.png)

## Run with Docker

```sh
docker build -t printability -f docker/Dockerfile .
docker run --rm -p 127.0.0.1:8501:8501 printability
```

Open http://localhost:8501 and upload a photo. The build downloads pretrained face models.

This was a quick prototype. Smile detection and rotated faces are unreliable, and the score is a heuristic rather than a measure of print quality.
