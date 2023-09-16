# pic_printability

App to check your pic printability in a simple ui

![image](https://github.com/selectqoma/pic_printability/assets/85152770/db9dbe0a-fdc3-4d1a-b278-95dd819ba6a6)

Based on dlib's face detection and nnabla face keypoint detection. It also uses an emotion detection model (simple CNN)trained on the FER2013 Dataset

How to:
  - clone repo and run: docker build -t printability -f docker/Dockerfile .
  - after it finished building: docker run printability. To to the url that should pop-up in your terminal to open the ui


This program was conceived very quickly ~ 5 hours and it has a lot of improvement points. Some of them are:
  - Smile detection is not robust enough for now (could be improved with simple CNN instead of relying on aspect ratio etc.)
  - Doesn't work with rotated faces or images (can be easily fixed)
  - The printability score can be improved (a lot)
  - A lot of different features could be added to the UI
