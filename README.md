# Pic Printability

App to check your pic's printability with a simple UI.

![Demo Image](https://github.com/selectqoma/pic_printability/assets/85152770/db9dbe0a-fdc3-4d1a-b278-95dd819ba6a6.png)

## Technologies Used

- Dlib's face detection
- NNabla face keypoint detection
- Emotion detection model (simple CNN) trained on the FER2013 Dataset

## Getting Started

### Prerequisites

- Docker

### Installation and Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/pic_printability.git
    ```

2. Navigate to the directory:
    ```bash
    cd pic_printability
    ```

3. Build the Docker image:
    ```bash
    docker build -t printability -f docker/Dockerfile .
    ```

4. Run the Docker container:
    ```bash
    docker run printability
    ```
   
5. Go to the URL that pops up in your terminal to open the UI.

## Limitations and Future Improvements

This program was conceived very quickly (~5 hours) and has several areas for improvement:

- **Smile Detection**: Currently not robust; could be improved with a simple CNN instead of relying on aspect ratios.
- **Image Rotation**: The app doesn't work well with rotated faces or images; this can be easily fixed.
- **Printability Score**: The algorithm for calculating printability can be significantly improved.
- **UI Features**: The user interface could be enriched with more functionalities.


