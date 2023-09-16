import cv2
import os
import onnxruntime
import dlib
import nnabla as nn
import nnabla.functions as F
from nnabla.logger import logger
from skimage import io, color
from args import get_args
from model import fan, resnet_depth
from utils import *
from external_utils import *
import numpy as np
from scipy.spatial import distance
from nnabla.ext_utils import get_extension_context
from concurrent.futures import ThreadPoolExecutor
from collections import Counter



MODEL_PATH = './2DFAN4_NNabla_model.h5'
FACE_DETECTOR_PATH = './mmod_human_face_detector.dat'
REFERENCE_SCALE = 195
EMOTION_DETECTION_MODEL_PATH = './emotion_detection.onnx'
NETWORK_SIZE = 4
CONTEXT = "cpu"
OUTPUT = "output.png"
TYPE_CONFIG = "float"
EMOTION_RANGES = ["Angry","Disgusted","Fearful","Happy","Neutral","Sad", "Surprised"]
EAR_THRESHOLD = 0.2
MOUTH_ASPECT_RATIO_THRESHOLD = 0.4


def main(np_image):

    model_path = MODEL_PATH
    face_detector_path = FACE_DETECTOR_PATH
    reference_scale = REFERENCE_SCALE
    network_size = NETWORK_SIZE
    context = CONTEXT
    output = OUTPUT
    type_config = TYPE_CONFIG

    

    ctx = get_extension_context(context, device_id=0, type_config=type_config)
    nn.set_default_context(ctx)
    nn.set_auto_forward(True)

    sess = onnxruntime.InferenceSession(EMOTION_DETECTION_MODEL_PATH)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    emotions = []

    image = np_image 
    if image.ndim == 2:
        image = color.gray2rgb(image)
    elif image.shape[-1] == 4:
        image = image[..., :3]

    face_detector = dlib.get_frontal_face_detector()
    detected_faces = face_detector(cv2.cvtColor(image[..., ::-1].copy(), cv2.COLOR_BGR2GRAY))
    detected_faces = [[d.left(), d.top(), d.right(), d.bottom()] for d in detected_faces]
    logger.info("Number of faces detected: {}".format(len(detected_faces)))

    if len(detected_faces) == 0:
        print("Warning: No faces were detected.")
        return None

    # Load FAN weights
    with nn.parameter_scope("FAN"):
        print("Loading FAN weights...")
        nn.load_parameters(model_path)

    landmarks = []
    emotions = []

    for _, d in enumerate(detected_faces):
        face = image[d[1]:d[3], d[0]:d[2]]
        emotion = predict_emotion(face, sess, input_name, output_name)
        emotions.append(emotion)
        center = [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0]
        center[1] = center[1] - (d[3] - d[1]) * 0.12
        scale = (d[2] - d[0] + d[3] - d[1]) / reference_scale
        inp = crop(image, center, scale)
        inp = nn.Variable.from_numpy_array(inp.transpose((2, 0, 1)))
        inp = F.reshape(F.mul_scalar(inp, 1 / 255.0), (1,) + inp.shape)
        with nn.parameter_scope("FAN"):
            out = fan(inp, network_size)[-1]
        pts, pts_img = get_preds_fromhm(out, center, scale)
        pts, pts_img = F.reshape(pts, (68, 2)) * \
            4, F.reshape(pts_img, (68, 2))

        landmarks.append(pts_img.d)
        emotions.append(emotion)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image_rgb, face_features = visualize_both_eyes_and_mouth(landmarks, image, output, detected_faces, emotion)
    mean_printability_score = get_mean_printability_score(face_features, emotions)
    overall_emotion, _ = Counter(emotions).most_common(1)[0]

    return image_rgb, mean_printability_score, overall_emotion

def get_mean_printability_score(face_features, emotions):

    printability_scores = []
    weights = []

    for i in range(len(face_features)):

        open_eyes  = face_features[i][0] >= 0.5
        smile = face_features[i][1] == 1

        weight = 1  

        if emotions[i] == "Happy" and open_eyes and smile:
            printability_scores.append(100)
            weight = 3  
        elif emotions[i] == "Happy" and open_eyes:
            printability_scores.append(80)
            weight = 3  
        elif emotions[i] in ["Neutral", "Surprised", "Unknown"] and open_eyes:
            printability_scores.append(50)
        elif not open_eyes and smile:
            printability_scores.append(25) 
        elif not open_eyes and not smile:
            printability_scores.append(0)
        elif emotions[i] in ["Sad", "Angry", "Disgusted", "Fearful"]:
            printability_scores.append(0)

        weights.append(weight)
    if len(printability_scores) > 0 and len(weights) > 0:
        weighted_mean = np.average(printability_scores, weights=weights)
        return weighted_mean
    else:
        return 0

def predict_emotion(face, sess, input_name, output_name):
    try:
        face_resized = cv2.resize(face, (48, 48))
    except:
        logger.info("Problem with processing face. Skipping...")
        return "Unknown"
    face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    face_resized = np.reshape(face_resized, (1, 48, 48, 1)).astype(np.float32)
    
    # Run ONNX inference
    result = sess.run([output_name], {input_name: face_resized})
    output_array = np.array(result[0])
    
    # Determine the emotion
    emotion = EMOTION_RANGES[np.argmax(output_array)]
    logger.info(output_array)
    return emotion


def calculate_eye_aspect_ratio(eye):
    return (np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4])) / (2 * np.linalg.norm(eye[0] - eye[3]))


def calculate_mouth_aspect_ratio(landmarks):
    left_corner = np.array(landmarks[48])
    right_corner = np.array(landmarks[54])
    upper_mid = np.mean(np.array([landmarks[51], landmarks[62], landmarks[66]]), axis=0)
    lower_mid = np.mean(np.array([landmarks[57], landmarks[58], landmarks[56]]), axis=0)
    
    vertical_distance = np.linalg.norm(upper_mid - lower_mid)
    horizontal_distance = np.linalg.norm(right_corner - left_corner)
    
    if horizontal_distance == 0:
        return 0.0
    
    mar = vertical_distance / horizontal_distance
    
    return mar


def is_mid_point_lower(landmarks):
    left_corner = landmarks[48]
    middle_point = landmarks[62]
    right_corner = landmarks[54]

    return middle_point[1] > left_corner[1] or middle_point[1] > right_corner[1]


def visualize_both_eyes_and_mouth(landmarks, image, output, detected_faces, emotion):

    face_features = []

    for face_landmarks in landmarks:
        left_eye = face_landmarks[36:42]
        right_eye = face_landmarks[42:48]
        mouth = face_landmarks[48:68]
        left_eyebrow = face_landmarks[17:22]
        right_eyebrow = face_landmarks[22:27]

        left_ear = calculate_eye_aspect_ratio(np.array(left_eye))
        right_ear = calculate_eye_aspect_ratio(np.array(right_eye))
        
        eyes_open_ratio = 0

        if left_ear > 0.2 and right_ear > 0.2:
            eyes_open_ratio = 1
        elif left_ear > 0.2 or right_ear > 0.2:
            eyes_open_ratio = 0.5
        else:
            eyes_open_ratio = 0

        lower_lip_mid_point_lower = is_mid_point_lower(face_landmarks)
        mouth_aspect_ratio = calculate_mouth_aspect_ratio(face_landmarks)

        logger.info(f"mar:{mouth_aspect_ratio}")
        logger.info(f"llmpl:{lower_lip_mid_point_lower}")

        smile = False

        if lower_lip_mid_point_lower:
            smile = True
        elif mouth_aspect_ratio < MOUTH_ASPECT_RATIO_THRESHOLD:
            smile = True


        face_features.append([eyes_open_ratio, smile])

        def draw_facial_feature_points(feature_points, color=(0, 255, 0)):
            n = len(feature_points)
            for i in range(n):
                x1, y1 = map(int, feature_points[i])
                x2, y2 = map(int, feature_points[(i+1) % n])
                cv2.line(image, (x1, y1), (x2, y2), color, 2)


        # Drawing eyes and mouth
        draw_facial_feature_points(left_eye)
        draw_facial_feature_points(right_eye)
        draw_facial_feature_points(mouth, color=(255, 0, 0))

        for d in detected_faces:
            cv2.rectangle(image, (d[0], d[1]), (d[2], d[3]), (0, 255, 255), 2)

        # Drawing eyebrows
        draw_facial_feature_points(left_eyebrow, color=(0, 0, 255))
        draw_facial_feature_points(right_eyebrow, color=(0, 0, 255))

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image_rgb, face_features 
