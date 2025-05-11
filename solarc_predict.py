import os
import numpy as np
import tensorflow as tf
import cv2
import json
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Constants
IMG_SIZE = (224, 224)
NUM_CLASSES = 7  # 클래스 수 (dx의 고유값 개수)
MODEL_PATH = 'checkpoints/best_model.h5'

# Class labels
CLASS_LABELS = {
    "bkl": "Benign Keratosis",
    "nv": "Melanocytic Nevus",
    "mel": "Melanoma",
    "df": "Dermatofibroma",
    "vasc": "Vascular Lesion",
    "akiec": "Actinic Keratosis",
    "bcc": "Basal Cell Carcinoma"
}

def build_model(num_classes):
    """
    모델 구조 정의
    """
    base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def load_model_weights(model_path):
    """
    모델 아키텍처를 정의하고, 가중치만 로드
    """
    model = build_model(NUM_CLASSES)
    
    if os.path.exists(model_path):
        try:
            model.load_weights(model_path)
            print(f"Weights loaded from {model_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
    else:
        print(f"Model file not found at {model_path}")
    
    return model

def preprocess_image(image_path):
    """
    이미지 전처리 함수
    - 이미지 리사이즈, 정규화
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found at {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)  # (1, 224, 224, 3) 형태로 변경

        return img

    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        return None

def predict_with_threshold(model, image_path, threshold=0.5):
    """
    이미지 경로를 입력받아 50% 이상 확률을 갖는 클래스들을 예측
    Args:
        model: TensorFlow 모델 객체
        image_path (str): 이미지 경로
        threshold (float): 확률 임계값 (기본값: 0.5)

    Returns:
        dict: 50% 이상 확률 클래스와 해당 확률
    """
    img = preprocess_image(image_path)

    if img is None:
        return {}

    # 모델 예측
    preds = model.predict(img)[0]  # (7,) 벡터 출력

    # 확률이 50% 이상인 클래스만 추출
    high_confidence = {
        CLASS_LABELS[label]: float(preds[i])
        for i, label in enumerate(CLASS_LABELS.keys())
        if preds[i] >= threshold
    }

    # 확률 내림차순 정렬
    high_confidence = dict(sorted(high_confidence.items(), key=lambda item: item[1], reverse=True))

    return high_confidence

def main():
    # 모델 로드
    model = load_model_weights(MODEL_PATH)

    # 사용자 입력
    image_path = input("Enter the path of the image to predict: ")

    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    # Threshold 입력 (기본값 0.5)
    try:
        threshold = float(input("Enter the confidence threshold (0-1, default 0.5): ") or 0.5)
        if not (0 <= threshold <= 1):
            raise ValueError
    except ValueError:
        print("Invalid threshold value. Using default value of 0.5.")
        threshold = 0.5

    # 예측 수행
    predictions = predict_with_threshold(model, image_path, threshold)

    if predictions:
        print("\nPredicted Diseases (≥ {:.2f} Confidence):".format(threshold))
        for disease, prob in predictions.items():
            print(f"{disease}: {prob * 100:.2f}%")
    else:
        print(f"No diseases detected with confidence ≥ {threshold}")

    # 예측 결과를 JSON으로 저장
    output_path = "prediction_output.json"
    with open(output_path, "w") as f:
        json.dump(predictions, f)

    print(f"\nPrediction results saved to {output_path}")

if __name__ == "__main__":
    main()

#주의: 경로 입력시 ''(따옴표) 금지, 한글 들어가게 금지
