import cv2
import dlib
import numpy as np
import argparse

def get_landmarks(gray, face_rect, predictor):
    shape = predictor(gray, face_rect)
    landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
    return landmarks

def compute_face_metrics(landmarks):
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    mouth = landmarks[48:68]

    left_eye_center = np.mean(left_eye, axis=0)
    right_eye_center = np.mean(right_eye, axis=0)
    eye_midpoint = (left_eye_center + right_eye_center) / 2.0
    mouth_center = np.mean(mouth, axis=0)

    eye_angle = np.degrees(np.arctan2(right_eye_center[1] - left_eye_center[1],
                                       right_eye_center[0] - left_eye_center[0]))
    face_angle = np.degrees(np.arctan2(mouth_center[1] - eye_midpoint[1],
                                        mouth_center[0] - eye_midpoint[0]))
    angle_diff = abs(eye_angle - face_angle)
    return {
        "left_eye_center": left_eye_center,
        "right_eye_center": right_eye_center,
        "eye_midpoint": eye_midpoint,
        "mouth_center": mouth_center,
        "eye_angle": eye_angle,
        "face_angle": face_angle,
        "angle_diff": angle_diff
    }

def draw_landmarks(image, landmarks, metrics=None):
    for (x, y) in landmarks:
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
    if metrics:
        left_eye = tuple(metrics["left_eye_center"].astype(int))
        right_eye = tuple(metrics["right_eye_center"].astype(int))
        cv2.line(image, left_eye, right_eye, (255, 0, 0), 2)
        eye_mid = tuple(metrics["eye_midpoint"].astype(int))
        mouth_center = tuple(metrics["mouth_center"].astype(int))
        cv2.line(image, eye_mid, mouth_center, (0, 255, 0), 2)
    return image

def analyze_image(image_path, predictor_path, threshold=20, display=False):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return

    height, width = image.shape[:2]
    scale = 800.0 / width
    image = cv2.resize(image, (800, int(height * scale)))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if len(faces) == 0:
        print("No faces detected in upright image.")
        return

    upright_metrics = []
    debug_upright = image.copy()
    for face in faces:
        landmarks = get_landmarks(gray, face, predictor)
        metrics = compute_face_metrics(landmarks)
        upright_metrics.append(metrics)
        debug_upright = draw_landmarks(debug_upright, landmarks, metrics)

    avg_angle_diff_upright = np.mean([m["angle_diff"] for m in upright_metrics])

    rotated = cv2.rotate(image, cv2.ROTATE_180)
    gray_rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    faces_rotated = detector(gray_rotated)
    if len(faces_rotated) == 0:
        print("No faces detected in rotated image.")
        return

    rotated_metrics = []
    debug_rotated = rotated.copy()
    for face in faces_rotated:
        landmarks = get_landmarks(gray_rotated, face, predictor)
        metrics = compute_face_metrics(landmarks)
        rotated_metrics.append(metrics)
        debug_rotated = draw_landmarks(debug_rotated, landmarks, metrics)

    avg_angle_diff_rotated = np.mean([m["angle_diff"] for m in rotated_metrics])

    print(f"Average angle difference (upright): {avg_angle_diff_upright:.2f}°")
    print(f"Average angle difference (rotated): {avg_angle_diff_rotated:.2f}°")

    if avg_angle_diff_upright > threshold and avg_angle_diff_rotated < threshold:
        print("Thatcher effect detected: The upright image shows abnormal feature orientation that normalizes when inverted.")
    else:
        print("No Thatcher effect detected.")

    if display:
        cv2.imshow("Upright Debug", debug_upright)
        cv2.imshow("Rotated Debug", debug_rotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.imwrite("debug_upright.jpg", debug_upright)
    cv2.imwrite("debug_rotated.jpg", debug_rotated)

def main():
    parser = argparse.ArgumentParser(
        description="Detect the Thatcher effect using facial landmark analysis.")
    parser.add_argument("-i", "--image", required=True,
                        help="Path to the input image.")
    parser.add_argument("-p", "--predictor", required=True,
                        help="Path to dlib's facial landmark predictor (e.g., shape_predictor_68_face_landmarks.dat).")
    parser.add_argument("-t", "--threshold", type=float, default=20,
                        help="Angle difference threshold to flag the Thatcher effect (default: 20°).")
    parser.add_argument("-d", "--display", action="store_true",
                        help="Display debug images with landmarks.")
    args = parser.parse_args()

    analyze_image(args.image, args.predictor, args.threshold, args.display)

if __name__ == "__main__":
    main()



