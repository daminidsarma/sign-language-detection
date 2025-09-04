import os
import pickle
import cv2
import mediapipe as mp

DATA_DIR = './data'
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def iter_class_dirs(root):
    """Yield class subfolders inside DATA_DIR, skip files like .gitignore."""
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p):
            yield name, p  # (label_name, absolute_path)

def iter_images(folder):
    """Yield full paths to images, skip non-images."""
    for name in os.listdir(folder):
        if name.lower().endswith(IMG_EXTS):
            yield os.path.join(folder, name)

def main():
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}. Run collect_imgs.py first.")

    data, labels = [], []

    with mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3) as hands:
        for label, class_dir in iter_class_dirs(DATA_DIR):
            for img_path in iter_images(class_dir):
                img = cv2.imread(img_path)
                if img is None:
                    # unreadable/corrupt file — skip
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                if not results.multi_hand_landmarks:
                    # no hand detected — skip this image
                    continue

                # collect landmarks, normalize by min x/y
                data_aux, xs, ys = [], [], []
                hand_landmarks = results.multi_hand_landmarks[0]  # first hand
                for lm in hand_landmarks.landmark:
                    xs.append(lm.x); ys.append(lm.y)
                min_x, min_y = min(xs), min(ys)
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)

                data.append(data_aux)
                labels.append(label)

    out_path = 'data.pickle'
    with open(out_path, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

    print(f"✅ Saved {len(data)} samples across {len(set(labels))} classes to {out_path}")

if __name__ == '__main__':
    main()
