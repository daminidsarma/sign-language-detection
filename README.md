# 🤟 Sign Language Detection (Python + OpenCV + Mediapipe)

A lightweight real-time **Sign Language Detection** system that recognizes English alphabets using **computer vision** and **machine learning**.  
This project aims to support inclusive communication for differently-abled users.

---

## 📂 Project Structure

sign-language-detector-python/
│
├── data/ # Collected dataset (hand landmarks & images)
├── Results/ # Model training/testing results
├── venv/ # Virtual environment (ignored in .gitignore)
│
├── collect_imgs.py # Script to capture hand sign images
├── create_dataset.py # Script to preprocess and create dataset
├── train_classifier.py # Train ML classifier on extracted features
├── inference_classifier.py # Run inference in real-time using webcam
│
├── data.pickle # Extracted features
├── model.p # Trained classifier model
├── requirements.txt # Python dependencies
├── README.md # Project documentation

yaml
Copy code

---

## 🚀 Features

- ✋ **Real-time hand tracking** using [Mediapipe Hands](https://google.github.io/mediapipe/solutions/hands).  
- 🔤 Detects **English alphabet letters (A–Z)** in sign language.  
- ⚡ Optimized for **low-latency inference (<50 ms per frame)**.  
- 📊 Includes dataset creation, training, and live testing pipeline.  

---

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/daminidsarma/sign-language-detection.git
   cd sign-language-detection
Create a virtual environment

bash
Copy code
python -m venv venv
venv\Scripts\activate    # On Windows
source venv/bin/activate # On Linux/Mac
Install dependencies

bash
Copy code
pip install -r requirements.txt
📊 Usage
1️⃣ Collect training images
bash
Copy code
python collect_imgs.py
Captures hand sign images for each alphabet.

Stores them in the data/ folder.

2️⃣ Create dataset
bash
Copy code
python create_dataset.py
Extracts 21 hand landmarks from Mediapipe Hands.

Saves features into data.pickle.

3️⃣ Train classifier
bash
Copy code
python train_classifier.py
Trains a simple ML model (e.g., RandomForest or SVM).

Saves trained model as model.p.

4️⃣ Run real-time detection
bash
Copy code
python inference_classifier.py
Opens webcam feed.

Detects and overlays predicted alphabet on video

📌 Future Improvements
🔡 Extend to words/phrases detection.

🌐 Deploy as a web app with Streamlit.

🤖 Integrate with speech synthesis (text-to-speech) for real-time communication.
