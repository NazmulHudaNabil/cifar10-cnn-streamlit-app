<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.18-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.55-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Keras-CNN-D00000?style=for-the-badge&logo=keras&logoColor=white" />
  <img src="https://img.shields.io/badge/Deployed-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white" />
</p>

<h1 align="center">🧠 CIFAR-10 Image Classification using CNN</h1>

<p align="center">
  A deep learning web application that classifies images into <b>10 categories</b> using a trained
  <b>Convolutional Neural Network (CNN)</b> on the CIFAR-10 dataset — built with TensorFlow/Keras and
  deployed with Streamlit.
</p>

<p align="center">
  <a href="https://cifar10-cnn-streamlit-app.onrender.com" target="_blank">
    <img src="https://img.shields.io/badge/🚀_Live_Demo-Click_Here-FF6B6B?style=for-the-badge" />
  </a>
</p>

---

## 📸 Screenshots

| Home Page | Prediction Page |
|:---------:|:---------------:|
| ![Home](https://via.placeholder.com/400x250/0E1117/FF6B6B?text=Home+Page) | ![Prediction](https://via.placeholder.com/400x250/0E1117/45B7D1?text=Prediction+Page) |

> 💡 *Replace the placeholder images above with actual screenshots of your app.*

---

## 🎯 Features

- 🏠 **Landing Section** — Gradient hero title, project description & animated category showcase
- 📊 **Dataset Explorer** — CIFAR-10 dataset info with real sample image grids loaded from the dataset
- 🧬 **Model Architecture** — Interactive CNN architecture diagram, training metrics & expandable model summary
- 🔮 **Live Prediction** — Upload any image, get instant classification with **Top-3 predictions** & confidence bars
- 🎨 **Dark Theme UI** — Glassmorphism cards, gradient text, smooth hover animations & responsive layout

---

## 🧩 Supported Classes

| | | | | |
|:---:|:---:|:---:|:---:|:---:|
| ✈️ Airplane | 🚗 Automobile | 🐦 Bird | 🐱 Cat | 🦌 Deer |
| 🐶 Dog | 🐸 Frog | 🐴 Horse | 🚢 Ship | 🚚 Truck |

---

## 🏗️ CNN Architecture

```
Input (32×32×3)
    │
    ├── Conv2D (32 filters, 3×3, ReLU)
    ├── MaxPool2D (2×2)
    ├── BatchNormalization
    │
    ├── Conv2D (64 filters, 3×3, ReLU)
    ├── MaxPool2D (2×2)
    ├── BatchNormalization
    │
    ├── Conv2D (128 filters, 3×3, ReLU)
    ├── MaxPool2D (2×2)
    ├── BatchNormalization
    │
    ├── Flatten
    ├── Dense (128, ReLU)
    ├── Dropout (0.5)
    └── Dense (10, Softmax) → Output
```

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| **Training Accuracy** | ~87.65% |
| **Validation Accuracy** | ~73.33% |
| **Optimizer** | Adam |
| **Loss Function** | Sparse Categorical Crossentropy |
| **Epochs** | 20 |
| **Batch Size** | 32 |

---

## ⚙️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.12** | Programming language |
| **TensorFlow / Keras** | Model training & inference |
| **Streamlit** | Web application framework |
| **NumPy** | Image preprocessing |
| **Pillow (PIL)** | Image handling |
| **Matplotlib** | Visualization & charts |
| **Render** | Cloud deployment |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/NazmulHudaNabil/cifar10-cnn-streamlit-app.git
cd cifar10-cnn-streamlit-app

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at **http://localhost:8501** 🎉

---

## 📁 Project Structure

```
cifar10-cnn-streamlit-app/
│
├── app.py                  # Main Streamlit application
├── model_clfar-10.keras    # Pre-trained CNN model
├── CIFAR_10.ipynb          # Model training notebook
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python version for deployment
├── .python-version         # Python version (Render)
├── run.sh                  # Local launch script
├── .streamlit/
│   └── config.toml         # Streamlit dark theme config
└── README.md
```

---

## 🌐 Deployment

This app is deployed on **Render**:

👉 **Live Demo:** [https://cifar10-cnn-streamlit-app.onrender.com](https://cifar10-cnn-streamlit-app.onrender.com)

### Deploy Your Own

1. Fork this repository
2. Create a new **Web Service** on [Render](https://render.com)
3. Connect your GitHub repo
4. Set the following:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Deploy! 🚀

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Md Nazmul Huda Nabil**

- GitHub: [@NazmulHudaNabil](https://github.com/NazmulHudaNabil)

---

<p align="center">
  ⭐ If you found this project useful, please consider giving it a star!
</p>
