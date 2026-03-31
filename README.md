cd location
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
then open http://localhost:8000/

# 🚔 AI Crime Prediction - Safe Route Finder

## 📌 Project Overview

This project predicts crime risk and finds the **safest route** between two locations using:

* Machine Learning (crime prediction)
* Real road routes (OpenRouteService API)
* Map visualization (Folium)

---

## ⚙️ Setup Instructions

### 1️⃣ Clone / Download Project

Download the project folder or clone from GitHub.

---

### 2️⃣ Create Virtual Environment

Open terminal in project folder and run:

```bash
python -m venv venv
```

---

### 3️⃣ Activate Virtual Environment

#### 🟢 Windows:

```bash
venv\Scripts\activate
```

#### 🟢 Mac/Linux:

```bash
source venv/bin/activate
```

---

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 API Setup (IMPORTANT)

We use **OpenRouteService** for real road routes.

### Steps:

1. Go to: https://openrouteservice.org/
2. Sign up / login
3. Generate a free API key

---

### Where to put API key

Open your Python file and find:

```python
API_KEY = "YOUR_API_KEY"
```

Replace with your key:

```python
API_KEY = "your_actual_api_key_here"
```

---

## ▶️ Run the Project

```bash
python your_file_name.py
```

---

## 📍 Output

* A file will be generated:

```
safe_route.html
```

* Open it in browser to see:

  * Multiple routes
  * Safest route highlighted in green
  * Start & End markers

---

## 🧠 How It Works

1. Fetches real road routes using API
2. Breaks route into multiple points
3. Sends each point to ML model
4. Calculates total crime risk
5. Selects safest path

---

## 📌 Notes

* Make sure `crime_model.pkl` is present in the same folder
* Internet connection required (for API)
* Free API has request limits

---

## 🚀 Future Improvements

* Add crime heatmap
* Real-time data
* Web app (FastAPI + frontend)
* Better route optimization

---

## 👨‍💻 Team Instructions

* Always activate virtual environment before running
* Do not share API key publicly
* Keep model file (`.pkl`) safe

---

## ✅ Done!

Now you’re ready to run the project 🚀
