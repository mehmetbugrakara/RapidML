# RapidML

**Your Friendly Desktop AutoML Companion**

Welcome to **RapidML**, a lightweight, easy-to-use desktop app designed to make your machine learning journey smoother. We truly believe RapidML will save you time and help you get reliable results without wrestling with code or command lines.

---

## 🌟 Why RapidML?

- **No-fuss GUI**: Built with Tkinter for instant desktop use—no programming skills required  
- **Instant EDA**: Sweetviz-powered HTML reports to explore your data visually  
- **Auto Preprocessing**: Handles missing values, encoding, and scaling automatically  
- **Powerful Models**: Train XGBoost, LightGBM, and CatBoost with sensible defaults  
- **One-Click Outputs**: Download performance metrics and predictions in Excel format  
- **Essential Charts**: Regression line plots, confusion matrices, ROC curves, and feature importance visuals  

---

## 📋 What You’ll Get

After a quick run, your chosen **output folder** will include:  

- `report.html` — Sleek interactive EDA report  
- `metrics_regression.xlsx` or `metrics_classification.xlsx` — Summary of model performance  
- `predictions.xlsx` (if you provide test data) — Side-by-side predictions  
- `*.png` files — Key charts: regression curves, confusion matrices, ROC curves, and feature importance  

---

## ⚙️ Quick Setup

1. **Clone this repo** (or download as ZIP):  
   ```bash
   git clone https://github.com/yourusername/RapidML.git
   cd RapidML
   ```

2. **Create & activate** a Conda env (Python 3.9):  
   ```bash
   conda create -n copilot python=3.9
   conda activate copilot
   ```

3. **Install dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) **Rename** for a cleaner launch:  
   ```bash
   ren app_gui.py app_gui.pyw
   ```

---

## 🚀 Running RapidML

Run the GUI with:

```bash
python app_gui.pyw
```

or

```bash
python app_gui.py
```

Then simply:

1. **Select** your training CSV/Excel file  
2. (Optional) **Select** a test file  
3. **Enter** the target column name  
4. **Choose** Regression or Classification  
5. **Pick** an output folder  
6. **Click** **Run**—and let RapidML do the rest  

---

## 🖥️ One-Click Desktop Shortcut

For fully iconified access:

1. **Locate** `pythonw.exe` in your Conda env:  
   ```bash
   where python
   ```  
   e.g.  
   ```
   C:\ProgramData\Anaconda3\envs\copilot\python.exe
   ```
   so `pythonw.exe` is in the same folder.

2. **Create** a desktop shortcut (right-click ▶ New ▶ Shortcut) and paste:

   ```
   "C:\ProgramData\Anaconda3\envs\copilot\pythonw.exe" "D:\Work\Projeler\Co-pilotpp_gui.pyw"
   ```

3. **Name** it **RapidML**, click **Finish**.  
4. (Optional) Edit **Properties** ▶ Set **Start in** to `D:\Work\Projeler\Co-pilot` and **Change Icon** to `app.ico`.

Double-click your new icon anytime—no console windows, just RapidML’s friendly interface.

---

## 📦 requirements.txt

```text
pandas
numpy
scikit-learn
matplotlib
sweetviz
pycaret
xgboost
lightgbm
catboost
joblib
```

---

## 📝 License

Distributed under the MIT License. Use RapidML with confidence—your data deserves a swift, reliable workflow.  
