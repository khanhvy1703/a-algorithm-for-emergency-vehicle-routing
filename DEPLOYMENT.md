# Deployment Guide - Making Your App Available to Everyone

## Option 1: Streamlit Cloud (FREE & EASIEST) ⭐

### What is it?
Streamlit turns Python scripts into web apps. Anyone can access it with just a URL - no installation needed!

### Steps to Deploy:

1. **Install Streamlit locally first to test**:
```bash
pip install streamlit matplotlib
```

2. **Test the web version locally**:
```bash
streamlit run app_streamlit.py
```
This opens in your browser at http://localhost:8501

3. **Deploy to Streamlit Cloud (FREE)**:
   - Go to https://streamlit.io/cloud
   - Sign in with GitHub
   - Click "New app"
   - Select your repository: `khanhvy1703/a-algorithm-for-emergency-vehicle-routing`
   - Branch: `divit`
   - Main file: `app_streamlit.py`
   - Click "Deploy"

4. **Share the URL**:
   - You'll get a URL like: `https://emergency-routing.streamlit.app`
   - Anyone can access it from any device!

### Requirements file for Streamlit:
Create `requirements_streamlit.txt`:
```
streamlit
matplotlib
numpy
```

---

## Option 2: GitHub Pages with PyScript (Also FREE)

This runs Python directly in the browser!

Create `index.html`:
```html
<!DOCTYPE html>
<html>
<head>
    <title>Emergency Vehicle Routing</title>
    <link rel="stylesheet" href="https://pyscript.net/latest/pyscript.css" />
    <script defer src="https://pyscript.net/latest/pyscript.js"></script>
</head>
<body>
    <h1>Emergency Vehicle Routing System</h1>
    <div id="output"></div>
    <py-script src="./scripts/main.py"></py-script>
</body>
</html>
```

Then enable GitHub Pages in your repo settings.

---

## Option 3: Desktop App with PyInstaller

Make it a downloadable .exe file (Windows) or .app (Mac):

1. **Install PyInstaller**:
```bash
pip install pyinstaller
```

2. **Create executable**:
```bash
pyinstaller --onefile --windowed scripts/main.py
```

3. **Find your app**:
   - Look in `dist/` folder
   - Share the .exe or .app file
   - Users just double-click to run!

---

## Option 4: Replit (FREE Online IDE)

1. Go to https://replit.com
2. Create new Python repl
3. Upload your files
4. Click "Run"
5. Share the link - others can run it directly in browser!

---

## Option 5: Google Colab (For Demo)

1. Upload main.py to Google Colab
2. Add this at the top:
```python
!pip install pygame
import sys
sys.path.append('/content')
```
3. Share the notebook link

---

## Recommended: Streamlit

**Why Streamlit is best for your project:**
- ✅ Completely FREE
- ✅ No installation needed for users
- ✅ Works on any device (phone, tablet, computer)
- ✅ Professional looking
- ✅ Easy to share (just a URL)
- ✅ Automatic updates when you push to GitHub

**Your app would be live at something like:**
```
https://emergency-vehicle-routing.streamlit.app
```

Anyone can access it instantly!