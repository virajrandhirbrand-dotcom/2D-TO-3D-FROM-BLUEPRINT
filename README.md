# 2D-TO-3D-FROM-BLUEPRINT


🏠 2D to 3D Print Converter
📌 Overview

The 2D to 3D Print Converter is a computer vision and AI-powered project that transforms 2D floor plans or blueprints into 3D printable models.
It allows architects, students, and designers to quickly visualize or print physical models of building layouts without needing manual 3D modeling in tools like Blender.

This project combines OpenCV, Python, and Three.js (for 3D visualization) and can also export models in OBJ/MTL or STL format for 3D printing.

🚀 Features

🧠 AI-based Floor Plan Enhancement – Cleans and sharpens uploaded blueprints automatically.

🧱 Wall, Door & Window Detection – Uses OpenCV / YOLOv8 for accurate feature extraction.

🏗️ 3D Model Generation – Converts detected structures into a 3D mesh (OBJ/MTL/STL format).

🌐 Web-based Interface – Simple Flask web app with drag & drop image upload.

🎨 Real-time 3D Visualization – Interactive 3D model viewer powered by Three.js.

🖨️ 3D Print Ready Output – Exports ready-to-print 3D files.

🧩 Tech Stack

Frontend: HTML, CSS, JavaScript, Three.js
Backend: Python (Flask)
AI / CV Tools: OpenCV, NumPy, YOLOv8
3D Formats: OBJ, MTL, STL
Optional: Blender / Open3D for mesh optimization

⚙️ Installation & Setup
1. Clone the Repository
git clone https://github.com/yourusername/2D-to-3D-Print.git
cd 2D-to-3D-Print

2. Install Dependencies
pip install -r requirements.txt

3. Run the Flask App
python app.py

4. Open in Browser

Go to → http://127.0.0.1:5000/

🖼️ How It Works

Upload a 2D floor plan image (JPG/PNG).

The system processes the image with OpenCV for contour and edge detection.

AI model identifies walls, windows, and doors.

The detected layout is converted into a 3D mesh using Python.

The user can view the model in 3D or export for 3D printing.

 Example Workflow
Input: floorplan.png
↓
Processed Image: edge_detected.png
↓
Generated 3D Model: model.obj / model.stl

 Folder Structure
2D-to-3D-Print/
│
├── static/               # CSS, JS, assets
├── templates/            # HTML files (index.html, result.html)
├── models/               # Generated 3D models
├── app.py                # Flask main file
├── process_blueprint.py  # AI/OpenCV logic
├── requirements.txt
└── README.md

 Future Enhancements

 Support for curved walls and complex shapes

 Add AR/VR visualization

Material and texture mapping for realism
 Export to glTF / FBX formats
