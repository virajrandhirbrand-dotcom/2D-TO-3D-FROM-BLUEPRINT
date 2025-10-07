import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from ultralytics import YOLO

# === CONFIG ===
BASE = os.getcwd()
UPLOADS = os.path.join(BASE, "upload")
STATIC = os.path.join(BASE, "static")
YOLO_MODEL = r"C:\Hackathon\runs\detect\blueprint_v843\weights\best.pt"

os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(STATIC, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder=STATIC)
app.config['UPLOAD_FOLDER'] = UPLOADS
model = YOLO(YOLO_MODEL)

# === DETECTION ===
def detect_objects(image_path):
    result = model(image_path)[0]
    wall_lines, doors = [], []
    for box in result.boxes.data.cpu().numpy():
        x1, y1, x2, y2, _, cls = box
        label = model.names[int(cls)]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = abs(x2 - x1), abs(y2 - y1)

        if label == "wall":
            length = max(w, h)
            if length < 20:
                continue
            angle = np.arctan2(h, w) if w != 0 else np.pi / 2
            dx = (length / 2) * np.cos(angle)
            dy = (length / 2) * np.sin(angle)
            if w > h:
                start = (int(cx - dx), int(cy))
                end = (int(cx + dx), int(cy))
            else:
                start = (int(cx), int(cy - dy))
                end = (int(cx), int(cy + dy))
            wall_lines.append((start, end))

        elif label == "door":
            doors.append(((x1, y1), (x2, y2)))

    return wall_lines, doors

# === GEOMETRY ===
def make_wall_segments(start, end, thickness=5, height=100):
    start, end = np.array(start, dtype=float), np.array(end, dtype=float)
    dx, dy = end - start
    length = np.hypot(dx, dy)
    if length < 10:
        return []

    ox, oy = -dy / length * thickness / 2, dx / length * thickness / 2
    base = np.array([
        [start[0] + ox, start[1] + oy, 0],
        [end[0] + ox, end[1] + oy, 0],
        [end[0] - ox, end[1] - oy, 0],
        [start[0] - ox, start[1] - oy, 0]
    ])
    top = base.copy()
    top[:, 2] = height
    v = np.vstack((base, top))
    f = np.array([
        [0,1,5], [0,5,4], [1,2,6], [1,6,5],
        [2,3,7], [2,7,6], [3,0,4], [3,4,7],
        [4,5,6], [4,6,7], [0,3,2], [0,2,1]
    ])
    return [(v, f)]

def make_door_block(x1, y1, x2, y2, depth=50, height=90):
    x_min, x_max = sorted([x1, x2])
    y_min, y_max = sorted([y1, y2])
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    door_width = max(abs(x2 - x1), abs(y2 - y1)) * 0.2
    door_thickness = depth
    z_offset = 5

    if abs(x2 - x1) > abs(y2 - y1):
        w, h = door_width, door_thickness
    else:
        w, h = door_thickness, door_width

    base = np.array([
        [cx - w/2, cy - h/2, z_offset], [cx + w/2, cy - h/2, z_offset],
        [cx + w/2, cy + h/2, z_offset], [cx - w/2, cy + h/2, z_offset]
    ])
    top = base.copy()
    top[:, 2] += height
    v = np.vstack((base, top))
    f = np.array([
        [0,1,5], [0,5,4], [1,2,6], [1,6,5],
        [2,3,7], [2,7,6], [3,0,4], [3,4,7],
        [4,5,6], [4,6,7], [0,3,2], [0,2,1]
    ])
    return [(v, f)]

def make_floor_from_bounds(vertices, thickness=10):
    if len(vertices) == 0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)
    xy = vertices[:, :2]
    min_x, min_y = np.min(xy, axis=0)
    max_x, max_y = np.max(xy, axis=0)
    z0, z1 = -thickness, 0
    v = np.array([
        [min_x, min_y, z0], [max_x, min_y, z0],
        [max_x, max_y, z0], [min_x, max_y, z0],
        [min_x, min_y, z1], [max_x, min_y, z1],
        [max_x, max_y, z1], [min_x, max_y, z1]
    ])
    f = np.array([
        [0,1,2], [0,2,3], [4,5,6], [4,6,7],
        [0,1,5], [0,5,4], [1,2,6], [1,6,5],
        [2,3,7], [2,7,6], [3,0,4], [3,4,7]
    ])
    return v, f

def make_textured_plane(image_path, bounds, z_offset=-0.1):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    min_x, min_y = np.min(bounds[:, :2], axis=0)
    max_x, max_y = np.max(bounds[:, :2], axis=0)

    v = np.array([
        [min_x, min_y, z_offset],
        [max_x, min_y, z_offset],
        [max_x, max_y, z_offset],
        [min_x, max_y, z_offset],
        [min_x, min_y, z_offset],
        [max_x, min_y, z_offset],
        [max_x, max_y, z_offset],
        [min_x, max_y, z_offset]
    ])
    f = np.array([
        [0,1,2], [0,2,3], [7,6,5], [7,5,4]
    ])
    vt = np.array([
        [0, 1], [1, 1], [1, 0], [0, 0],
        [0, 1], [1, 1], [1, 0], [0, 0]
    ])
    return v, f, vt

# === 3D CONVERSION ===
def convert_to_3d(image_path):
    wall_lines, doors = detect_objects(image_path)

    all_v, all_f, all_mtl, groups = [], [], [], []
    face_offset = 0
    vt_all, vt_idx_all = [], []

    for w_start, w_end in wall_lines:
        for v, f in make_wall_segments(w_start, w_end):
            all_v.append(v)
            all_f.append(f + face_offset)
            all_mtl.extend(["Wall"] * len(f))
            groups.extend(["Wall"] * len(f))
            face_offset += len(v)

    for (x1, y1), (x2, y2) in doors:
        for v, f in make_door_block(x1, y1, x2, y2):
            all_v.append(v)
            all_f.append(f + face_offset)
            all_mtl.extend(["Door"] * len(f))
            groups.extend(["Door"] * len(f))
            face_offset += len(v)

    if all_v:
        v_stack = np.vstack(all_v)
        floor_v, floor_f = make_floor_from_bounds(v_stack)
        all_v.append(floor_v)
        all_f.append(floor_f + face_offset)
        all_mtl.extend(["Floor"] * len(floor_f))
        groups.extend(["Floor"] * len(floor_f))
        face_offset += len(floor_v)

        plane_v, plane_f, vt = make_textured_plane(image_path, v_stack)
        all_v.append(plane_v)
        all_f.append(plane_f + face_offset)
        vt_all.extend(vt)
        vt_idx_all.extend([[0, 1, 2], [0, 2, 3], [7, 6, 5], [7, 5, 4]])
        all_mtl.extend(["Blueprint"] * len(plane_f))
        groups.extend(["Blueprint"] * len(plane_f))

    final_v = np.vstack(all_v)
    final_f = np.vstack(all_f)

    center = final_v.mean(axis=0)
    final_v -= center
    scale = 100.0 / np.ptp(final_v[:, :2])
    final_v *= scale

    obj_path = os.path.join(STATIC, "output_3d.obj")
    mtl_path = os.path.join(STATIC, "output_3d.mtl")

    with open(mtl_path, "w") as mtl:
        mtl.write("newmtl Wall\nKd 0.8 0.8 0.8\n")
        mtl.write("newmtl Floor\nKd 0.6 0.6 0.6\n")
        mtl.write("newmtl Door\nKd 0.4 0.2 0.1\n")
        mtl.write("newmtl Blueprint\nmap_Kd blueprint_texture.jpg\n")

    with open(obj_path, "w") as obj:
        obj.write("mtllib output_3d.mtl\n")
        for v in final_v:
            obj.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        for vt in vt_all:
            obj.write(f"vt {vt[0]:.3f} {vt[1]:.3f}\n")
        for i, f in enumerate(final_f):
            obj.write(f"usemtl {all_mtl[i]}\ng {groups[i]}\n")
            if all_mtl[i] == "Blueprint":
                vi = [str(idx + 1) for idx in f]
                if i - len(vt_idx_all) >= 0 and (i - len(vt_idx_all)) < len(vt_idx_all):
                    vti = [str(idx + 1) for idx in vt_idx_all[i - len(vt_idx_all)]]
                else:
                    vti = ["1", "1", "1"]  # fallback
                obj.write(f"f {' '.join(f'{vi[k]}/{vti[k]}' for k in range(3))}\n")
            else:
                obj.write(f"f {' '.join(str(idx + 1) for idx in f)}\n")

    blueprint_texture = os.path.join(STATIC, "blueprint_texture.jpg")
    cv2.imwrite(blueprint_texture, cv2.imread(image_path))
    return obj_path

# === ROUTES ===
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return "No file uploaded", 400
        path = os.path.join(UPLOADS, file.filename)
        file.save(path)

        wall_lines, doors = detect_objects(path)
        img = cv2.imread(path)
        for (x1, y1), (x2, y2) in wall_lines:
            cv2.line(img, (x1, y1), (x2, y2), (200, 200, 200), 2)
        for (x1, y1), (x2, y2) in doors:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (42, 42, 165), 2)
            cv2.putText(img, "door", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (42, 42, 165), 1)

        preview_path = os.path.join(STATIC, "preview.jpg")
        cv2.imwrite(preview_path, img)

        convert_to_3d(path)
        return render_template("result.html", model_path="output_3d.obj", preview_path="preview.jpg")
    return render_template("index.html")

@app.route("/download")
def download():
    return send_from_directory(STATIC, "output_3d.obj", as_attachment=True)

@app.route('/result', methods=['POST'])
def result():
    file = request.files['file']
    if file:
        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)
        convert_to_3d(path)
        return render_template("result.html")


if __name__ == "__main__":
    app.run(debug=True)

