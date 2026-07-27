import cv2
import numpy as np
import math
import os
import time

from ultralytics import YOLO
from google.colab import files

 

MODEL_PATH = r"/content/best-2.pt"
VIDEO_PATH = r"/content/192.168.1.101_01_2026051116544357.mp4"
OUTPUT_PATH = r"/content/output_video_with_measurements.mp4"


 

PIXELS_PER_INCH = 96
INCH_TO_CM = 2.54

 

print("  جاري تحميل الموديل...")
model = YOLO(MODEL_PATH)
print("  تم تحميل الموديل بنجاح")


 

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f" الفيديو غير موجود: {VIDEO_PATH}")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(" فشل فتح الفيديو")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("\n🎥 معلومات الفيديو")
print(f"الأبعاد: {width} x {height}")
print(f"FPS: {fps:.2f}")
print(f"عدد الفريمات: {total_frames}")


 

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))


  

def calculate_distance(point1, point2, pixels_per_inch=PIXELS_PER_INCH):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    pixels = math.hypot(dx, dy)
    inches = pixels / pixels_per_inch
    cm = inches * INCH_TO_CM
    return pixels, inches, cm

 

CLASS_COLORS = {
    "r1": (255, 0, 0),
    "r2": (0, 255, 255),
    "h": (255, 0, 255),
    "wire": (255, 165, 0)
}




def process_image(image):
    image_copy = image.copy()

     results = model.predict(source=image, conf=0.40, save=False, verbose=False)

     all_detections = []
    wires_detections = []
    clamps_R2_detections = []
    h_detections = []

    shortest_wire = None
    longest_wire = None
    reference_clamp = None
 

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = result.names[cls_id]
            confidence = float(box.conf[0].cpu().numpy())

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            detection = {
                "class": cls_name,
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "confidence": confidence
            }

            all_detections.append(detection)

            cls_lower = cls_name.lower().strip()

            if cls_lower in ["wire", "wires", "سلك", "اسلاك"]:
                wires_detections.append(detection)
            elif cls_lower == "r2":
                clamps_R2_detections.append(detection)
            elif cls_lower == "h":
                h_detections.append(detection)
 

    if len(clamps_R2_detections) > 0 and len(wires_detections) > 0:
        wire_y_values = [w["center"][1] for w in wires_detections]
        min_wire_y = min(wire_y_values)
        max_wire_y = max(wire_y_values)

        clamps_after = []   # R2 أعلى كل الأسلاك (y أصغر من min_wire_y) → يُحسب
        clamps_before = []  # R2 أسفل الأسلاك أو متداخل معها → لا يُحسب

        for clamp in clamps_R2_detections:
            clamp_y = clamp["center"][1]
            if clamp_y < min_wire_y:
                clamps_after.append(clamp)
            else:
                clamps_before.append(clamp)

         if clamps_after:
            reference_clamp = max(clamps_after, key=lambda c: c["confidence"])
        else:
            reference_clamp = None

 

    if reference_clamp is not None and len(wires_detections) >= 2:
        wires_sorted_by_y = sorted(wires_detections, key=lambda w: w["center"][1])

        shortest_wire = wires_sorted_by_y[0]
        longest_wire = wires_sorted_by_y[-1]

        dist_long_short_px, dist_long_short_in, dist_long_short_cm = calculate_distance(
            longest_wire["center"], shortest_wire["center"]
        )

        cv2.line(
            image_copy,
            (int(longest_wire["center"][0]), int(longest_wire["center"][1])),
            (int(shortest_wire["center"][0]), int(shortest_wire["center"][1])),
            (0, 0, 255), 3
        )

        mid_x = int((longest_wire["center"][0] + shortest_wire["center"][0]) / 2)
        mid_y = int((longest_wire["center"][1] + shortest_wire["center"][1]) / 2)

         cv2.rectangle(image_copy, (mid_x - 10, mid_y - 25), (mid_x + 420, mid_y + 10), (0, 0, 0), -1)
        cv2.putText(image_copy, f"Long-Short: {dist_long_short_px:.0f}px | {dist_long_short_cm:.2f}cm",
                    (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

 
    if reference_clamp is not None and len(wires_detections) > 0:
        closest_wire = min(
            wires_detections,
            key=lambda w: calculate_distance(reference_clamp["center"], w["center"])[0]
        )

        dist_clamp_px, dist_clamp_in, dist_clamp_cm = calculate_distance(
            reference_clamp["center"], closest_wire["center"]
        )

        cv2.line(
            image_copy,
            (int(reference_clamp["center"][0]), int(reference_clamp["center"][1])),
            (int(closest_wire["center"][0]), int(closest_wire["center"][1])),
            (0, 255, 0), 3
        )

        mid_x = int((reference_clamp["center"][0] + closest_wire["center"][0]) / 2)
        mid_y = int((reference_clamp["center"][1] + closest_wire["center"][1]) / 2)

         cv2.rectangle(image_copy, (mid_x - 10, mid_y - 25), (mid_x + 440, mid_y + 10), (0, 0, 0), -1)
        cv2.putText(image_copy, f"R2-Closest: {dist_clamp_px:.0f}px | {dist_clamp_cm:.2f}cm",
                    (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    for h_det in h_detections:
        hx1, hy1, hx2, hy2 = h_det["bbox"]

        h_height_px = hy2 - hy1
        h_height_in = h_height_px / PIXELS_PER_INCH
        h_height_cm = h_height_in * INCH_TO_CM

        text_x = int(hx2) + 10
        text_y = int((hy1 + hy2) / 2)

        cv2.rectangle(image_copy, (text_x - 5, text_y - 20), (text_x + 260, text_y + 10), (0, 0, 0), -1)
        cv2.putText(image_copy, f"H Height: {h_height_px:.0f}px | {h_height_cm:.2f}cm",
                    (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)



    for det in all_detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cls_name = det["class"].strip().lower()

        if cls_name in CLASS_COLORS:
            color = CLASS_COLORS[cls_name]
        else:
            color = (200, 200, 200)

        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)

        label = f'{det["class"].upper()} {det["confidence"]:.2f}'
        cv2.putText(image_copy, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if shortest_wire is not None and det["center"] == shortest_wire["center"]:
            cv2.putText(image_copy, "SHORTEST", (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        if longest_wire is not None and det["center"] == longest_wire["center"]:
            cv2.putText(image_copy, "LONGEST", (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


    info_text = (
        f"Frame: {width}x{height}"
        f" | All: {len(all_detections)}"
        f" | Wires: {len(wires_detections)}"
        f" | R2: {len(clamps_R2_detections)}"
    )

    cv2.rectangle(image_copy, (10, 10), (900, 45), (0, 0, 0), -1)
    cv2.putText(image_copy, info_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image_copy



print("\n🎥 بدء معالجة الفيديو...")
start_time = time.time()

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    processed_frame = process_image(frame)

    out.write(processed_frame)

    if frame_count % 30 == 0 or frame_count == total_frames:
        print(f"⏳ تمت معالجة {frame_count} / {total_frames} فريم")

cap.release()
out.release()

elapsed_time = time.time() - start_time

print("\n✅ اكتملت معالجة الفيديو!")
print(f"⏱️ الوقت المستغرق: {elapsed_time:.2f} ثانية")
print(f"🎥 تم حفظ الفيديو في: {OUTPUT_PATH}")



print("\n⬇️ جاري تحميل الفيديو الناتج...")
files.download(OUTPUT_PATH)
