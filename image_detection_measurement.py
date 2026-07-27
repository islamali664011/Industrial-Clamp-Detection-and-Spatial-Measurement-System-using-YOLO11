
import cv2
import numpy as np
import math
import os
import time
from ultralytics import YOLO
from google.colab.patches import cv2_imshow

MODEL_PATH  = r"/content/runs/detect/Head_Detection/yolo11_1280/weights/best.pt"
IMAGE_PATH  = r"/content/Head1.jpg"      
OUTPUT_PATH = r"/content/output_image_with_measurements.jpg"

 
PIXELS_PER_INCH = 96
INCH_TO_CM      = 2.54


print("  جاري تحميل الموديل...")
model = YOLO(MODEL_PATH)
print(" تم تحميل الموديل بنجاح")


if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f" الصورة غير موجودة في: {IMAGE_PATH}")

frame = cv2.imread(IMAGE_PATH)
if frame is None:
    raise RuntimeError(f" فشل في قراءة الصورة: {IMAGE_PATH}")

height, width = frame.shape[:2]

print(f"\nمعلومات الصورة:")
print(f"  الأبعاد: {width} x {height}")


def calculate_distance(point1, point2, pixels_per_inch=PIXELS_PER_INCH):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    distance_pixels = math.hypot(dx, dy)
    distance_inches = distance_pixels / pixels_per_inch
    distance_cm     = distance_inches * INCH_TO_CM
    return distance_pixels, distance_inches, distance_cm


CLASS_COLORS = {
    'r1':    (255, 0, 0),       
    'r2':    (0, 255, 255),       
    'h':     (255, 0, 255),       
    'wire':  (255, 165, 0),       
}

 
def process_image(image):
    image_copy = image.copy()

     results = model.predict(source=image, conf=0.40, save=False, verbose=False)

     all_detections = []

    wires_detections     = []
    clamps_R2_detections = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id     = int(box.cls[0].cpu().numpy())
            cls_name   = result.names[cls_id]
            confidence = float(box.conf[0].cpu().numpy())

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            detection = {
                'class':      cls_name,
                'bbox':       (x1, y1, x2, y2),
                'center':     (cx, cy),
                'confidence': confidence
            }

             all_detections.append(detection)

             cls_lower = cls_name.strip().lower()
            if cls_lower in ['wire', 'wires', 'سلك', 'اسلاك']:
                wires_detections.append(detection)
            elif cls_lower == 'r2':
                clamps_R2_detections.append(detection)

 
    shortest_wire = None
    longest_wire  = None
    shortest_data = None
    longest_data  = None
    reference_clamp = None

    if clamps_R2_detections and len(wires_detections) >= 2:
        reference_clamp = max(clamps_R2_detections, key=lambda c: c['confidence'])

        wire_distances = []
        for wire in wires_detections:
            dist_px, dist_in, dist_cm = calculate_distance(reference_clamp['center'], wire['center'])
            wire_distances.append({
                'wire':        wire,
                'distance_px': dist_px,
                'distance_in': dist_in,
                'distance_cm': dist_cm
            })

        wire_distances_sorted = sorted(wire_distances, key=lambda w: w['distance_px'])

        shortest_data = wire_distances_sorted[0]
        shortest_wire = shortest_data['wire']

        longest_data = wire_distances_sorted[-1]
        longest_wire = longest_data['wire']

         cv2.line(image_copy,
                 (int(reference_clamp['center'][0]), int(reference_clamp['center'][1])),
                 (int(shortest_wire['center'][0]),   int(shortest_wire['center'][1])),
                 (0, 255, 0), 3)

        mid_x = int((reference_clamp['center'][0] + shortest_wire['center'][0]) / 2)
        mid_y = int((reference_clamp['center'][1] + shortest_wire['center'][1]) / 2)
        text = f"R2->Shortest: {shortest_data['distance_cm']:.2f} cm"
        cv2.rectangle(image_copy, (mid_x - 10, mid_y - 22), (mid_x + 290, mid_y + 5), (0, 0, 0), -1)
        cv2.putText(image_copy, text, (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

         cv2.line(image_copy,
                 (int(longest_wire['center'][0]),  int(longest_wire['center'][1])),
                 (int(shortest_wire['center'][0]), int(shortest_wire['center'][1])),
                 (0, 0, 255), 3)

        mid_x = int((longest_wire['center'][0] + shortest_wire['center'][0]) / 2)
        mid_y = int((longest_wire['center'][1] + shortest_wire['center'][1]) / 2)
        text = f"Long-Short: {longest_data['distance_cm']:.2f} cm"
        cv2.rectangle(image_copy, (mid_x - 10, mid_y - 22), (mid_x + 270, mid_y + 5), (0, 0, 0), -1)
        cv2.putText(image_copy, text, (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    
    for det in all_detections:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        cls_name = det['class'].strip().lower()

        # تحديد اللون حسب الـ class
        if cls_name in CLASS_COLORS:
            color = CLASS_COLORS[cls_name]
        else:
            color = (200, 200, 200)      

        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)

        label = f"{det['class'].upper()} {det['confidence']:.2f}"
        cv2.putText(image_copy, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

         if shortest_wire and det['center'] == shortest_wire['center']:
            cv2.putText(image_copy, "SHORTEST", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        if longest_wire and det['center'] == longest_wire['center']:
            cv2.putText(image_copy, "LONGEST", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

 
    info_text = f"Image: {width}x{height} | All: {len(all_detections)} | Wires: {len(wires_detections)} | R2: {len(clamps_R2_detections)}"
    cv2.rectangle(image_copy, (10, 10), (750, 40), (0, 0, 0), -1)
    cv2.putText(image_copy, info_text, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image_copy

 
print("\n بدء معالجة الصورة...")
start_time = time.time()

# معالجة الصورة
processed_image = process_image(frame)

# حفظ وعرض الصورة
cv2.imwrite(OUTPUT_PATH, processed_image)
cv2_imshow(processed_image)

elapsed_time = time.time() - start_time
print(f"\n اكتملت معالجة الصورة!")
print(f الوقت المستغرق: {elapsed_time:.2f} ثانية")
print(f  تم حفظ الصورة في: {OUTPUT_PATH}")
