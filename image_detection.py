from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

 model = YOLO("/content/runs/detect/Head_Detection/yolo11_1280/weights/best.pt")

 image_path = "/content/Head1.jpg"    

 results = model.predict(
    source=image_path,
    imgsz=1280,
    conf=0.25,
    save=True
)

 result_image = results[0].plot()

plt.figure(figsize=(10, 10))
plt.imshow(result_image[..., ::-1])   
plt.axis("off")
plt.show()
