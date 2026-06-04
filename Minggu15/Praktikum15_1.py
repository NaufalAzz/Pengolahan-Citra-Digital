import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
from collections import Counter

def praktikum_7_1():
    print("COMPUTER VISION: OBJECT DETECTION WITH YOLOv8")
    print("=" * 50)

    # Download sample image
    def download_sample_image():
        print("Mengunduh gambar sampel...")

        url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg"

        response = requests.get(url)
        img = Image.open(BytesIO(response.content))

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    try:
        # Load YOLOv8 Nano Model
        print("Memuat model YOLOv8n...")

        model = YOLO("yolov8n.pt")

        # Download image
        image = download_sample_image()

        # Detection
        print("Melakukan deteksi objek...")

        results = model(image)

        result_image = image.copy()

        detected_objects = []

        # Ambil hasil deteksi
        for result in results:

            boxes = result.boxes

            for box in boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                confidence = float(box.conf[0])

                class_id = int(box.cls[0])

                class_name = model.names[class_id]

                detected_objects.append(class_name)

                label = f"{class_name}: {confidence:.2f}"

                # Bounding box
                cv2.rectangle(
                    result_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Label
                cv2.putText(
                    result_image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        num_detections = len(detected_objects)

        # Tampilkan hasil
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title("Gambar Asli")
        ax1.axis("off")

        ax2.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        ax2.set_title(f"Hasil Deteksi ({num_detections} objek)")
        ax2.axis("off")

        plt.tight_layout()
        plt.show()

        # Statistik Deteksi
        print("\nHASIL DETEKSI")
        print("-" * 40)

        print(f"Total objek terdeteksi: {num_detections}")

        counts = Counter(detected_objects)

        print("\nJumlah objek berdasarkan kelas:")

        for obj_name, count in counts.items():
            print(f"- {obj_name}: {count}")

        return result_image, num_detections

    except Exception as e:
        print(f"\nTerjadi error: {e}")
        return None, 0


if __name__ == "__main__":

    result_image, detections = praktikum_7_1()

    print("\nProgram selesai.")