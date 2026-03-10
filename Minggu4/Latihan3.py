# LATIHAN 3: REAL-TIME VIDEO ENHANCEMENT
import cv2
import numpy as np
import time

print("=== LATIHAN 3: REAL-TIME VIDEO ENHANCEMENT ===")

# CLASS REAL TIME ENHANCEMENT

class RealTimeEnhancement:
    def __init__(self, target_fps=30):
        self.target_fps = target_fps
        self.history_buffer = []
        self.max_buffer_size = 5
        # CLAHE untuk adaptive enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))


    def enhance_frame(self, frame, enhancement_type='adaptive'):
        """
        Enhance single frame with real-time constraints
        Parameters:
        frame: Input video frame
        enhancement_type: Type of enhancement
        Returns:
        Enhanced frame
        """
        # Convert ke grayscale agar komputasi lebih ringan
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Enhancement Method

        if enhancement_type == 'adaptive':
            # CLAHE untuk meningkatkan kontras
            enhanced = self.clahe.apply(gray)
        elif enhancement_type == 'histogram':
            enhanced = cv2.equalizeHist(gray)
        elif enhancement_type == 'gamma':
            gamma = 0.8
            normalized = gray / 255.0
            gamma_corrected = np.power(normalized, gamma)
            enhanced = np.uint8(gamma_corrected * 255)
        else:
            enhanced = gray

        # Temporal Consistency

        # Simpan beberapa frame sebelumnya
        self.history_buffer.append(enhanced)
        if len(self.history_buffer) > self.max_buffer_size:
            self.history_buffer.pop(0)
        # Rata-rata frame untuk mengurangi flicker
        smoothed = np.mean(self.history_buffer, axis=0).astype(np.uint8)

        return smoothed

# MAIN PROGRAM
cap = cv2.VideoCapture(0)
enhancer = RealTimeEnhancement(target_fps=30)
prev_time = time.time()
print("Tekan ESC untuk keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Enhance frame
    enhanced_frame = enhancer.enhance_frame(frame, 'adaptive')
    # Hitung FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    # Tampilkan FPS pada frame
    cv2.putText(frame,
                f"FPS: {int(fps)}",
                (10,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    # Tampilkan video
    cv2.imshow("Original Video", frame)
    cv2.imshow("Enhanced Video", enhanced_frame)
    # Tekan ESC untuk keluar
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Program selesai.")