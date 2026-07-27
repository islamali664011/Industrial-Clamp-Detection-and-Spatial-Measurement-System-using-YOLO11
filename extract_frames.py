import cv2
from google.colab.patches import cv2_imshow

video_path = "/content/192.168.1.101_01_2026051116544357.mp4"

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 25

time_ranges = [
    (1 * 60 + 1,  1 * 60 + 3),   # 1:01 → 1:03
    (1 * 60 + 50, 1 * 60 + 52),  # 1:50 → 1:52
    (2 * 60 + 31, 2 * 60 + 32),  # 2:31 → 2:32
    (3 * 60 + 21, 3 * 60 + 23),  # 3:21 → 3:23
]

shot_count = 0

for start_time, end_time in time_ranges:

    for sec in range(start_time, end_time + 1):

        # 9 فريمات داخل كل ثانية
        for offset in [i / 9 for i in range(9)]:

            time_point = sec + offset
            frame_number = int(time_point * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            ret, frame = cap.read()

            if not ret:
                continue

            minutes = int(time_point // 60)
            seconds = int(time_point % 60)
            milliseconds = int((time_point % 1) * 1000)

            print(
                f"Frame #{shot_count+1} at "
                f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
            )

            cv2_imshow(frame)

            shot_count += 1

cap.release()

print(f"\nTotal Frames Extracted: {shot_count}")
