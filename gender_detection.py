import cv2
import os
from deepface import DeepFace


image_files = [
    "sabih.jpg",
    "ahmed.jpg",
    "abd.jpg",
    "sufyan.jpg",
    "cat.jpg",
    "dure.jpg",
    "woman1.jpg",
]


for image_file in image_files:
    image_path = os.path.join(os.getcwd(), image_file)

    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Could not read image {image_file}")
        continue

    results = DeepFace.analyze(frame, actions=["gender"], enforce_detection=False)

    for result in results:
        x, y, w, h = (
            result["region"]["x"],
            result["region"]["y"],
            result["region"]["w"],
            result["region"]["h"],
        )
        dominant_gender = result["dominant_gender"]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.putText(
            frame,
            dominant_gender, # shown in detection
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (36, 255, 12),
            2,
        )

    cv2.imshow("Gender Detection", frame)
    cv2.waitKey(0)
cv2.destroyAllWindows()


# Project is running