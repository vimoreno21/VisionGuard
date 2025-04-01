import cv2
from mtcnn import MTCNN  # pip install mtcnn
from deepface import DeepFace
import os
import matplotlib.pyplot as plt

SAVE_DIR = "./database/captured_images"
Vic_dir = "./database/victoria"
Other_dir = "./database/other"
Group_dir = "./database/group"

SAVE_PATH = os.path.join(Vic_dir, "10.jpeg")

# Load an image (or capture from RTSP)
img = cv2.imread(SAVE_PATH)  
if img is None:
    print("❌ Failed to load image")
    exit()

# Detect faces
detector = MTCNN()
results = detector.detect_faces(img)
if not results:
    print("❌ No face detected.")
    exit()

# Take the first face box
x, y, w, h = results[0]['box']

# Expand bounding box slightly so it's not so tight
margin = 20
height, width, _ = img.shape

# Adjust x, y by margin
x_new = max(0, x - margin)
y_new = max(0, y - margin)

# Expand w, h
w_new = min(width - x_new, w + margin * 2)
h_new = min(height - y_new, h + margin * 2)

# Crop with the expanded box
cropped_face = img[y_new:y_new + h_new, x_new:x_new + w_new]
print("✅ Cropped face shape:", cropped_face.shape)

# Save the cropped face for debugging
cv2.imwrite("cropped_face.jpg", cropped_face)
print("✅ Cropped face saved as 'cropped_face.jpg'! Open it manually if needed.")

# Show the cropped face using Matplotlib
plt.imshow(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
plt.axis("off")  # Hide axes
plt.title("Cropped Face")
plt.show(block=True)  # Force it to wait before continuing

# Run DeepFace analysis on the cropped face
cropped_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
for db_name in ["victoria", "other", "group"]:
    db_path = f"./database/{db_name}"

    try:
        results = DeepFace.find(
            img_path=SAVE_PATH,
            db_path=db_path,
            model_name="VGG-Face",
            distance_metric="cosine",
            verbose=0
        )

        if results and not results[0].empty:
            df = results[0]
            best_match = df.iloc[0]
            distance = best_match["distance"]

            # set a threshold for matching
            if distance < 0.5:
                print(f"✅ Match found in {db_name} database!")
                print(f"🔹 Matched with: {best_match['identity']} (Distance: {distance:.3f})")
            else:
                print(f"⚠️ No strong match in {db_name} (Closest match distance: {distance:.3f})")
        else:
            print(f"❌ No match found in {db_name} database.")

    except Exception as e:
        print(f"Error searching in {db_name} database: {e}")

    print("\n\n")
