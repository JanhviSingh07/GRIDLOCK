from ultralytics import YOLO

# Tera trained model load karo
model = YOLO("model/best.pt")

print("✅ Model loaded successfully!")
print("Classes:", model.names)