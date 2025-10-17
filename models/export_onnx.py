from ultralytics import YOLO

# Load the YOLOv11n model
model = YOLO(r"C:\Users\dylan\Desktop\YOLOs-CPP-main\models\Model_6n.pt")

# Export the model to ONNX format
model.export(format="onnx")