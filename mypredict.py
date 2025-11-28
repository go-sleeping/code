from ultralytics import YOLO

model = YOLO(r"yolo11n.pt")
model.predict(
    source=r"H:\adadaw",
    save=True,
    show=False,
)