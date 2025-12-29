from ultralytics import YOLO

# Load a model
model = YOLO("model/best.pt")  # load a custom model

# Predict with the model
source = 'data/train/images/1747700043358_427b831e-c99c-eac5-355a-3772383dda37.jpg'  # predict on an image
results = model(source=source, conf=0.01)
# Access the results\
result_list = []
for result in results:
    for box in result.boxes:
        class_id = int(box.cls)
        result_list.append(model.names[class_id])
result.show()
print(box)