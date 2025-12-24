from ultralytics import YOLO
# load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
# we use name = main for the GPU functionality
if __name__ == '__main__':

    model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

# use the model
   # results = model.train(data="config.yaml", epochs=150)  # train the model
# one epoch is one complete pass
# through the entire training dataset during the
# training of a machine learning model, allowing it to update i ts parameters and improve its performance.

 # Train the model
    model.train(
        data=r'C:\Users\ontar\PycharmProjects\ObjectDetection\classification_dataset',
        epochs=50,
        imgsz=224,
        batch=16
    )