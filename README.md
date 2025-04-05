
---

![Animal Detection Example](../images/animal_detection_example.png)
*Live detection of animals like dogs and cats using MobileNet SSD.*

---

## 🐶 Real-Time Animal Detection (MobileNet SSD)

```python
import cv2
import numpy as np

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt',
                               'MobileNetSSD_deploy.caffemodel')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]

            if label in ['cat', 'dog', 'bird', 'horse', 'cow', 'sheep']:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {round(confidence*100, 1)}%", 
                            (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Animal Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 🧠 What's Happening?

- **Face Detection:** Uses Haar Cascades, a simple and fast algorithm ideal for frontal face detection.
- **Object Detection:** MobileNet SSD is a deep learning model trained on the COCO dataset, capable of detecting multiple object classes with high accuracy.
- **Real-Time Feedback:** Frames are processed in real-time from the webcam using OpenCV’s `VideoCapture`.

---


## 🎯 Conclusion

With just a few lines of Python and some pre-trained models, you can bring powerful real-time vision to your own desktop. Try extending it with more categories, facial recognition, or even fun filters using AR overlays.

Let us know what you build!

