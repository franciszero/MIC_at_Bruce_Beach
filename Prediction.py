#
import glob
from ultralyticsplus import YOLO


if __name__ == '__main__':
    model = YOLO("./models/YOLOv8/weights/YOLOv8x_BB426_760_t31_b.pt")
    model.overrides['conf'] = 0.5  # NMS confidence threshold
    model.overrides['iou'] = 0.5  # NMS IoU threshold
    model.overrides['agnostic_nms'] = False  # NMS class-agnostic
    model.overrides['max_det'] = 1000  # maximum number of detections per image

    files = glob.glob("./data/images/*")
    for file in files:
        results = model.predict(file, stream=True, verbose=False)
        for result in results:
            print("There are %d people in %s." % (len(result.boxes), file.rsplit("/", 1)[1]))
