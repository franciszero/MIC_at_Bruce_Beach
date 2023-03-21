from sahi.models import detectron2
from sahi.utils.cv import read_image, visualize_object_predictions
from sahi.predict import get_sliced_prediction

detection_model = detectron2.Detectron2DetectionModel(model_path='e2edet_final.pth')

image = read_image('../../data/images/Picture1.png')

result = get_sliced_prediction(
    image,
    detection_model,
    slice_height=2048,
    slice_width=2048,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    verbose=2,
)

visualization_result = visualize_object_predictions(
    image,
    object_prediction_list=result.object_prediction_list,
    output_dir="",
    file_name="",
)

result.export_visuals(export_dir="./Picture1_visual.jpg")

# # Jupyter Notebook only
# from IPython.display import Image
# Image("./Picture1_visual.png")
