import fiftyone as fo
import sys
from fiftyone.core.dataset import delete_dataset, dataset_exists, list_datasets

dataset = fo.load_dataset('YOLOv8x_BB426_760_t31_b')
session = fo.launch_app(dataset, 
        # address="0.0.0.0", 
        port=5151,
        remote=True, 
        )
session.wait()
