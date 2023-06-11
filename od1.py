import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')  # Suppress Matplotlib warnings

# Path to the pre-trained model and label map
pretrained_model_dir = "path/to/pretrained_model_directory"
label_map_path = "path/to/label_map.pbtxt"

# Path to your input image
image_path = "path/to/your/image.jpg"

# Load label map
category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

# Load the model
tf.keras.backend.clear_session()
configs = model_builder.ModelConfig(
    model_name='ssd_mobilenet_v2_fpnlite',
    is_training=False,
    use_depthwise=True
)
model = model_builder.build(configs=configs, model_checkpoint=pretrained_model_dir)
model.restore(pretrained_model_dir).expect_partial()


# Run object detection on input image
def run_inference(image_path):
    image_np = np.array(Image.open(image_path))
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    preprocessed_image, shapes = model.preprocess(input_tensor)
    prediction_dict = model.predict(preprocessed_image, shapes)
    detections = model.postprocess(prediction_dict, shapes)
    return detections


detections = run_inference(image_path)


# Visualize the detection results
def plot_detections(image_np, detections, category_index):
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        (detections['detection_classes'][0].numpy() + 1).astype(int),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0.3,
        agnostic_mode=False
    )
    plt.imshow(image_np_with_detections)
    plt.axis('off')
    plt.show()


# Visualize the results
image_np = np.array(Image.open(image_path))
plot_detections(image_np, detections, category_index)
