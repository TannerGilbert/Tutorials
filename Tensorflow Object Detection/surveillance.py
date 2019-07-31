import argparse
from PIL import Image
import cv2
import numpy as np
import os
import pandas as pd
import datetime
import tensorflow as tf
import sys
sys.path.append("C:/Users/Gilbert/Desktop/Programming/TensorflowModels/research/slim")
sys.path.append("C:/Users/Gilbert/Desktop/Programming/TensorflowModels/research/object_detection")
from utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops
from utils import label_map_util


def run_inference_for_single_image(image, sess, tensor_dict):
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def load_model(PATH_TO_FROZEN_GRAPH: str, PATH_TO_LABELS: str):
    """
    Load model from frozen inference graph
    """
    # Load frozen inference graph into memory
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Loading labelmap
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    return graph, category_index


def run_surveillance(model_path, labelmap_path, show_video_steam, label_to_look_for, output_directory, threshold):
    # Load model
    graph, category_index = load_model(model_path, labelmap_path)
    # Initialize Video Capture
    cap = cv2.VideoCapture(0)
    # Create output directory if not already created
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(output_directory+'/images', exist_ok=True)

    if os.path.exists(output_directory+'/results.csv'):
        df = pd.read_csv(output_directory+'/results.csv')
    else:
        df = pd.DataFrame(columns=['timestamp', 'img_path'])

    # Open detection graph
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            while True:
                try:
                    # Read frame from video
                    ret, image_np = cap.read()

                    # Copy image for later
                    image_show = np.copy(image_np)

                    image_height, image_width, _ = image_np.shape
                    # Actual detection.
                    output_dict = run_inference_for_single_image(image_np, sess, tensor_dict)

                    if show_video_steam:
                        # Visualize results
                        vis_util.visualize_boxes_and_labels_on_image_array(
                            image_np,
                            output_dict['detection_boxes'],
                            output_dict['detection_classes'],
                            output_dict['detection_scores'],
                            category_index,
                            instance_masks=output_dict.get('detection_masks'),
                            use_normalized_coordinates=True,
                            line_thickness=8)
                        cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            cap.release()
                            cv2.destroyAllWindows()
                            break

                    # Get data(label, xmin, ymin, xmax, ymax)
                    output = []
                    for index, score in enumerate(output_dict['detection_scores']):
                            if score < threshold:
                                continue
                            label = category_index[output_dict['detection_classes'][index]]['name']
                            ymin, xmin, ymax, xmax = output_dict['detection_boxes'][index]
                            output.append((label, int(xmin * image_width), int(ymin * image_height), int(xmax * image_width), int(ymax * image_height)))

                    # Save incident (could be extended to send a email or something)
                    for l, x_min, y_min, x_max, y_max in output:
                        if l == label_to_look_for:
                            array = cv2.cvtColor(np.array(image_show), cv2.COLOR_RGB2BGR)
                            image = Image.fromarray(array)
                            cropped_img = image.crop((x_min, y_min, x_max, y_max))
                            file_path = output_directory+'/images/'+str(len(df))+'.jpg'
                            cropped_img.save(file_path, "JPEG", icc_profile=cropped_img.info.get('icc_profile'))
                            df.loc[len(df)] = [datetime.datetime.now(), file_path]
                            df.to_csv(output_directory+'/results.csv', index=None)

                except Exception as e:
                    print(e)
                    cap.release()
                    cv2.destroyAllWindows()
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Surveillance System')
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to the frozen inference graph')
    parser.add_argument('-l', '--labelmap', type=str, required=True, help='Path to labelmap')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Threshold for bounding boxes')
    parser.add_argument('-s', '--show', default=True, action='store_true', help='Show window')
    parser.add_argument('-la', '--label', default='person', type=str, help='Label name to detect')
    parser.add_argument('-o', '--output_directory', default='results', type=str, help='Directory for the outputs')
    args = parser.parse_args()
    run_surveillance(args.model_path, args.labelmap, args.show, args.label, args.output_directory, args.threshold)
