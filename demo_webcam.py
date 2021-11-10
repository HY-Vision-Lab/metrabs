#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import tensorflow as tf
import time
import pyrealsense2 as rs
import numpy as np

import poseviz
FPS = 30

def main():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 60)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

    # Start streaming
    pipeline.start(config)

    print(tf.config.list_physical_devices('GPU'))
    with tf.device('/gpu:0'):
        model = tf.saved_model.load(download_model('metrabs_mob3l_y4t'))
        skeleton = 'smpl+head_30'
        joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
        joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()
        viz = poseviz.PoseViz(joint_names, joint_edges, snap_to_cam_on_scene_change=False, show_field_of_view=False, viz_fps=60)
        i = 0
        # time_1 = time.time()
        # for frame in pipeline.wait_for_frames(): # frames_from_webcam()
        while True:
            frame = pipeline.wait_for_frames()
            color_frame = frame.get_color_frame()
            depth_frame = frame.get_depth_frame()

            if not color_frame or not depth_frame:
                continue
            # print(depth_frame)
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_BONE)
            depth_colormap = np.asanyarray(depth_colormap)

            # time_1 = time.time()
            pred = model.detect_poses(
                color_image, skeleton=skeleton, default_fov_degrees=55, detector_threshold=0.05, max_detections=1)
            # time_2 = time.time(); print(time_2 - time_1)
            camera = poseviz.Camera.from_fov(55, depth_image.shape[:2])

            # if (pred['boxes'].shape[0] == 1):
            viz.update(depth_image, pred['boxes'], pred['poses3d'], camera)
            # images = np.hstack((color_image, color_image))
            cv2.namedWindow('Realsense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Realsense", color_image)
            # time_3 = time.time()
            #print("\t", time_3 - time_2, sep = '')
            # click "q" if you want to terminate this.
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
            # i += 1
            #
            # if (i >= FPS):
            #     print(time.time() - time_1)
            # i = i % (FPS);




def frames_from_webcam():
    '''
    # global FPS

    prev_time = 0
    # "C:/Users/HYU/Documents/metrabs/img/dance.mp4"
    # 0: built-in camera(if laptop)
    # 1: logitech camera
    # 2: intelrealsense camera
    cap = cv2.VideoCapture(2)
    cap.set(3, 640)
    cap.set(4, 480)
    ret, frame = cap.read()

    current_time = time.time() - prev_time
    while (ret is True) and (current_time > 1./FPS):
        ret, frame = cap.read()
        current_time = time.time() - prev_time
        yield frame[..., ::-1]
        check = frame_bgr = cap.read()[1]

    print(len(frame))
    '''
    cap = cv2.VideoCapture(2)
    frame_bgr = cap.read()[1]
    while (frame_bgr) is not None:
        yield frame_bgr[..., ::-1]
        frame_bgr = cap.read()[1]


def download_model(model_type):
    server_prefix = 'https://omnomnom.vision.rwth-aachen.de/data/metrabs'
    model_zippath = tf.keras.utils.get_file(
        fname=f'C:/Users/HYU/Documents/metrabs/{model_type}.zip',
        origin=f'{server_prefix}/{model_type}.zip',
        extract=True, cache_subdir='models')
    model_path = os.path.join(os.path.dirname(model_zippath), model_type)
    return model_path


if __name__ == '__main__':
    main()
