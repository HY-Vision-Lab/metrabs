#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import sys
import cv2
import tensorflow as tf
import time

import numpy as np

import poseviz

def main():
    parser = argparse.ArgumentParser(description="image")
    parser.add_argument('--cam', type=int, default=0, help="camera id")
    parser.add_argument('--device', type=str, default='gpu', help="gpu/cpu")
    args = parser.parse_args()

    print(tf.config.list_physical_devices('GPU'))
    device_ = '/' + args.device + ':0'
    with tf.device(device_):
        model = tf.saved_model.load(download_model('metrabs_mob3l_y4t'))
        skeleton = ''   # to show all-trained skeleton
        joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
        joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()
        viz = poseviz.PoseViz(joint_names, joint_edges, snap_to_cam_on_scene_change=False, show_field_of_view=False, viz_fps=30)
        fin = False
        for frame in frames_from_webcam(args):
            cam = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if (fin):
                cam = cv2.flip(cam, 1)
                frame = np.fliplr(frame)

            pred = model.detect_poses(
                frame, skeleton=skeleton, default_fov_degrees=55, detector_threshold=0.1, max_detections=1)


            camera = poseviz.Camera.from_fov(55, frame.shape[:2])
            viz.update(frame, pred['boxes'], pred['poses3d'], camera)

            cv2.namedWindow('Video', cv2.WINDOW_AUTOSIZE)

            cv2.imshow("Video", cam)

            if(cv2.waitKey(1) & 0xFF) == ord('o'):
                if fin == True: fin = False
                else: fin = True
            elif (cv2.waitKey(1) & 0xFF) == ord('q'):
                sys.exit(0)



def frames_from_webcam(args):
    cap = cv2.VideoCapture(args.cam)

    frame_bgr = cap.read()[1]
    while (frame_bgr) is not None:
        yield frame_bgr[..., ::-1]
        frame_bgr = cap.read()[1]



def download_model(model_type):
    model_path = os.path.join(os.path.dirname("./{model}.zip".format(model=model_type)), model_type)
    return model_path


if __name__ == '__main__':
    main()