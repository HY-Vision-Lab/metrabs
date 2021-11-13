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
    parser.add_argument('--camera_id', type=int, default=0, help="camera id")
    parser.add_argument('--device', type=str, default='gpu', help="gpu/cpu")
    parser.add_argument('--skeleton', type=str, default='smpl_24', help="skeleton type (smpl24, smpl+head_30)")

    args = parser.parse_args()

    print(tf.config.list_physical_devices('GPU'))
    device_ = '/' + args.device + ':0'
    with tf.device(device_):

        ''' settings '''
        fps = 30.0
        time_period = 1.0 / fps
       
        # Load the model
        print('Loading the tensorflow model...')
        model = tf.saved_model.load(download_model('metrabs_mob3l_y4t'))
        print('Model loading complete')
        skeleton = args.skeleton   # to show all-trained skeleton
        joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
        joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()

        FLIP_IMAGE_LR = False

        # display webcam window
        cv2.namedWindow('Video', cv2.WINDOW_AUTOSIZE)

        # initialize the poseviz window
        viz = poseviz.PoseViz(joint_names, 
                              joint_edges, 
                              snap_to_cam_on_scene_change=False, 
                              show_field_of_view=False, 
                              viz_fps=fps)

        # open video stream
        cap = cv2.VideoCapture(args.camera_id)
        prev_time = 0.0

        # start the video fetching loop
        while True:
            ret, frame_bgr = cap.read()
            # time_duration = time.time() - prev_time

            if (ret is True):
            # if (ret is True) and (time_duration > np.max([time_period - 0.01, 0.0])):
                if (FLIP_IMAGE_LR):
                    frame_bgr=cv2.flip(frame_bgr, 1)

                # obtain rgb frame
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


                # predict the 3D pose
                pred = model.detect_poses(frame_rgb, 
                                          skeleton=skeleton, 
                                          default_fov_degrees=55, 
                                          detector_threshold=0.1, 
                                          max_detections=1)
                
                # obtain camera position
                camera = poseviz.Camera.from_fov(55, frame_rgb.shape[:2])
                # update the 3D visualization
                viz.update(frame_rgb, pred['boxes'], pred['poses3d'], camera)                                          

                # print("Elapsed time (s): %.3f\n" % time_duration)
                prev_time = time.time()

                # display current video image
                cv2.imshow("Video", frame_bgr)                

            if(cv2.waitKey(1) & 0xFF) == ord('f'):
                # Toggle image flip                
                FLIP_IMAGE_LR = True if FLIP_IMAGE_LR is False else False

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                # get out of the loop
                viz.close()
                break

        # release the cap and qit
        cap.release()
        cv2.destroyAllWindows()


def frames_from_webcam(args):
    cap = cv2.VideoCapture(args.camera_id)

    frame_bgr = cap.read()[1]
    while (frame_bgr) is not None:
        yield frame_bgr[..., ::-1]
        frame_bgr = cap.read()[1]



def download_model(model_type):
    model_path = os.path.join(os.path.dirname("./{model}.zip".format(model=model_type)), model_type)
    return model_path


if __name__ == '__main__':
    main()