#!/usr/bin/env python3

from BlazeposeRenderer import BlazeposeRenderer
import argparse
import mediapipe_utils

from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.osc_bundle_builder import OscBundleBuilder
from pythonosc.osc_bundle import OscBundle
from time import time


#Get a copy of the keypoints dictionary
KP_DICT = mediapipe_utils.KEYPOINT_DICT

# Initialize the OSC client with the target IP address and port number
osc_client = udp_client.SimpleUDPClient("127.0.0.1", 12345)


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--edge', action="store_true",
                    help="Use Edge mode (postprocessing runs on the device)")
parser_tracker = parser.add_argument_group("Tracker arguments")                 
parser_tracker.add_argument('-i', '--input', type=str, default="rgb", 
                    help="'rgb' or 'rgb_laconic' or path to video/image file to use as input (default=%(default)s)")
parser_tracker.add_argument("--pd_m", type=str,
                    help="Path to an .blob file for pose detection model")
parser_tracker.add_argument("--lm_m", type=str,
                    help="Landmark model ('full' or 'lite' or 'heavy') or path to an .blob file")
parser_tracker.add_argument('-xyz', '--xyz', action="store_true", 
                    help="Get (x,y,z) coords of reference body keypoint in camera coord system (only for compatible devices)")
parser_tracker.add_argument('-c', '--crop', action="store_true", 
                    help="Center crop frames to a square shape before feeding pose detection model")
parser_tracker.add_argument('--no_smoothing', action="store_true", 
                    help="Disable smoothing filter")
parser_tracker.add_argument('-f', '--internal_fps', type=int, 
                    help="Fps of internal color camera. Too high value lower NN fps (default= depends on the model)")                    
parser_tracker.add_argument('--internal_frame_height', type=int, default=640,                                                                                    
                    help="Internal color camera frame height in pixels (default=%(default)i)")                    
parser_tracker.add_argument('-s', '--stats', action="store_true", 
                    help="Print some statistics at exit")
parser_tracker.add_argument('-t', '--trace', action="store_true", 
                    help="Print some debug messages")
parser_tracker.add_argument('--force_detection', action="store_true", 
                    help="Force person detection on every frame (never use landmarks from previous frame to determine ROI)")

parser_renderer = parser.add_argument_group("Renderer arguments")
parser_renderer.add_argument('-3', '--show_3d', choices=[None, "image", "world", "mixed"], default=None,
                    help="Display skeleton in 3d in a separate window. See README for description.")
parser_renderer.add_argument("-o","--output",
                    help="Path to output video file")
 

args = parser.parse_args()

if args.edge:
    from BlazeposeDepthaiEdge import BlazeposeDepthai
else:
    from BlazeposeDepthai import BlazeposeDepthai
tracker = BlazeposeDepthai(input_src=args.input, 
            pd_model=args.pd_m,
            lm_model=args.lm_m,
            smoothing=not args.no_smoothing,   
            xyz=args.xyz,            
            crop=args.crop,
            internal_fps=args.internal_fps,
            internal_frame_height=args.internal_frame_height,
            force_detection=args.force_detection,
            stats=True,
            trace=args.trace)   

renderer = BlazeposeRenderer(
                tracker, 
                show_3d=args.show_3d, 
                output=args.output)

while True:
    # Run blazepose on next frame
    frame, body = tracker.next_frame()
    if frame is None: break
    # Draw 2d skeleton
    frame = renderer.draw(frame, body)
    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        break

    if body is not None:
        print(body.norm_landmarks[KP_DICT["left_shoulder"]])

        # Initialize the OSC Bundle Builder
        bundle_builder = OscBundleBuilder(time())

        # Assume 'body' is an instance of Body that contains keypoints detected
        for keypoint_name, keypoint_index in KP_DICT.items():
            # Get the keypoint coordinates
            keypoint_coords = body.norm_landmarks[keypoint_index]

            # Initialize the OSC Message Builder
            msg_builder = OscMessageBuilder(address=f"/pose/{keypoint_name}")
            
            # Add the keypoint coordinates to the message
            msg_builder.add_arg(keypoint_coords[0])  # x coordinate
            msg_builder.add_arg(keypoint_coords[1])  # y coordinate
            msg_builder.add_arg(keypoint_coords[2])  # z coordinate (if available)

            # Build the message
            msg = msg_builder.build()

            # Add the message to the bundle
            bundle_builder.add_content(msg)

        # Build the bundle
        bundle = bundle_builder.build()

        # Send the bundle
        osc_client.send(bundle)

renderer.exit()
tracker.exit()