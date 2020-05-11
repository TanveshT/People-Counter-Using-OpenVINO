"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

import numpy as np

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    #Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def process_outputs(frame, outputs, width, height, threshold):
    
    manhDX = 0
    manhDY = 0
    count = 0
    
    frameMidX = width//2
    frameMidY = height//2
    
    midX = 0
    midY = 0
    
    for box in outputs[0][0]:
        conf = box[2]
        if conf >= threshold:
            count += 1
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
   
            midX = (xmax + xmin)//2
            midY = (ymax + ymin)//2
       
    flag = False
    
    euclid_d = (((384 - midX)**2) + ((216 - midY)**2)) ** 0.5
    
    
    if midX != 0 or midY != 0:
        if euclid_d > 100:
            flag = True

            
    cv2.putText(frame, str(euclid_d), (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)
        #cv2.putText(frame, str(width) + " " + str(height), (40, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), thickness=2)
        
    return frame, count, flag

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    
    frame_count = 0 
    #A flag to check if image or not
    single_image_mode = False
    
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    # Load the model through `infer_network` ###
    infer_network.load_model(model = args.model , device = args.device, cpu_extension = args.cpu_extension)

    #Handle the input stream ###
    if args.input == 'CAM':
        stream = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.png'):
        single_input_image = True
        stream = args.input
    else:
        stream = args.input
    
    capture = cv2.VideoCapture(stream)

    width = int(capture.get(3))
    height = int(capture.get(4))

    model_input_shape = infer_network.get_input_shape()

    #Initializing necessary variables for calculations
    total_people = 0
    people_in_last_frame = 0
    start_time = time.time()
    duration = None
    people_in_frame = 0
    thres = 0.45
    current_frame_request_id = 0
    next_frame_request_id = 1
    prev_flag = False
    
    if not capture.isOpened():
        exit()
    
    _, current_frame = capture.read()
    
    processed_frame = cv2.resize(current_frame, (model_input_shape[3], model_input_shape[2]))
    processed_frame = processed_frame.transpose((2,0,1))
    processed_frame = processed_frame.reshape(1,*processed_frame.shape)
    
    executable_net = infer_network.exec_net(image = processed_frame, request_id = current_frame_request_id)
    
    #Loop until stream is over ###
    while capture.isOpened():

        #Read from the video capture ###
        flag, next_frame = capture.read()

        if not flag:
            break
        
        #Needed with cv2.imshow() method
        key_pressed = cv2.waitKey(60)

        #Pre-process the image as needed ###
        processed_frame = cv2.resize(next_frame, (model_input_shape[3], model_input_shape[2]))
        processed_frame = processed_frame.transpose((2,0,1))
        processed_frame = processed_frame.reshape(1,*processed_frame.shape)

        #Start asynchronous inference for specified request ###
        executable_net = infer_network.exec_net(image = processed_frame, request_id = next_frame_request_id)
       
        # Wait for the result ###
        if infer_network.wait(request_id = current_frame_request_id) == 0:
        
            #Get the results of the inference request ###
            outputs = infer_network.get_output(request_id = current_frame_request_id)

            
            #Extract any desired stats from the results ###
            frame, people_in_frame, cur_flag = process_outputs(current_frame, outputs, width, height, thres)
            
            #cv2.imwrite('output',out_frame)
            current_time = time.time()
            cv2.putText(frame, 'Inference Time: ' + "%.2f" % (current_time - start_time), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, str(frame_count), (30,210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            #cv2.imshow('Output', out_frame) 

            #Calculate and send relevant information on ###
            if people_in_frame > people_in_last_frame and not prev_flag:
                total_people += (people_in_frame - people_in_last_frame)
                new_people_time = time.time()
                client.publish("person", json.dumps({"total": total_people}))

            if people_in_frame < people_in_last_frame:
                duration = time.time() - new_people_time
                client.publish("person/duration", json.dumps({"duration": duration}))

            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            client.publish("person", json.dumps({"count":people_in_frame}))
            ### Topic "person/duration": key of "duration" ###
           
        
            people_in_last_frame = people_in_frame
            prev_flag = cur_flag
        
            current_frame = next_frame
            current_frame_request_id, next_frame_request_id = next_frame_request_id, current_frame_request_id
        
        if key_pressed == 27:
           break

        frame_count += 1
        #Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        #Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite('out.jpg', frame)


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()