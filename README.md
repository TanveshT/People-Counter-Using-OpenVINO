# People Counter Application at the Edge using the Intel's OpenVINO toolkit.

This is my implementation of the project.


## To Run

```python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm```