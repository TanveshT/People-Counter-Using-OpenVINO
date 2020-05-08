# Project Write-Up

## Explaining Custom Layers

Custom layers are the layers which are not included into the OpenVINO framework as mentioned in the document https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html.

The process behind converting custom layers involves registering those custom layers as extensions to the model optimizer. The process differs from framework to framework. Like for Tensorflow the unsupported subgraph can be replaced and for Caffe registering these layers as Custom would do.

Some of the potential reasons for handling custom layers are that without handling them the model optimizer won't be able to recognize them and hence Intermediate Representation (IR format) can't be obtained. The OpenVINO framework has support for many layers already but there may be few for which it isn't supported so at that time either a Inference extension or else while conversion of the model a extension should be added to model optimizer.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were by running the model on same video and checking the following things:

  - How accurately the people are getting detected in the frame pre and post conversion i.e if a person is getting left out of the detection even after lowering the threshold then it is not good
  - Secondly checking the speed of the inference on the basis how fast the frame is getting returned.

The difference between model accuracy pre- and post-conversion was dropped as I was reduced certainly while using the MobileNet SSD models but when I tried the Faster RCNN Inception v2 and SSD Resnet the accuracy was certain but the inference time was affected a lot.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are:

- A retail store onwer can track the number of people everyday which are visiting the store and also see at which section of the store they are visiting the most and according to it the owner can perform analysis on the products customers are interested in and making various business related decisions accordingly.
- The counter app can also be used for Prison cells where one can monitor the number of prisoners in a particular location and if it is above a certain number it could trigger a alarm for the security.
- A use case also can be attaching a ReID model on top of the counting app but this will certainly violate some hardware restrictions these can be used for counting the unique number of people arrived at a certain location.

Each of these use cases would be useful because these use cases need a way of monitoring the people and the applcations suffices that purpose.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

- Ligthing will certaining cause a effect in inference as the model won't be able to predict the person's features if it matches the background of the image.
- Model Acccuracy: A Edge applcation needs a good accuracy if it is deployed in a certain situation if critical decision process is needed. In the prison cell situation it is needed to have accuracy of the application.
- Camera Focal Length/ Image Size: While testing the application on my local devive I certainly saw a inference change when the focal length wasn't accurately achieved. The image size will affect the inference if it is a resolution image but it increases the accuracy of the model as more crisp image = more accurately the model will be able to detect features.
  
## Model Research

In investigating potential people counter models, I tried each of the following three models:

- Model 1: SSD_RESNET50_V1_FPN_COCO
  - http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
  - I converted the model to an Intermediate Representation with the following arguments </br>
  ```python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/pipeline.config --tensorflow_object_detection_api_pipeline_config --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --reverse_input_channel```
  - The model was insufficient for the app because it was taking a lot of time while inferencing on the frame.

  
- Model 2: FASTER RCNN INCEPTION v2
  - http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
  - I converted the model to an Intermediate Representation with the following arguments</br>
  ```python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb faster_rcnn_inception_v2_coco_2018_01_28/pipeline.config --tensorflow_object_detection_api_pipeline_config --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --reverse_input_channel```
  - This model also faced a similar issue liek the ssd resnet50 model that it was taking a lot of time while inferencing a frame.


- Model 3: SSD MOBILENET V1 FPN COCO
  - http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
  - I converted the model to an Intermediate Representation with the following arguments </br>
   ```python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/pipeline.config --tensorflow_object_detection_api_pipeline_config --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --reverse_input_channel```
  - The model was insufficient for the app because it wasn't able to accurately detect the people in frame even after reducing the threshold. The model was certainly fast but due to the accuracy issues I wasn't able to use it.
 
