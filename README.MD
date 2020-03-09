Project: Customer Counting & Profiling System

Benefits to using this system:

- Measures daily foot falls into the store
- Helps in determining conversion ratio
- Evaluates Effectives of promotional campaign
- Optimizes staff deployment
- Assist in comparing current location traffic with new one for long-term investment decision
- Creates customer group based on gender, couples
- Profiles customer based in props they carry (handbag, backpack, etc.)

Usage by Sector:
- Casinos
- Library
- Retail/Shopping Mall
- Public Sector

Different Computer Vision model types

There will be 3 model will be utlized

1. person-detection-retail-0013
Use case and High-level description
This is a pedestrian detector for the Retail scenario. It is based on MobileNetV2-like backbone that includes depth-wise convolutions to reduce the amount of computation for 

the 3x3 convolution block. The single SSD head from 1/16 scale feature map has 12 clustered prior boxes.

Example
person-detection-retail-0013.png
Specification
METRIC	VALUE
AP	88.62%
Pose coverage	Standing upright, parallel to image plane
Support of occluded pedestrians	YES
Occlusion coverage	<50%
Min pedestrian height	100 pixels (on 1080p)
GFlops	2.300
MParams	0.723
Source framework	Caffe*
Average Precision (AP) is defined as an area under the precision/recall curve.

Performance
Inputs
name: "input" , shape: [1x3x320x544] - An input image in the format [BxCxHxW], where:

B - batch size
C - number of channels
H - image height
W - image width
Expected color order is BGR.

Outputs
The net outputs blob with shape: [1, 1, N, 7], where N is the number of detected bounding boxes. For each detection, the description has the format: [image_id, label, conf, 

x_min, y_min, x_max, y_max]
image_id - ID of the image in the batch
label - predicted class ID
conf - confidence for the predicted class
(x_min, y_min) - coordinates of the top left bounding box corner
(x_max, y_max) - coordinates of the bottom right bounding box corner.

2. person-reidentification-retail-0031
Use Case and High-Level Description
This is a person reidentification model for a general scenario. It uses a whole body image as an input and outputs an embedding vector to match a pair of images by the cosine 

distance. The model is based on the RMNet backbone developed for fast inference. A single reidentification head from the 1/16 scale feature map outputs an embedding vector of 

256 floats.

Example
person-reidentification-retail-0031.png
Specification
METRIC	VALUE
Market-1501 rank@1 accuracy	0.7791
Market-1501 mAP	0.6180
Pose coverage	Standing upright, parallel to image plane
Support of occluded pedestrians	YES
Occlusion coverage	<50%
GFlops	0.028
MParams	0.280
Source framework	Caffe*
The cumulative matching curve (CMC) at rank-1 is accuracy denoting the possibility to locate at least one true positive in the top-1 rank. Mean Average Precision (mAP) is the 

mean across Average Precision (AP) of all queries. AP is defined as the area under the precision and recall curve.

Performance
Inputs
Name: data , shape: [1x3x96x48]. An input image in the format [BxCxHxW], where:

B - batch size
C - number of channels
H - image height
W - image width
The expected color order is BGR.

Outputs
The net outputs a blob with the [1, 256, 1, 1] shape named descriptor, which can be compared with other descriptors using the cosine distance.


3. person-attributes-recognition-crossroad-0230
Use Case and High-Level Description
This model presents a person attributes classification algorithm analysis scenario. It produces probability of person attributions existing on the sample and a position of 

two point on sample, which can be used for color prob (like, color picker in graphical editors)

Examples
person-attributes-recognition-crossroad-230-1.png person-attributes-recognition-crossroad-230-2.png person-attributes-recognition-crossroad-230-3.png person-attributes-

recognition-crossroad-230-4.png person-attributes-recognition-crossroad-230-5.png
Specification
METRIC	VALUE
Pedestrian pose	Standing person
Occlusion coverage	<20%
Min object width	80 pixels
Supported attributes	is_male, has_bag, has_backpack, has hat, has longsleeves, has longpants, has longhair, has coat_jacket
GFlops	0.174
MParams	0.735
Source framework	PyTorch*
Accuracy
ATTRIBUTE	F1
is_male	0.91
has_bag	0.66
has_backpack	0.77
has_hat	0.64
has_longsleeves	0.21
has_longpants	0.83
has_longhair	0.83
has_coat_jacket	NA
Performance
Inputs
name: "input" , shape: [1x3x160x80] - An input image in following format [1xCxHxW], where
- C - number of channels
    - H - image height
    - W - image width.

The expected color order is BGR.
Outputs
The net outputs a blob named 453 with shape: [1, 8, 1, 1] across eight attributes: [is_male, has_bag, has_backpack, has_hat, has_longsleeves, has_longpants, has_longhair, 

has_coat_jacket]. Value > 0.5 means that an attribute is present.
The net outputs a blob named 456 with shape: [1, 2, 1, 1]. It is location of point with top color.
The net outputs a blob named 459 with shape: [1, 2, 1, 1]. It is location of point with bottom color.
