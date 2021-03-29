---
layout: post
bigtitle:  "From Points to Parts"
subtitle:   ": 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network"
categories:
    - blog
    - papers
    - pointcloud
tags:
    - point-cloud
    - detection
comments: true
published: true
---

IEEE 28 February 2020 [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9018080)

## Abstract
3D object detection from LiDAR point cloud is a challenging problem in 3D scene understanding and has many practical applications.

In this paper, we extend our preliminary work PointRCNN to a novel and strong point-cloud-based 3D object detection framework, the part-aware and aggregation neural network (Part-A<sup>2</sup> net). The whole framework consists of the part-aware stage and the part-aggregation stage.

Firstly, the part-aware stage for the first time fully utilizes free-of-charge part supervisions derived from 3D ground-truth boxes to simultaneously predict high quality 3D proposals and accurate intra-object part locations.

The predicted intra-object part locations within the same proposal are grouped by our new-designed RoI-aware point cloud pooling module, which results in an effective representation to encode the geometry-specific features of each 3D proposal.

Then the part-aggregation stage learns to re-score the box and refine the box location by exploring the spatial relationship of the pooled intra-object part locations.  
Extensive experiments are conducted to demonstrate the performance improvements from each component of our proposed framework.

Our Part-A2 net outperforms all existing 3D detection methods and achieves new state-of-the-art on KITTI 3D object detection dataset by utilizing only the LiDAR point cloud data.
