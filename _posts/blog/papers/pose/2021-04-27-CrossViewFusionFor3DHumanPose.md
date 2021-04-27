---
layout: post
bigtitle:  "Cross View Fusion for 3D Human Pose Estimation"
subtitle:   "."
categories:
    - blog
    - papers
    - pose-estimation
tags:
    - pose
comments: true
published: true
---



# Cross View Fusion for 3D Human Pose Estimation

ICCV 2019 [paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Qiu_Cross_View_Fusion_for_3D_Human_Pose_Estimation_ICCV_2019_paper.pdf)

* toc
{:toc}

## Abstract
We present an approach to recover absolute 3D human poses from multi-view images by incorporating multi-view geometric priors in our model. It consists of two separate steps: (1) estimating the 2D poses in multi-view images and (2) recovering the 3D poses from the multi-view 2D poses.  

> 우리는 모델에 multi-view geometric 사전확률을 통합하여 multi-view images에서 absolute 3D human poses를 복구하는 접근 방식을 제시한다.   이 단계는 두 단계로 구성됩니다. (1) 멀티 뷰 영상에서 2D 포즈를 추정하는 것과 (2) 멀티 뷰 2D 포즈에서 3D 포즈를 복구하는 것입니다.


First, we introduce a cross-view fusion scheme into CNN to jointly estimate 2D poses for multiple views. Consequently, the 2D pose estimation for each view already benefits from other views.  
Second, we present a recursive Pictorial Structure Model to recover the 3D pose from the multi-view 2D poses.  
It gradually improves the accuracy of 3D pose with affordable computational cost.  
We test our method on two public datasets H36M and Total Capture.  
The Mean Per Joint Position Errors on the two datasets are 26mm and 29mm, which outperforms the state-of-the-arts remarkably (26mm vs 52mm, 29mm vs 35mm).


## 1. Introduction
The task of 3D pose estimation has made significant progress due to the introduction of deep neural networks.  
Most efforts [16, 13, 33, 17, 23, 19, 29, 28, 6] have been devoted to estimating relative 3D poses from monocular images.  
The estimated poses are centered around the pelvis joint thus do not know their absolute locations in the environment (world coordinate system).  

In this paper, we tackle the problem of estimating absolute 3D poses in the world coordinate system from multiple cameras [1, 15, 4, 18, 3, 20].  
Most works follow the pipeline of first estimating 2D poses and then recovering 3D pose from them.  
However, the latter step usually depends on the performance of the first step which unfortunately often has large errors in practice especially when occlusion or motion blur occurs in images.  
This poses a big challenge for the final 3D estimation.  

On the other hand, using the Pictorial Structure Model (PSM) [14, 18, 3] for 3D pose estimation can alleviate the influence of inaccurate 2D joints by considering their spatial dependence.  
It discretizes the space around the root joint by an N ×N ×N grid and assigns each joint to one of the $$N^3$$ bins (hypotheses).  
It jointly minimizes the projection error between the estimated 3D pose and the 2D pose, along with the discrepancy of the spatial configuration of joints and its prior structures.  
However, the space discretization causes large quantization errors.  
For example, when the space surrounding the human is of size 2000mm and N is 32, the quantization error is as large as 30mm. We could reduce the error by increasing N, but the inference cost also increases at $$O(N^3)$$, which is usually intractable.  

Second, we present Recursive Pictorial Structure Model (RPSM), to recover the 3D pose from the estimated multi-view 2D pose heatmaps.  
Different from PSM which directly discretizes the space into a large number of bins in order to control the quantization error, RPSM recursively discretizes the space around each joint location (estimated in the previous iteration) into a finer-grained grid using a small number of bins.  
As a result, the estimated 3D pose is refined step by step.  
Since N in each step is usually small, the inference speed is very fast for a single iteration.  
In our experiments, RPSM decreases the error by at least 50% compared to PSM with little increase of inference time.  

For 2D pose estimation on the H36M dataset [11], the average detection rate over all joints improves from 89% to 96%.  
The improvement is significant for the most challenging “wrist” joint.  
For 3D pose estimation, changing PSM to RPSM dramatically reduces the average error from 77mm to 26mm.  
Even compared with the state-of-the-art method with an average error 52mm, our approach also cuts the error in half.  
We further evaluate our approach on the Total Capture dataset [27] to validate its generalization ability.  
It still outperforms the state-of-the-art [26].
