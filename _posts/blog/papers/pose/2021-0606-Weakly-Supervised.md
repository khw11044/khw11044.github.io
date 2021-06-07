---
layout: post
bigtitle:  "Weakly-Supervised 3D Human Pse Learning via Multi-view Images in the Wild"
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



# Weakly-Supervised 3D Human Pse Learning via Multi-view Images in the Wild

Umar Iqbal, Pavlo Molchanov, Jan Kautz, NVIDIA et al. Proceedings of the IEEE/CVF International Conference on Computer Vision. 2020. [paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Iqbal_Weakly-Supervised_3D_Human_Pose_Learning_via_Multi-View_Images_in_the_CVPR_2020_paper.html)

* toc
{:toc}

## Abstract
One major challenge for monocular 3D human pose estimation in-the-wild is the acquisition of training data that contains unconstrained images annotated with accurate 3D poses.  
In this paper, we address this challenge by proposing a weakly-supervised approach that does not require 3D annotations and learns to estimate 3D poses from unlabeled multi-view data, which can be acquired easily in in-the-wild environments.  
We propose a novel end-to-end learning framework that enables weakly-supervised training using multi-view consistency.  
Since multi-view consistency is prone to degenerated solutions, we adopt a 2.5D pose representation and propose a novel objective function that can only be minimized when the predictions of the trained model are consistent and plausible across all camera views.  
We evaluate our proposed approach on two large scale datasets (Human3.6M and MPII-INF-3DHP) where it achieves state-of-the-art performance among semi-/weaklysupervised methods.

>  in-the-wild에서 monocular 3D human pose estimation을 위한 한 가지 주요 과제는 정확한 3D poses로 annotated된 제약 없는 images을 포함하는 training data의 획득이다.  
본 논문에서는 3D annotations이 필요하지 않고 unlabeled multi-view data에서 3D poses를 estimate하는 방법을 학습하는 weakly-supervised approach을 제안함으로써 이 과제를 해결한다, 이는 in-the-wild 환경에서 쉽게 획득할 수 있다.  
multi-view 일관성을 사용하여 weakly-supervised training을 가능하게 하는 새로운 end-to-end learning framework를 제안한다.  
multi-view 일관성은 퇴화된 솔루션에 노출되기 쉽기 때문에 2.5D pose representation을 채택하고 훈련된 모델의 예측이 모든 camera views에서 일관되고 그럴듯할 때만 최소화할 수 있는 새로운 objective function를 제안한다.  
우리는 두 개의 대규모 데이터 세트(Human3.6M and MPII-INF-3DHP)에 대해 제안된 접근 방식을 평가한다, 그것은 semi-/weaklysupervised methods 중 state-of-the-art 성능을 달성한다.

## 1. Introduction

Learning to estimate 3D body pose from a single RGB image is of great interest for many practical applications.  
The state-of-the-art methods [6,16,17,28,32,39–41,52,53] in this area use images annotated with 3D poses and train deep neural networks to directly regress 3D pose from images.  
While the performance of these methods has improved significantly, their applicability in in-the-wild environments has been limited due to the lack of training data with ample diversity.  
The commonly used training datasets such as Human3.6M [10], and MPII-INF-3DHP [22] are collected in controlled indoor settings using sophisticated multi-camera motion capture systems.  
While scaling such systems to unconstrained outdoor environments is impractical, manual annotations are difficult to obtain and prone to errors.  
Therefore, current methods resort to existing training data and try to improve the generalizabilty of trained models by incorporating additional weak supervision in form of various 2D annotations for in-the-wild images [27,39,52].  
While 2D annotations can be obtained easily, they do not provide sufficient information about the 3D body pose, especially when the body joints are foreshortened or occluded.  
Therefore, these methods rely heavily on the ground-truth 3D annotations, in particular, for depth predictions.

Instead of using 3D annotations, in this work, we propose to use unlabeled multi-view data for training.  
We assume this data to be without extrinsic camera calibration.  
Hence, it can be collected very easily in any in-the-wild setting.  
In contrast to 2D annotations, using multi-view data for training has several obvious advantages e.g., ambiguities arising due to body joint occlusions as well as foreshortening or motion blur can be resolved by utilizing information from other views.  
There have been only few works [14, 29, 33, 34] that utilize multi-view data to learn monocular 3D pose estimation models.  
While the approaches [29,33] need extrinsic camera calibration, [33,34] require at least some part of their training data to be labelled with ground-truth 3D poses.  
Both of these requirements are, however, very hard to acquire for unconstrained data, hence, limit the applicability of these methods to controlled indoor settings.  
In [14], 2D poses obtained from multiple camera views are used to generate pseudo ground-truths for training.  
However, this method uses a pre-trained pose estimation model which remains fixed during training, meaning 2D pose errors remain unaddressed and can propagate to the generated pseudo ground-truths.

In this work, we present a weakly-supervised approach for monocular 3D pose estimation that does not require any 3D pose annotations at all.  
For training, we only use a collection of unlabeled multi-view data and an independent collection of images annotated with 2D poses.  
An overview of the approach can be seen in Fig. 1.  
Given an RGB image as input, we train the network to predict a 2.5D pose representation [12] from which the 3D pose can be reconstructed in a fully-differentiable way.  
Given unlabeled multi-view data, we use a multi-view consistency loss which enforces the 3D poses estimated from different views to be consistent up to a rigid transformation.  
However, naively enforcing multi-view consistency can lead to degenerated solutions.  
We, therefore, propose a novel objective function which is constrained such that it can only be minimized when the 3D poses are predicted correctly from all camera views.  
The proposed approach can be trained in a fully end-to-end manner, it does not require extrinsic camera calibration and is robust to body part occlusions and truncations in the unlabeled multi-view data.  
Furthermore, it can also improve the 2D pose predictions by exploiting multi-view consistency during training.

We evaluate our approach on two large scale datasets where it outperforms existing methods for semi-/weakly supervised methods by a large margin.  
We also show that the MannequinChallenge dataset [18], which provides in-the-wild videos of people in static poses, can be effectively exploited by our proposed method to improve the generalizability of trained models, in particular, when their is a significant domain gap between the training and testing environments.

## 2. Related Work

We discuss existing methods for monocular 3D human pose estimation with varying degree of supervision.

**Fully-supervised methods**  
aim to learn a mapping from 2D information to 3D given pairs of 2D-3D correspondences as supervision. The recent methods in this direction adopt deep neural networks to directly predict 3D poses from images [16, 17, 41, 53]. Training the data hungry neural networks, however, requires large amounts of training images with accurate 3D pose annotations which are very hard to acquire, in particular, in unconstrained scenarios. To this end, the approaches in [5, 35, 45] try to augment the training data using synthetic images, however, still need real data to obtain good performance. More recent methods try to improve the performance by incorporating additional data with weak supervision i.e., 2D pose annotations [6, 28, 32, 39, 40, 52], boolean geometric relationship between body parts [27, 31, 37], action labels [20], and temporal consistency [2]. Adverserial losses during training [50] or testing [44] have also been used to improve the performance of models trained on fully-supervised data.

Other methods alleviate the need of 3D image annotations by directly lifting 2D poses to 3D without using any image information e.g., by learning a regression network from 2D joints to 3D [9, 21, 24] or by searching nearest 3D poses in large databases using 2D projections as the query [3, 11, 31]. Since these methods do not use image information for 3D pose estimation, they are prone to reprojection ambiguities and can also have discrepancies between the 2D and 3D poses.

In contrast, in this work, we present a method that combines the benefits of both paradigms i.e., it estimates 3D pose from an image input, hence, can handle the reprojection ambiguities, but does not require any images with 3D pose annotations.

**Semi-supervised methods**  
require only a small subset of training data with 3D annotations and assume no or weak supervision for the rest. The approaches [33,34,51] assume that multiple views of the same 2D pose are available and use multi-view constraints for supervision. Closest to our approach in this category is [34] in that it also uses multiview consistency to supervise the pose estimation model. However, their method is prone to degenerated solutions and its solution space cannot be constrained easily. Consequently, the requirement of images with 3D annotations is inevitable for their approach. In contrast, our method is weakly-supervised. We constrain the solution space of our method such that the 3D poses can be learned without any 3D annotations. In contrast to [34], our approach can easily be applied to in-the-wild scenarios as we will show in our experiments. The approaches [43, 48] use 2D pose annotations and re-projection losses to improve the performance of models pre-trained using synthetic data. In [19], a pretrained model is iteratively improved by refining its predictions using temporal information and then using them as supervision for next steps. The approach in [30] estimates the 3D poses using a sequence of 2D poses as input and uses a re-projection loss accumulated over the entire sequence for supervision. While all of these methods demonstrate impressive results, their main limiting factor is the need of ground-truth 3D data.

**Weakly-supervised methods**  
do not require paired 2D- 3D data and only use weak supervision in form of motioncapture data [42], images/videos with 2D annotations [25, 47], collection of 2D poses [4, 7, 46], or multi-view images [14, 29]. Our approach also lies in this paradigm and learns to estimate 3D poses from unlabeled multi-view data. In [42], a probabilistic 3D pose model learned using motion-capture data is integrated into a multi-staged 2D pose estimation model to iteratively refine 2D and 3D pose predictions. The approach [25] uses a re-projection loss to train the pose estimation model using images with only 2D pose annotations. Since re-projection loss alone is insufficient for training, they factorize the problem into the estimation of view-point and shape parameters and provide inductive bias via a canonicalization loss. Similar in spirit, the approaches [4,7,46] use collection of 2D poses with reprojection loss for training and use adversarial losses to distinguish between plausible and in-plausible poses. In [47], non-rigid structure from motion is used to learn a 3D pose estimator from videos with 2D pose annotations. The closest to our work are the approaches of [14, 29] in that they also use unlabeled multi-view data for training. The approach of [29], however, requires calibrated camera views that are very hard to acquire in unconstrained environments. The approach [14] estimates 2D poses from multi-view images and reconstructs corresponding 3D pose using Epipolar geometry. The reconstructed poses are then used for training in a fully-supervised way.

The main drawback of this method is that the 3D poses remain fixed throughout the training, and the errors in 3D reconstruction directly propagate to the trained models. This is, particularly, problematic if the multi-view data is captured in challenging outdoor environments where 2D pose estimation may fail easily. In contrast, in this work, we propose an end-to-end learning framework which is robust to challenges posed by the data captured in in-the-wild scenarios. It is trained using a novel objective function which can simultaneously optimize for 2D and 3D poses. In contrast to [14], our approach can also improve 2D predictions using unlabeled multi-view data. We evaluate our approach on two challenging datasets where it outperforms existing methods for semi-/weaklysupervised learning by a large margin.

## 3. Method

### 3.1. 2.5D Pose Representation

#### 3.1.1 Differentiable 3D Reconstruction

### 3.2. 2.5D Pose Regression

### 3.3. WeaklySupervised Training

**Heatmap Loss**

**Multi-View Consistency Loss**

**Limb Length Loss**

**Additional Regularization**

## 4. Experiments

### 4.1. Datasets

### 4.2. Evaluation Metrics

### 4.3. Ablation Studies

### 4.4. Comparison with the State-of-the-Art

## 5. Conclusion

We have presented a weakly-supervised approach for 3D human pose estimation in the wild.  
Our proposed approach does not require any 3D annotations and can learn to estimate 3D poses from unlabeled multi-view data.  
This is made possible by a novel end-to-end learning framework and a novel objective function which is optimized to predict consistent 3D poses across different camera views.  
The proposed approach is very practical since the required training data can be collected very easily in in-the-wild scenarios.  
We demonstrated state-of-the-art performance on two challenging datasets.

Acknowledgments: We are thankful to Kihwan Kim and Adrian Spurr for helpful discussions.
