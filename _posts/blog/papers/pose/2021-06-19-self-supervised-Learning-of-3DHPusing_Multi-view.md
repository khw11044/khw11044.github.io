---
layout: post
bigtitle:  "Self-Supervised Learning of 3D Human Pose using Multi-view Geometry"
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



# Self-Supervised Learning of 3D Human Pose using Multi-view Geometry

Kocabas, Muhammed, Salih Karagoz, and Emre Akbas. "Self-supervised learning of 3d human pose using multi-view geometry." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019. [paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Kocabas_Self-Supervised_Learning_of_3D_Human_Pose_Using_Multi-View_Geometry_CVPR_2019_paper.html)

* toc
{:toc}

## Abstract

Training accurate 3D human pose estimators requires large amount of 3D ground-truth data which is costly to collect.  
Various weakly or self supervised pose estimation methods have been proposed due to lack of 3D data.   
Nevertheless, these methods, in addition to 2D ground-truth poses, require either additional supervision in various forms (e.g. unpaired 3D ground truth data, a small subset of labels) or the camera parameters in multiview settings.  
To address these problems, we present EpipolarPose, a self-supervised learning method for 3D human pose estimation, which does not need any 3D ground-truth data or camera extrinsics.  
During training, EpipolarPose estimates 2D poses from multi-view images, and then, utilizes epipolar geometry to obtain a 3D pose and camera geometry which are subsequently used to train a 3D pose estimator.  
We demonstrate the effectiveness of our approach on standard benchmark datasets (i.e. Human3.6M and MPI-INF-3DHP) where we set the new state-of-the-art among weakly/self-supervised methods.  
Furthermore, we propose a new performance measure Pose Structure Score (PSS) which is a scale invariant, structure aware measure to evaluate the structural plausibility of a pose with respect to its ground truth.  
Code and pretrained models are available at [https://github.com/mkocabas/EpipolarPose](https://github.com/mkocabas/EpipolarPose)


## 1. Introduction

![Fig1](/assets/img/Blog/papers/Pose/Self-Supervised Learning of 3D Human Pose using Multi-view Geometry/Fig1.PNG)

Human pose estimation in the wild is a challenging problem in computer vision.  
Although there are large-scale datasets [2, 20] for two-dimensional (2D) pose estimation, 3D datasets [15, 23] are either limited to laboratory settings or limited in size and diversity.  
Since collecting 3D human pose annotations in the wild is costly and 3D datasets are limited, researchers have resorted to weakly or self supervised approaches with the goal of obtaining an accurate 3D pose estimator by using minimal amount of additional supervision on top of the existing 2D pose datasets.  
Various methods have been developed to this end.  
These methods, in addition to ground-truth 2D poses, require either additional supervision in various forms (such as unpaired 3D ground truth data[41], a small subset of labels [31]) or (extrinsic) camera parameters in multiview settings [30].  
To the best of our knowledge, there is only one method [9] which can produce a 3D pose estimator by using only 2D ground-truth.  
In this paper, we propose another such method.  

Our method, “EpiloparPose,” uses 2D pose estimation and epipolar geometry to obtain 3D poses, which are subsequently used to train a 3D pose estimator.  
EpipolarPose works with an arbitrary number of cameras (must be at least 2) and it does not need any 3D supervision or the extrinsic camera parameters, however, it can utilize them if provided.  
On the Human3.6M [15] and MPI-INF-3DHP [23] datasets, we set the new state-of-the-art in 3D pose estimation for weakly/self-supervised methods.  

Human pose estimation allows for subsequent higher level reasoning, e.g. in autonomous systems (cars, industrial robots) and activity recognition.  
In such tasks, structural errors in pose might be more important than the localization error measured by the traditional evaluation metrics such as MPJPE (mean per joint position error) and PCK (percentage of correct keypoints).  
These metrics treat each joint independently, hence, fail to asses the whole pose as a structure.  
Figure 4 shows that structurally very different poses yield the same MPJPE with respect to a reference pose.  
To address this issue, we propose a new performance measure, called the Pose Structure Score (PSS), which is sensitive to structural errors in pose.  
PSS computes a scale invariant performance score with the capability to score the structural plausibility of a pose with respect to its ground truth.  
Note that PSS is not a loss function, it is a performance measure that can be used along with MPJPE and PCK to account for structural errors made by a pose estimator.

To compute PSS, we first need to model the natural distribution of ground-truth poses.  
To this end, we use an unsupervised clustering method.  
Let $$\textbf{p}$$ be the predicted pose for an image whose ground-truth is $$\textbf{q}$$.  
First, we find which cluster centers are closest to $$\textbf{p}$$ and $$\textbf{q}$$.  
If both of them are closest to (i.e. assigned to) the same cluster center, then the pose structure score (PSS) of $$\textbf{p}$$ is said to be 1, otherwise 0.

**Contributions** Our contributions are as follows:

+ We present EpipolarPose, a method that can predict 3D human pose from a single-image. For training, EpipolarPose does not require any 3D supervision nor camera extrinsics. It creates its own 3D supervision by utilizing epipolar geometry and 2D ground-truth poses.

+ We set the new state-of-the-art among weakly/selfsupervised methods for 3D human pose estimation.

+ We present Pose Structure Score (PSS), a new performance measure for 3D human pose estimation to better capture structural errors.

## 2. Related Work

Our method, EpipolarPose, is a single-view method during inference; and a multi-view, self-supervised method during training.  
Before discussing such methods in the literature, we first briefly review entirely single-view (during both training and inference) and entirely multi-view methods for completeness.

**Single-view methods**
In many recent works, convolutional neural networks (CNN) are used to estimate the coordinates of the 3D joints directly from images [38, 39, 40, 35, 23].  
Li and Chan [19] were the first to show that deep neural networks can achieve a reasonable accuracy in 3D human pose estimation from a single image.  
They used two deep regression networks and body part detection.  
Tekin et al. [38] show that combining traditional CNNs for supervised learning with auto-encoders for structure learning can yield good results.  
Contrary to common regression practice, Pavlakos et al. [29] were the first to consider 3D human pose estimation as a 3D keypoint localization problem in a voxel space.  
Recently, “integral pose regression” proposed by Sun et al. [36] combined volumetric heat maps with a soft-argmax activation and obtained state-of-the-art results.

Additionally, there are two-stage approaches which decompose the 3D pose inference task into two independent stages: estimating 2D poses, and lifting them into 3D space [8, 24, 22, 11, 46, 8, 40, 23].   
Most recent methods in this category use state-of-the-art 2D pose estimators [7, 43, 25, 17] to obtain joint locations in the image plane.  
Martinez et al. [22] use a simple deep neural network that can estimate 3D pose given the estimated 2D pose computed by a state-of-the-art 2D pose estimator.  
Pavlakos et al. [28] proposed the idea of using ordinal depth relations among joints to bypass the need for full 3D supervision.

Methods in this category require either full 3D supervision or extra supervision (e.g. ordinal depth) in addition to full 3D supervision.

**Multi-view methods**  
Methods in this category require multi-view input both during testing and training.  
Early work [1, 5, 6, 3, 4] used 2D pose estimations obtained from calibrated cameras to produce 3D pose by triangulation or pictorial structures model.  
More recently, many researchers [10] used deep neural networks to model multi-view input with full 3D supervision.

**Weakly/self-supervised methods**  
Weak and self supervision based methods for human pose estimation have been explored by many [9, 31, 41, 30] due to lack of 3D annotations.  
Pavlakos et al. [30] use a pictorial structures model to obtain a global pose configuration from the keypoint heatmaps of multi-view images.  
Nevertheless, their method needs full camera calibration and a keypoint detector producing 2D heatmaps.

Rhodin et al. [31] utilize multi-view consistency constraints to supervise a network.  
They need a small amount of 3D ground-truth data to avoid degenerate solutions where poses collapse to a single location.  
Thus, lack of in-the-wild 3D ground-truth data is a limiting factor for this method [31].

Recently introduced deep inverse graphics networks [18, 44] have been applied to the human pose estimation problem [41, 9].  
Tung et al. [41] train a generative adversarial network which has a 3D pose generator trained with a reconstruction loss between projections of predicted 3D poses and input 2D joints and a discriminator trained to distinguish predicted 3D pose from a set of ground truth 3D poses.  
Following this work, Drover et al. [9] eliminated the need for 3D ground-truth by modifying the discriminator to recognize plausible 2D projections.

To the best of our knowledge, EpipolarPose and Drover et al.’s method are the only ones that do not require any 3D supervision or camera extrinsics.  
While their method does not utilize image features, EpipolarPose makes use of both image features and epipolar geometry and produces much more accurate results (4.3 mm less error than Drover et al.’s method).

![Fig1](/assets/img/Blog/papers/Pose/Self-Supervised Learning of 3D Human Pose using Multi-view Geometry/Fig1.PNG)

## 3. Models and Methods  

The overall training pipeline of our proposed method, EpipolarPose, is given in Figure 2. The orange-background part shows the inference pipeline. For training of EpipolarPose, the setup is assumed to be as follows. There are n cameras ($$n ≥ 2$$ must hold) which simultaneously take the picture of the person in the scene. The cameras are given id numbers from 1 to $$n$$ where consecutive cameras are close to each other (i.e. they have small baseline).  
The cameras produce images $$I_1, I_2, . . . I_n$$. Then, the set of consecutive image pairs, $$\{(I_i, I_{i+1})|i = 1, 2, . . . , n−1\}$$, form the training examples.

### 3.1. Training

In the training pipeline of EpipolarPose (Figure 2), there are two branches each starting with the same pose estimation network (a ResNet followed by a deconvolution network [36]).  
These networks were pre-trained on the MPII Human Pose dataset (MPII) [2].  
During training, only the pose estimation network in the upper branch is trained; the other one is kept frozen.

EpipolarPose can be trained using more than 2 cameras but for the sake of simplicity, here we will describe the training pipeline for $$n = 2$$.  
For $$n = 2$$, each training example contains only one image pair.  
Images $$I_i$$ and $$I_{i+1}$$ are fed into both the 3D (upper) branch and 2D (lower) branch pose estimation networks to obtain volumetric heatmaps $$\hat{H} ,H \in mathcal{R}^{w \times h \times d}$$ respectively, where $$w, h$$ are the spatial size after deconvolution, $$d$$ is the depth resolution defined as a hyperparameter.  
After applying soft argmax activation function $$\varphi(·)$$ we get 3D pose $$\hat{V} \in \mathcal{R}^{J \times 3} and 2D pose $$U \in R^{J \times 2} outputs where $$J$$ is the number of body joints.  
From a given volumetric heatmap, one can obtain both a 3D pose (by applying softargmax to all 3 dimensions) and a 2D pose (by applying softargmax to only $$x, y$$).

As an output of 2D pose branch, we want to obtain the 3D human pose $$V$$ in the global coordinate frame.  
Let the 2D coordinate of the $$j^{th}$$ joint in the $$i^{th}$$ image be $$U_{i,j} = [x_{i,j} , y_{i,j}]$$ and its 3D coordinate be $$[X_j , Y_j ,Z_j]$$, we can describe the relation between them assuming a pinhole image projection model  

$$\begin{bmatrix}
x_{i,j}  \\
y_{i,j} \\
w_{i,j}
\end{bmatrix} = K[R|RT]
\begin{bmatrix}
X_j  \\
Y_j \\
Z_j \\
1
\end{bmatrix} , K = \begin{bmatrix}
f_{x} & 0 & c_x  \\
0 & f_{y} & c_y \\
0 & 0 & 1
\end{bmatrix}, T = \begin{bmatrix}
T_{x}  \\
T_{y} \\
T_{z}
\end{bmatrix} \tag{1}$$

where $$w_{i,j}$$ is the depth of the $$j^{th}$$ joint in the $$i^{th}$$ camera’s image with respect to the camera reference frame, $$K$$ encodes the camera intrinsic parameters (e.g., focal length $$f_x$$ and $$f_y$$, principal point $$c_x$$ and $$x_y$$), $$R$$ and $$T$$ are camera extrinsic parameters of rotation and translation, respectively.  
We omit camera distortion for simplicity.

When camera extrinsic parameters are not available, which is usually the case in dynamic capture environments, we can use body joints as calibration targets.  
We assume the first camera as the center of the coordinate system, which means $$R$$ of the first camera is identity.  
For corresponding joints in $$U_i$$ and $$U{i+1}$$, in the image plane, we find the fundamental matrix $$F$$ satisfying $$U_{i,j}FU_{i+1,j} = 0$$ for $$\forall j$$ using the RANSAC algorithm. From $$F$$, we calculate the essential matrix $$E$$ by $$E = K^TFK$$.  
By decomposing $$E$$ with $$SVD$$, we obtain 4 possible solutions to $$R$$.  
We decide on the correct one by verifying possible pose hypotheses by doing cheirality check.  
The cheirality check basically means that the triangulated 3D points should have positive depth [26].  
We omit the scale during training, since our model uses normalized poses as ground truth.

Finally, to obtain a 3D pose $$V$$ for corresponding synchronized 2D images, we utilize triangulation (i.e. epipolar geometry) as follows.  
For all joints in ($$I_i, I_{i+1}$$) that are not occluded in either image, triangulate a 3D point $$[X_j , Y_j ,Z_j ]$$ using polynomial triangulation [12].  
For settings including more than 2 cameras, we calculate the vector-median to find the median 3D position.

To calculate the loss between 3D pose in camera frame $$\hat{V}$$ predicted by the upper (3D) branch, we project $$V$$ onto corresponding camera space, then minimize $$smooth_{L1} (V − \hat{V} )$$ to train the 3D branch where

$$smooth_{L1}(x)=
\begin{cases}
0.5x^2 & {if |x| < 1}\\
|x| - 0.5, &  {otherwise}
\end{cases} \tag{2}$$

**Why do we need a frozen 2D pose estimator?**

In the training pipeline of EpipolarPose, there are two branches each of which is starting with a pose estimator.  
While the estimator in the upper branch is trainable, the other one in the lower branch is frozen.  
The job of the lower branch estimator is to produce 2D poses.  
One might question the necessity of the frozen estimator since we could obtain 2D poses from the trainable upper branch as well.  
When we tried to do so, our method produced degenerate solutions where all keypoints collapse to a single location.  
In fact, other multi-view methods faced the same problem [31, 37].   
Rhodin et al. [31] solved this problem by using a small set of ground-truth examples, however, obtaining such groundtruth may not be feasible in most of the in the wild settings.  
Another solution proposed recently [37] is to minimize angular distance between estimated relative rotation $$\hat{R}$$ (computed via Procrustes alignment of the two sets of keypoints) and the ground truth $$R$$.  
Nevertheless, it is hard to obtain ground truth $$R$$ in dynamic capture setups.  
To overcome these shortcomings, we utilize a frozen 2D pose detector during training time only.

### 3.2. Inference

Inference involves the orange-background part in Figure 2.  
The input is just a single image and the output is the estimated 3D pose $$\hat{V}$$ obtained by a soft-argmax activation, $$\varphi(·)$$, on 3D volumetric heatmap $$\hat{H}_i$$.

### 3.3. Refinement, an optional posttraining

In the literature there are several techniques [22, 11, 39] to lift detected 2D keypoints into 3D joints.  
These methods are capable of learning generalized $$2D \rightarrow 3D$$ mapping which can be obtained from motion capture (MoCap) data by simulating random camera projections.  
Integrating a refinement unit (RU) to our self supervised model can further improve the pose estimation accuracy.  
In this way, one can train EpipolarPose on his/her own data which consists of multiple view footages without any labels and integrate it with RU to further improve the results.  
To make this possible, we modify the input layer of RU to accept noisy 3D detections from EpipolarPose and make it learn a refinement strategy. (See Figure 3)

The overall RU architecture is inspired by [22, 11].  
It has 2 computation blocks which have certain linear layers followed by Batch Normalization [14], Leaky ReLU [21] activation and Dropout layers to map 3D noisy inputs to more reliable 3D pose predictions.   
To facilitate information flow between layers, we add residual connections [13] and apply intermediate loss to expedite the intermediate layers’ access to supervision.

![Fig1](/assets/img/Blog/papers/Pose/Self-Supervised Learning of 3D Human Pose using Multi-view Geometry/Fig3.PNG)

![Fig1](/assets/img/Blog/papers/Pose/Self-Supervised Learning of 3D Human Pose using Multi-view Geometry/Fig4.PNG)

### 3.4. Pose Structure Score

As we discussed in Section 1, traditional distance-based evaluation metrics (such as MPJPE, PCK) treat each joint independently, hence, fail to asses the whole pose as a structure.  
In Figure 4, we present example poses that have the same MPJPE but are structurally very different, with respect to a reference pose.

We propose a new performance measure, called the Pose Structure Score (PSS), which is sensitive to structural errors in pose.  
PSS computes a scale invariant performance score with the capability to assess the structural plausibility of a pose with respect to its ground truth.  
Note that PSS is not a loss function, it is a performance score that can be used along with MPJPE and PCK to account for structural errors made by the pose estimator.  
PSS is an indicator about the deviation from the ground truth pose that has the potential to cause a wrong inference in a subsequent task requiring semantically meaningful poses, e.g. action recognition, human-robot interaction.

**How to compute PSS?**

**Implementation details**

![Fig1](/assets/img/Blog/papers/Pose/Self-Supervised Learning of 3D Human Pose using Multi-view Geometry/Fig5.PNG)

![Fig1](/assets/img/Blog/papers/Pose/Self-Supervised Learning of 3D Human Pose using Multi-view Geometry/Fig6.PNG)

## 4. Experiments

**Datasets.** We first conduct experiments on the Human3.6M (H36M) large scale 3D human pose estimation benchmark [15]. It is one of the largest datasets for 3D human pose estimation with 3.6 million images featuring 11 actors performing 15 daily activities, such as eating, sitting, walking and taking a photo, from 4 camera views. We mainly use this dataset for both quantitative and qualitative evaluation.

We follow the standard protocol on H36M and use the subjects 1, 5, 6, 7, 8 for training and the subjects 9, 11 for evaluation.  
Evaluation is performed on every $$64^{th}$$ frame of the test set.  
We include average errors for each method.

To demonstrate further applicability of our method, we use MPI-INF-3DHP (3DHP) [23] which is a recent dataset that includes both indoor and outdoor scenes. We follow the standard protocol: The five chest-height cameras and the provided 17 joints (compatible with H36M) are used for training. For evaluation, we use the official test set which includes challenging outdoor scenes. We report the results in terms of PCK and NPCK to be consistent with [31]. Note that we do not utilize any kind of background augmentation to boost the performance for outdoor test scenes.

**Metrics.** We evaluate pose accuracy in terms of MPJPE (mean per joint position error), PMPJPE (procrustes aligned mean per joint position error), PCK (percentage of correct keypoints), and PSS at scales @50 and @100.  
To compare our model with [31], we measured the normalized metrics NMPJPE and NPCK, please refer to [31] for further details.  
Note that PSS, by default, uses normalized poses during evaluation.  
In the presented results “n/a” means “not applicable” where it’s not possible to measure respective metric with provided information, “-” means “not available”.   
For instance, it’s not possible to measure MPJPE or PCK when $$R$$, the camera rotation matrix, is not available.  
For some of the previous methods with open source code, we indicate their respective PSS scores.  
We hope, in the future, PSS will be adapted as an additional performance measure, thus more results will become available for complete comparisons.

### 4.1. Results

**Can we rely on the labels from multi view images?**  


**Comparison to Stateoftheart**

**Weakly/Self Supervised Methods**

## 5. Conclusion

In this work, we have shown that even without any 3D ground truth data and the knowledge of camera extrinsics, multi view images can be leveraged to obtain self supervision. At the core of our approach, there is EpipolarPose which can utilize 2D poses from multi-view images using epipolar geometry to self-supervise a 3D pose estimator. EpipolarPose achieved state-of-the-art results in Human3.6M and MPI-INF-3D-HP benchmarks among weakly/self-supervised methods. In addition, we discussed the weaknesses of localization based metrics i.e. MPJPE and PCK for human pose estimation task and therefore proposed a new performance measure Pose Structure Score (PSS) to score the structural plausibility of a pose with respect to its ground truth.
