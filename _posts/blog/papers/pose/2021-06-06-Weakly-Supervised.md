---
layout: post
bigtitle:  "Human Pose as Calibration Pattern; 3D Human Pose Estimation with Multiple Unsynchronized and Uncalibrated Cameras"
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



# Human Pose as Calibration Pattern; 3D Human Pose Estimation with Multiple Unsynchronized and Uncalibrated Cameras

Takahashi, Kosuke, et al. "Human pose as calibration pattern; 3D human pose estimation with multiple unsynchronized and uncalibrated cameras." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 2018. [paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w34/html/Takahashi_Human_Pose_As_CVPR_2018_paper.html)

* toc
{:toc}

## Abstract

This paper proposes a novel algorithm of estimating 3D human pose from multi-view videos captured by unsynchronized and uncalibrated cameras. In a such configuration, the conventional vision-based approaches utilize detected 2D features of common 3D points for synchronization and camera pose estimation, however, they sometimes suffer from difficulties of feature correspondences in case of wide baselines. For such cases, the proposed method focuses on that the projections of human joints can be associated each other robustly even in wide baseline videos and utilizes them as the common reference points. To utilize the projections of joint as the corresponding points, they should be detected in the images, however, these 2D joint sometimes include detection errors which make the estimation unstable. For dealing with such errors, the proposed method introduces two ideas. The first idea is to relax the reprojection errors for avoiding optimizing to noised observations. The second idea is to introduce an geometric constraint on the prior knowledge that the reference points consists of human joints. We demonstrate the performance of the proposed al- gorithm of synchronization and pose estimation with qualitative and quantitative evaluations using synthesized and real data.

## 1. Introduction

Measuring 3D human pose is important for analyzing the mechanics of the human body in various research fields, such as biomechanics, sports science and so on. In general, some additional devices, e.g. optical markers [1] and inertial sensors [2], are introduced for measuring 3D human pose. While these approaches have advantages in terms of high estimation quality, i.e. precision and robustness, it is sometimes difficult to utilize them in some practical scenarios, such as monitoring people in daily life or evaluating the performance of each player in a sports game, due to inconveniences of installing the devices.

To estimate 3D human pose in such cases, vision-based motion capture techniques have been studied in the field of computer vision [9]. Basically, they utilize multi-view cameras or depth sensors and assume that they are synchronized and calibrated beforehand. Such synchronization and calibration are troublesome to establish and maintain; typically, the cameras are connected by wires and capture the same reference object. Some 2D local features based synchronization and calibration methods have also been proposed for easy-to-use multi-view imaging systems in case for which the preparation cannot be done. However, they sometimes suffer from difficulties of feature correspondences in case for which the multiple cameras are scattered with wide baselines, which the erroneous correspondences affect the stability and precision of estimation severely.

This paper addresses the problem of 3D human pose estimation from multi-view videos captured by unsynchronized and uncalibrated cameras with wide baselines. The key feature of this paper is its focus on using the projections of human joints to derive robust point associations for use as common reference points. To detect the projections of human joints some 2D form of pose detector is needed [7,24,25], however, 2D joint positions sometime include detection errors which make the estimation unstable. To deal with such errors, the proposed method introduces two ideas. The first idea is to relax the reprojection errors to avoid the optimization of noisy observations. The second idea is to introduce a geometric constraint based on the a priori knowledge that the reference points are actually human joints.

The key contribution of this paper is to propose a novel algorithm for 2D human joint based multi-camera synchronization, camera pose estimation and 3D human pose estimation. This algorithm enables us to obtain 3D human pose easily and stably even in a practical and challenging scenes, such as sports games.

The reminder of this paper is organized as follow. Section 2 reviews related works in terms of synchronization, extrinsic calibration and human pose estimation. Section 3 introduces our proposed algorithm using the detected 2D joint positions as the corresponding points in multi-view images. Section 4 reports the performance evaluations and Section 5 provides discussions on the proposed method. Section 6 concludes this paper.

## 2. RelatedWorks

This section introduces the related works of our research in terms of (1) camera synchronization, (2) extrinsic camera calibration, and (3) human pose estimation.

**Camera Synchronization**  Multiple camera synchronization significantly impacts the estimation precision of multiview applications, such as 3D reconstruction. In general, the cameras are wired and receive a trigger signal from an external sensor telling the camera when to acquire an image. However, these wired connections can be troublesome to establish when the cameras are widely scattered as happens when capturing a sports games.

Audio-based approaches [9,18] estimate the time shift of multi-view videos in a software post-processing step, however, the significant difference in camera position degrades the estimation precision due to the delay in sound arrival.

Some image-based methods [3, 22] are able to synchronize the cameras even in such cases. Cenek et al. [3] estimates the time shift by using epipolar geometry on the corresponding points in the multi-view images. Tamaki et al. [22] detected the same table tennis ball in sequential frames and utilized them to establish point correspondences for computing epipolar geometry. Given the scale of the capture environment envisaged, our method is also based on epipolar geometry and so uses detected 2D joint positions as the corresponding points.

**Extrinsic Camera Calibration** Extrinsic camera calibration is an essential technique for 3D analysis and understanding from multi-view videos and various proposals have been made for various camera settings. Most proposals utilize detected 2D features, such as chess board corners or local features, e.g. SIFT [13], as the corresponding points. These approaches have difficulty in establishing reliable feature correspondence if the multiple cameras are scattered with wide baselines, as erroneous correspondences degrade the stability and precision of estimation severely.

For such cases, some studies utilize a priori knowledge of the scene. Huang et al. [11] use the trajectories of pedestrians in calibrating multiple fixed cameras based on the assumption that the cameras can capture the same pedestrians for a long time. Namdar et al. [10] assume that the cameras capture a sport scene in a stadium and calibrate them by introducing vanishing points computed from the lines on the sports field.

In addition, some studies [6, 16, 20] propose calibration algorithm that utilizes a priori knowledge that that the scenes contain humans. The silhouette-based approaches [5,19] establish the correspondences between special points on the silhouette boundaries, called frontier points [8], across the multiple views. These points are the projections of 3D points tangent to the epipolar plane. The epipolar geometry can be recovered from the correspondences of the frontier points.

Puwein et al. [16] proposed using detected 2D human joints in multi-view images as common reference points and using these points to compute the extrinsic parameters. Our method is inspired by [16]. In [16], the error function consists of reprojection error, a kinematic structure term, a smooth motion term and so on, is minimized in the bundle adjustment manner. Our work, on the other hand, introduces a relaxed reprojection error for robust estimation in the face of very noisy data; it also solves the synchronization problem.

**2D Human Pose Estimation** Conventional studies of 2D human pose estimation problem fall into two basic groups: pictral structure approach [4, 15], in which spatial correlations between each part are expressed as a tree-structured graphical model with kinematic priors that couple connected limbs, and hierarchical model approach [21, 23], which represents the relationships between parts at different scales and sizes in a hierarchical tree structure.

Given the rapid improvement in neural network techniques, a lot of neural network based 2D pose detectors have been proposed [7,24,25]. Toshev et al. [24] solve 2D human pose as a regression problem by introducing the AlexNet architecture, which was originally used for object recognition. Wei et al. [25] achieve high precise pose estimation by introducing CNN to the Pose Machine [17]. Caoet al. [7] consider the connectivity of each joint by introducing a part affinity field to the work of [25]; they achieve robust estimation of multi-person pose in real time.

## 3. Proposed Method

This section describes our proposed method for estimating 3D human pose with unsynchronized and uncalibrated multiple cameras with wide baselines.

### 3.1. Problem Formulation

#### 3.1.1 Relaxed Reprojection Error

#### 3.1.2 Constraints on Human Joints

### 3.2. Algorithm

Figure 4 illustrates the processing flow of the proposed method. First, it detects 2D joint positions from the input multi-view videos using a 2D pose detector such as [7,24,25]. Since the 2D pose detector output includes detection errors and joint detection sometimes fails due to selfocclusion, the proposed method applies a median filter after applying a cubic spline interpolation method to the output data. Next, select two cameras and the initial values of each parameter are estimated by the standard SfM approach using assumed time shift $d_i$; that is, it estimates the essential matrix for selected cameras, decomposes it into the extrinsic parameters, and estimates 3D joint positions through triangulation. The extrinsic camera parameters of the other cameras are estimated by solving PnP problems [12]. Then, Eq.(4) is computed using the initial parameters and minimized by the Levenberg-Marquardt algorithm over parameters $P$, $J$, and $L$. Finally, the parameters yielding the smallest value of Eq. (4) with $d_i$ are selected as the optimized parameters.

## 4. Evaluations

This section describes the performance evaluations of the proposed method with synthesized data and real data.

### 4.1. Evaluations with Synthesized Data

#### 4.1.1 Experimental Environment

#### 4.1.2 Results

Figure 6 plots the average errors of synchronization, extrinsic parameters, and 3D joint positions for 10 trials at each noise level. From these results, we can see that all methods estimate the time shift with error of 0.006 ∼ 0.012 seconds. As to the extrinsic parameters, the proposed method offers robust estimation even if large detection error is assumed while the conventional methods suffer degraded performace. Especially, Method2 significantly degrades in $\sigma > 1$ cases. We consider that the reason is that the Method2, which minimizes the reprojection errors strictly, is significantly affected by the noise and detection failures. The proposed method estimates 3D joint positions robustly while the Method1 degrades with noisy data. Method2 also estimates 3D joint positions with comparable precisions to the proposed method in spite of its degraded extrinsic parameters. This is considered that the adjusting the scale and initial position in case of evaluating the 3D positions absorbs this degradation.

Figure 7 renders one example of the estimated camera positions and 3D joint positions in 3D space with $\sigma = 5$. This figure shows that the proposed method estimates reasonable camera positions and 3D joints.

From the above, we can conclude that the proposed method is more robust than the conventional methods even if the input data includes significant noise, especially in terms of the extrinsic parameters.

### 4.2. Evaluations with Real Data

This section demonstrates that the proposed method works with real data in a practical scenario.

#### 4.2.1 Experimental Environment

Figure 8 shows the configuration of the evaluation that used real data. The two cameras (CASIO EX100) with 640 × 480 resolution and 120 fps were set with a wide baseline. Camera 0 and Camera 1 had focal lengths of 200mm and 165mm, respectively. The input video consisted of 1000 frames, that is about 8.3 seconds. These cameras captured one player throwing a ball. In the 2D pose estimation step, Cao et al. [7]’s method is
utilized. This evaluation used, in addition to Method1 and Method2 introduced in Section 4.1, Zhang’s method [26] as benchmarks.

#### 4.2.2 Results

Table 1 reports the estimation error of extrinsic parameters between each method and [26]. Figure 9 visualizes the estimated camera positions and 3D joint positions in 3D space. In Figure 9, while the camera positions estimated by Method1, Zhang’s method and the proposed method are almost same, that of Method2 diverged significantly. The reason is that the detected 2D poses include severe noise and Method2, which minimizes the reprojection errors strictly, is optimized to the noisy data same as in the evaluations with synthesized data, whereas the proposed method avoids the problem by relaxing the reprojection errors.
From these results, we can see that the proposed method works robustly with severely noise-degraded data in practical situations.

## 5. Discussion
### 5.1. Precision of 2D Human Pose Detector

The 2D pose detector is utilized in the first step of the proposed method and it has significant effects on the estimation precision. Here we investigate the performance of 2D pose detector [7] utilized in this paper.
Table shows the average, standard deviation, smallest value and biggest value of estimation error, that is euclidean norm of 2D human pose detected by [7] and its ground truth, in 700 frames with 1920 × 1080 resolutions. In the evaluations in Section 4, the $μ_p$ and $\sigma_p$ in the relaxed reprojection error are set based on these results.

### 5.2. Multiplayer Cases

As introduced in Section 3.1, our algorithm assume that there is a single player in the shared field-of-view of multiple cameras, however, it can be extend to multi-player cases. By considering the multi-players, it is considered that the estimation precision by the proposed method is improved because the number of constraints increase and the 2D joint positions, which are recognized as the corresponding points, cover more wide area in image planes of each camera. To deal with multi-player cases, the person identification problem and occlusion handling are to be solved in addition. This extension is included in our future works.

## 6. Conclusion

This paper proposed a novel 3D human joint position estimation algorithm for unsynchronized and uncalibrated cameras with wide baselines. The method focuses on the major skeleton joints and the constancy of joint separation. The 2D human pose is detected from the multi-view images and joint position estimates are used in the structure-frommotion manner. The proposed method provides an objective function consisting of a relaxed reprojection error term and human joint error term in order to achieve robust estimation even if the input data is noisy; the objective term is optimized. Evaluations using synthesized data and real data showed that the proposed method works robustly with noise-corrupted data. Future works include evaluations that use marker-based motion capture techniques and extension to the multi-player cases.
