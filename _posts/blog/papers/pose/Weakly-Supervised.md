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
We propose a novel end-to-end learning framework that enables weakly-supervised training using multi-view consistency. Since multi-view consistency is prone to degenerated solutions, we adopt a 2.5D pose representation and propose a novel objective function that can only be minimized when the predictions of the trained model are consistent and plausible across all camera views. We evaluate our proposed approach on two large scale datasets (Human3.6M and MPII-INF-3DHP) where it achieves state-of-the-art performance among semi-/weaklysupervised methods.

## 1. Introduction

Learning to estimate 3D body pose from a single RGB image is of great interest for many practical applications. The state-of-the-art methods [6,16,17,28,32,39–41,52,53] in this area use images annotated with 3D poses and train deep neural networks to directly regress 3D pose from images. While the performance of these methods has improved significantly, their applicability in in-the-wild environments has been limited due to the lack of training data with ample diversity. The commonly used training datasets such as Human3.6M [10], and MPII-INF-3DHP [22] are collected in controlled indoor settings using sophisticated multi-camera motion capture systems. While scaling such systems to unconstrained outdoor environments is impractical, manual annotations are difficult to obtain and prone to errors. Therefore, current methods resort to existing training data and try to improve the generalizabilty of trained models by incorporating additional weak supervision in form of various 2D annotations for in-the-wild images [27,39,52]. While 2D annotations can be obtained easily, they do not provide sufficient information about the 3D body pose, especially when the body joints are foreshortened or occluded. Therefore, these methods rely heavily on the ground-truth 3D annotations, in particular, for depth predictions.

Instead of using 3D annotations, in this work, we propose to use unlabeled multi-view data for training. We assume this data to be without extrinsic camera calibration. Hence, it can be collected very easily in any in-the-wild setting. In contrast to 2D annotations, using multi-view data for training has several obvious advantages e.g., ambiguities arising due to body joint occlusions as well as foreshortening or motion blur can be resolved by utilizing information from other views. There have been only few works [14, 29, 33, 34] that utilize multi-view data to learn monocular 3D pose estimation models. While the approaches [29,33] need extrinsic camera calibration, [33,34] require at least some part of their training data to be labelled with ground-truth 3D poses. Both of these requirements are, however, very hard to acquire for unconstrained data, hence, limit the applicability of these methods to controlled indoor settings. In [14], 2D poses obtained from multiple camera views are used to generate pseudo ground-truths for training. However, this method uses a pre-trained pose estimation model which remains fixed during training, meaning 2D pose errors remain unaddressed and can propagate to the generated pseudo ground-truths.

In this work, we present a weakly-supervised approach for monocular 3D pose estimation that does not require any 3D pose annotations at all. For training, we only use a collection of unlabeled multi-view data and an independent collection of images annotated with 2D poses. An overview of the approach can be seen in Fig. 1. Given an RGB image as input, we train the network to predict a 2.5D pose representation [12] from which the 3D pose can be reconstructed in a fully-differentiable way. Given unlabeled multi-view data, we use a multi-view consistency loss which enforces the 3D poses estimated from different views to be consistent up to a rigid transformation. However, naively enforcing multi-view consistency can lead to degenerated solutions. We, therefore, propose a novel objective function which is constrained such that it can only be minimized when the 3D poses are predicted correctly from all camera views. The proposed approach can be trained in a fully end-to-end manner, it does not require extrinsic camera calibration and is robust to body part occlusions and truncations in the unlabeled multi-view data. Furthermore, it can also improve the 2D pose predictions by exploiting multi-view consistency during training.

We evaluate our approach on two large scale datasets where it outperforms existing methods for semi-/weaklysupervised methods by a large margin. We also show that the MannequinChallenge dataset [18], which provides inthe- wild videos of people in static poses, can be effectively exploited by our proposed method to improve the generalizability of trained models, in particular, when their is a significant domain gap between the training and testing environments.

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

Our approach assumes we have synchronized video streams from $$C$$ cameras with known projection matrices $$P_c$$ capturing performance of a single person in the scene.  
We aim at estimating the global 3D positions $$y_{j,t}$$ of a fixed set of human joints with indices $$j \in (1..J)$$ at timestamp $$t$$.  
For each timestamp the frames are processed independently (i.e. without using temporal information), thus we omit the index $$t$$ for clarity.

For each frame, we crop the images using the bounding boxes either estimated by available off-the-shelf 2D human detectors or from ground truth (if provided).  
Then we feed the cropped images Ic into a deep convolutional neural network backbone based on the "simple baselines" architecture [21].

The convolutional neural network backbone with learnable weights $$\theta$$ consists of a ResNet-152 network (output denoted by $$g_{\theta}$$), followed by a series of transposed convolutions that produce intermediate heatmaps (the output denoted by $$f_{\theta}$$) and a $$1 \times 1$$ - kernel convolutional neural network that transforms the intermediate heatmaps to interpretable joint heatmaps (output denoted by $$h+{\theta}$$; the number of output channels is equal to the number of joints J).  
In the two following sections we describe two different methods to infer joints’ 3D coordinates by aggregating information from multiple views.



**Algebraic triangulation approach.**  

![Fig1](/assets/img/Blog/papers/Pose/learnable/fig1.PNG)  
Figure 1. Outline of the approach based on algebraic triangulation with learned confidences. The input for the method is a set of RGB
images with known camera parameters. The 2D backbone produces the joints’ heatmaps and camera-joint confidences. The 2D positions
of the joints are inferred from 2D joint heatmaps by applying soft-argmax. The 2D positions together with the confidences are passed to
the algebraic triangulation module that outputs the triangulated 3D pose. All blocks allow backpropagation of the gradients, so the model
can be trained end-to-end.

In the algebraic triangulation baseline we process each joint j independently of each other.  
The approach is built upon triangulating the 2D positions obtained from the j-joint’s backbone heatmaps from different views: $$H_{c,j} = h_{\theta}(I_c)_j$$ (Figure 1).  
To estimate the 2D positions we first compute the softmax across the spatial axes:
$$H^{'}_{c,j} = \exp(\alpha H_{c,j})/ \sum^W_{r_x=1} \sum^H_{r_y=1} \exp(\alpha H_{c,j}(r))), \; \; \; (1)$$

where parameter $$\alpha$$ is discussed below.  
Then we calculate the 2D positions of the joints as the center of mass of the corresponding heatmaps (so-called soft-argmax operation):

$$\mathrm{x}_{c,j} = \sum^W_{r_x=1} \sum^H_{r_y=1} \mathbf{r} \cdot (H^{'}_{c,j}(\mathbf{r})), \; \; \; (2)$$

An important feature of soft-argmax is that rather than getting the index of the maximum, it allows the gradients to flow back to heatmaps $$H_c$$ from the output 2D position of the joints $$\mathrm{x}$$.  
Since the backbone was pretrained using a loss other than soft-argmax (MSE over heatmaps without softmax [16]), we adjust the heatmaps via multiplying them by an ’inverse temperature’ parameter $$\alpha = 100$$ in (1), so at the start of the training the soft-argmax gives an output close to the positions of the maximum.

To infer the 3D positions of the joints from their 2D estimates $$\mathrm{x}_{c,j}$$ we use a linear algebraic triangulation approach [1].  
The method reduces the finding of the 3D coordinates of a joint $$y_j$$ to solving the overdetermined system of equations on homogeneous 3D coordinate vector of the joint $$\tilde{y}$$:
$$A_{j}\tilde{\mathrm{y}}_j = 0 \; \; \; (3)$$  

where $$A_j \in \mathbb{R}^{(2C,4)}$$ is a matrix composed of the components from the full projection matrices and $$x_{c,j}$$ (see [1] for full details).

A naive triangulation algorithm assumes that the joint coordinates from each view are independent of each other and thus all make comparable contributions to the triangulation.  
However, on some views the 2D position of the joints cannot be estimated reliably (e.g. due to joint occlusions), leading to unnecessary degradation of the final triangulation result.

This greatly exacerbates the tendency of methods that optimize algebraic reprojection error to pay uneven attention to different views.  
The problem can be dealt with by applying RANSAC together with the Huber loss (used to score reprojection errors corresponding to inliers).  
However, this has its own drawbacks.  
E.g. using RANSAC may completely cut off the gradient flow to the excluded cameras.

To address the aforementioned problems, we add learnable weights $$w_c$$ to the coefficients of the matrix corresponding to different views:

$$(\mathbf{w}_j \circ A_j)\tilde{\mathrm{y}}_j = 0, \; \; \; (4)$$

where $$\mathbf{w}_j = (w_{1,j}, w_{1,j}, w_{2,j}, w_{2,j} , ... ,, w_{C,j} ,w_{C,j}); \circ$$ denotes the Hadamard product (i.e. i-th row of matrix A is multiplied by i-th element of vector $$\mathbf{w}$$).  
The weights $$w_{c,j}$$ are estimated by a convolutional network $$q^{\phi}$$ with learnable parameters $$\phi$$ (comprised of two convolutional layers, global average pooling and three fully-connected layers), applied to the intermediate output of the backbone:

$$w_{c,j} = (q^{\phi}(g^{\theta}(I_c)))_j \; \; \; (5)$$

This allows the contribution of the each camera view to be controlled by the neural network branch that is learned jointly with the backbone joint detector.


The equation (4) is solved via differentiable Singular Value Decomposition of the matrix $$B = UDV^T$$ , from which $$\tilde{\mathrm{y}}$$ is set as the last column of V.  
The final nonhomogeneous value of $$\tilde{\mathrm{y}}$$ is obtained by dividing the homogeneous 3D coordinate vector $$\tilde{\mathrm{y}}$$ by its fourth coordinate: $$\mathrm{y} = \tilde{\mathrm{y}}/(\tilde{\mathrm{y}})_4$$.
**Volumetric triangulation approach.**

![Fig2](/assets/img/Blog/papers/Pose/learnable/fig2.PNG)  
Figure 2. Outline of the approach based on volumetric triangulation. The input for the method is a set of RGB images with known camera parameters. The 2D backbone produces intermediate feature maps that are unprojected into volumes with subsequent aggreagation to a fixed size volume. The volume is passed to a 3D convolutional neural network that outputs the interpretable 3D heatmaps. The output 3D positions of the joints are inferred from 3D joint heatmaps by computing soft-argmax. All blocks allow backpropagation of the gradients, so the model can be trained end-to-end.

The main drawback of the baseline algebraic triangulation approach is that the images $$I_c$$ from different cameras are processed independently from each other, so there is no easy way to add a 3D human pose prior and no way to filter out the cameras with wrong projection matrices.

To solve this problem we propose to use a more complex and powerful triangulation procedure.  
We unproject the feature maps produced by the 2D backbone into 3D volumes (see Figure 2).  
This is done by filling a 3D cube around the person via projecting output of the 2D network along projection rays inside the 3D cube.  
The cubes obtained from multiple views are then aggregated together and processed.  
For such volumetric triangulation approach, the 2D output does not have to be interpretable as joint heatmaps, thus, instead of unprojecting $$H_c$$ themselves, we use the output of a trainable single layer convolutional neural network $$o^{\gamma}$$ with $$1 \times 1$$ kernel and $$\mathrm{K}$$ output channels (the weights of this layer are denoted by ) applied to the input from the backbone intermediate heatmaps $$f^{\theta}(I_c)$$:

$$M_{c,k} = o^{\gamma}(f^{\theta}(I_c))_k \; \; \; (6)$$

To create the volumetric grid, we place a $$L \times L \times L$$ - sized 3D bound box in the global space around the human pelvis (the position of the pelvis is estimated by the algebraic triangulation baseline described above, $$L$$ denotes the size of the box in meters) with the Y-axis perpendicular to the ground and a random orientation of the X-axis.  
We discretize the bounding box by a volumetric cube $$V^{coords} \in \mathbb{R}^{64,64,64,3}$$, filling it with the global coordinates of the center of each voxel (in a similar way to [5]).

For each view, we then project the 3D coordinates in $$V^{coords}$$ to the plane: $$V^{proj}_c = P_cV^{coords}$$ (note that $$V^{proj}_c \in \mathbb{R}^{64,64,64,2}$$ ) and fill a cube $$V^{view}_c \in \mathbb{R}^{64,64,64,K}$$ by bilinear sampling [4] from the maps $$M_{c,k}$$ of the corresponding camera view using 2D coordinates in $$V^{proj}_c$$ :

$$V^{view}_{c,k} = M_{c,k}\{V^{proj}_c\}, \; \; \; \; (7)$$

where $$\{ \cdot \}$$ denotes bilinear sampling.  
We then aggregate the volumetric maps from all views to form an input to the further processing that does not depend on the number of camera views.  
We study three diverse methods for the aggregation:

1. Raw summation of the voxel data:
$$V^{input}_{k} = \sum_c V^{view}_{c,k}, \; \; \; \; (8)$$

2. Summation of the voxel data with normalized confidence multipliers $$d_c$$ (obtained similarly to $$w_c$$ using a branch attached to backbone):
$$V^{input}_k = \sum_c(d_c \cdot V^{view}_{c,k}) / \sum_c d_c \; \; (9)$$

3. Calculating a relaxed version of maximum. Here, we first compute the softmax for each individual voxel $$V^{view}_c$$ across all cameras, producing the volumetric coefficient distribution $$V^w_{c,k}$$ with the role similar to scalars $$d_c$$:

$$V_{c,k}^w = \exp(V^{view}_{c,k}) / \sum_c \exp(V^{view}_{c,k}) \; \; (10)$$

Then, the voxel maps from each view are summed with the volumetric coefficients $$V^w_c$$ :

$$V^{input}_k = \sum_c V^w_{c,k} \circ V^{view}_c \; \; \; (11)$$

Aggregated volumetric maps are then fed into a learnable volumetric convolutional neural network $$u^v$$ (with weights denoted by $$v$$), with architecture similar to V2V [11], producing the interpretable 3D-heatmaps of the output joints:

$$V^{output}_j = (u^v(V^{input}))_j \; \; \; (12)$$

Next, we compute softmax of $$V^{output}_j$$ across the spatial axes (similar to (1)):

$$V'^{output}_j = \exp(V^{output}_j) / (\sum^W_{r_x=1}\sum^H_{r_y=1}\sum^D_{r_z=1} \exp(V^{output}_j (\mathbf{r}))), \; \; \; (13) $$

and estimate the center of mass for each of the volumetric joint heatmaps to infer the positions of the joints in 3D:

$$y_i = \sum^W_{r_x=1}\sum^H_{r_y=1}\sum^D_{r_z=1} \mathbf{r} \cdot V'^{output}_j(\mathbf{r}), \; \; \; (14)$$

Lifting to 3D allows getting more robust results, as the wrong predictions are spatially isolated from the correct ones inside the cube, so they can be discarded by convolutional operations.  
The network also naturally incorporates the camera parameters (uses them as an input), allows modelling the human pose prior and can reasonably handle multimodality in 2D detections.

**Losses.**

For both of the methods described above, the gradients pass from the output prediction of 3D joints’ coordinates $$y_j$$ to the input RGB-images $$I_c$$ making the pipeline trainable end-to-end. For the case of algebraic triangulation, we apply a soft version of per-joint Mean Square Error
(MSE) loss to make the training more robust to outliers. This variant leads to better results compared to raw MSE or L1 (mean absolute error):

![c1](/assets/img/Blog/papers/Pose/learnable/ev(15).PNG)  

Here, $$\varepsilon$$ denotes the threshold for the loss, which is set to (20 cm)<sup>2</sup> in the experiments.  
The final loss is the average over all valid joints and all scenes in the batch.

For the case of volumetric triangulation, we use the L1 loss with a weak heatmap regularizer, which maximizes the prediction for the voxel that has inside of it the ground-truth joint:

$$\mathcal{L}^{vol}(\theta, \gamma, \nu) = \sum_j \vert y_j - y_j^{gt} \vert - \beta \cdot log(V_j^{output}(y^{gt}_j)) \; \; (16)$$

Without the second term, for some of the joints (especially, pelvis) the produced output volumetric heatmaps are not interpretable, probably due to insufficient size of the training datasets [16].  
Setting the $$\beta $$to a small value ($$\beta = 0.01$$) makes them interpretable, as the produced heatmaps always have prominent maxima close to the prediction.  
At the same time, such small $$\beta$$ does not seem to have any effect on the final metrics, so its use can be avoided if interpretability is not needed.  
We have also tried the loss (15) from the algebraic triangulation instead of L1, but it performed worse in our experiments.

## 4. Experiments

**Human3.6M dataset.**

**CMU Panoptic dataset.**

## 5. Conclusion

We have presented two novel methods for the multi-view 3D human pose estimation based on learnable triangulation that achieve state-of-the-art performance on the Human3.6M dataset.  
The proposed solutions drastically reduce the number of views needed to achieve high accuracy, and produce smooth pose sequences on the CMU Panoptic dataset without any temporal processing (see our project page for demonstration), pointing that it can potentially improve the ground truth annotation of the dataset.  
An ability to transfer the trained method between setups is demonstrated for the CMU Panoptic -> Human3.6M pair.  
The volumetric triangulation strongly outperformed all other approaches both on CMU Panoptic and Human3.6M datasets.  
We speculate that due to its ability to learn a human pose prior this method is robust to occlusions and partial views of a person.  
Another important advantage of this method is that it explicitly takes the camera parameters as independent input.  
Finally, volumetric triangulation also generalizes to monocular images if human’s approximate position is known, producing results close to state of the art.

One of the major limitations of our approach is that it supports only a single person in the scene.  
This problem can be mitigated by applying available ReID solutions to the 2D detections of humans, however there might be more seamless solutions.  
Another major limitation of the volumetric triangulation approach is that it relies on the predictions of the algebraic triangulation.  
This leads to the need for having at least two camera views that observe the pelvis, which might be a problem for some applications.  
The performance of our method can also potentially be further improved by adding multi-stage refinement in a way similar to [18].
