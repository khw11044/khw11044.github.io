---
layout: post
bigtitle:  "Self-Supervised Multi-View Synchronization Learning for 3D Pose Estimation"
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



# Self-Supervised Multi-View Synchronization Learning for 3D Pose Estimation

Jenni, Simon, and Paolo Favaro. "Self-Supervised Multi-View Synchronization Learning for 3D Pose Estimation." Proceedings of the Asian Conference on Computer Vision. 2020. [paper](https://openaccess.thecvf.com/content/ACCV2020/html/Jenni_Self-Supervised_Multi-View_Synchronization_Learning_for_3D_Pose_Estimation_ACCV_2020_paper.html)

* toc
{:toc}

## Abstract

Current state-of-the-art methods cast monocular 3D human pose estimation as a learning problem by training neural networks on large data sets of images and corresponding skeleton poses.  
In contrast, we propose an approach that can exploit small annotated data sets by fine-tuning networks pre-trained via self-supervised learning on (large) unlabeled data sets.  
To drive such networks towards supporting 3D pose estimation during the pre-training step, we introduce a novel self-supervised feature learning task designed to focus on the 3D structure in an image.  
We exploit images extracted from videos captured with a multi-view camera system.  
The task is to classify whether two images depict two views of the same scene up to a rigid transformation.  
In a multi-view data set, where objects deform in a non-rigid manner, a rigid transformation occurs only between two views taken at the exact same time, i.e., when they are synchronized.  
We demonstrate the effectiveness of the synchronization task on the Human3.6M data set and achieve state-of-the-art results in 3D human pose estimation.

## 1. Introduction

The ability to accurately reconstruct the 3D human pose from a single real image opens a wide variety of applications including computer graphics animation, postural analysis and rehabilitation, human-computer interaction, and image understanding [1].  
State-of-the-art methods for monocular 3D human pose estimation employ neural networks and require large training data sets, where each sample is a pair consisting of an image and its corresponding 3D pose annotation [2].  
The collection of such data sets is expensive, time-consuming, and, because it requires controlled lab settings, the diversity of motions, viewpoints, subjects, appearance and illumination, is limited (see Fig. 1).  
Ideally, to maximize diversity, data should be collected in the wild.  
However, in this case precise 3D annotation is difficult to obtain and might require costly human intervention.

In this paper, we overcome the above limitations via self-supervised learning (SSL).  
SSL is a method to build powerful latent representations by training a neural network to solve a so-called pretext task in a supervised manner, but without manual annotation [3].  
The pretext task is typically an artificial problem, where a model is asked to output the transformation that was applied to the data.  
One way to exploit models trained with SSL is to transfer them to some target task on a small labeled data set via fine-tuning [3].  
The performance of the transferred representations depends on how related the pretext task is to the target task.  
Thus, to build latent representations relevant to 3D human poses, we propose a pretext task that implicitly learns 3D structures.  
To collect data suitable to this goal, examples from nature point towards multi-view imaging systems.  
In fact, the visual system in many animal species hinges on the presence of two or multiple eyes to achieve a 3D perception of the world [4, 5].  
3D perception is often exemplified by considering two views of the same scene captured at the same time and by studying the correspondence problem.  
Thus, we take inspiration from this setting and pose the task of determining if two images have been captured at exactly the same time.  
In general, the main difference between two views captured at the same time and when they are not, is that the former are always related by a rigid transformation and the latter is potentially not (e.g., in the presence of articulated or deformable motion, or multiple rigid motions).  
Therefore, we propose as pretext task the detection of synchronized views, which translates into a classification of rigid versus non-rigid motion.


![Fig1](/assets/img/Blog/papers/Pose/Self-Supervised_Multi-View_Synchronization_Learning/Fig1.PNG)

As shown in Fig. 1, we aim to learn a latent representation that can generalize across subjects and that is sensitive to small pose variations.  
Thus, we train our model with pairs of images, where the subject identity is irrelevant to the pretext task and where the difference between the task categories are small pose variations.  
To do so, we use two views of the same subject as the synchronized (i.e., rigid motion) pair and two views taken at different (but not too distant) time instants as the unsynchronized (i.e., non-rigid motion) pair.<sup>1</sup>  
Since these pairs share the same subject (as well as the appearance) in all images, the model cannot use this as a cue to learn the correct classification.  
Moreover, we believe that the weight decay term that we add to the loss may also help to reduce the effect of parameters that are subject-dependent, since there are no other incentives from the loss to preserve them.   
Because the pair of unsynchronized images is close in time, the deformation is rather small and forces the model to learn to discriminate small pose variations.  
Furthermore, to make the representation sensitive to left-right symmetries of the human pose, we also introduce in the pretext task as a second goal the classification of two synchronized views into horizontally flipped or not.  
This formulation of the SSL task allows to potentially train a neural network on data captured in the wild simply by using a synchronized multi-camera system.  
As we show in our experiments, the learned representation successfully embeds 3D human poses and it further improves if the background can also be removed from the images.  
We train and evaluate our SSL pre-training on the Human3.6M data set [2], and find that it yields state-of-the-art results when compared to other methods under the same training conditions.  
We show quantitatively and qualitatively that our trained model can generalize across subjects and is sensitive to small pose variations.  
Finally, we believe that this approach can also be easily incorporated in other methods to exploit additional available labeling (e.g., 2D poses). Code will be made available on our project page [https://sjenni.github.io/multiview-sync-ssl](https://sjenni.github.io/multiview-sync-ssl).

Our contributions are:  
1) A novel self-supervised learning task for multi-view data to recognize when two views are synchronized and/or flipped;  
2) Extensive ablation experiments to demonstrate the importance of avoiding shortcuts via the static background removal and the effect of different feature fusion strategies;  
3) State-of-the-art performance on 3D human pose estimation benchmarks.

## 2 Prior work

In this section, we briefly review literature in self-supervised learning, human pose estimation, representation learning and synchronization, that is relevant to our approach.

**Self-supervised learning.**  
Self-supervised learning is a type of unsupervised representation learning that has demonstrated impressive performance on image and video benchmarks.  
These methods exploit pretext tasks that require no human annotation, i.e., the labels can be generated automatically.  
Some methods are based on predicting part of the data [6–8], some are based on contrastive learning or clustering [9–11], others are based on recognizing absolute or relative transformations [12–14].  
Our task is most closely related to the last category since we aim to recognize a transformation in time between two views.

**Unsupervised learning of 3D.**  
Recent progress in unsupervised learning has shown promising results for learning implicit and explicit generative 3D models from natural images [15–17].  
The focus in these methods is on modelling 3D and not performance on downstream tasks.  
Our goal is to learn general purpose 3D features that perform well on downstream tasks such as 3D pose estimation.  
Synchronization. Learning the alignment of multiple frames taken from different views is an important component of many vision systems.  
Classical approaches are based on fitting local descriptors [18–20].  
More recently, methods based on metric learning [21] or that exploit audio [22] have been proposed.  
We provide a simple learning based approach by posing the synchronization problem as a binary classification task.  
Our aim is not to achieve synchronization for its own sake, but to learn a useful image representation as a byproduct.

**Monocular 3D pose estimation.**  
State-of-the-art 3D pose estimation methods make use of large annotated in-the-wild 2D pose data sets [23] and data sets with ground truth 3D pose obtained in indoor lab environments.  
We identify two main categories of methods:  
1) Methods that learn the mapping to 3D pose directly from images [24–32] often trained jointly with 2D poses [33–37], and  
2) Methods that learn the mapping of images to 3D poses from predicted or ground truth 2D poses [38–46].  
To deal with the limited amount of 3D annotations, some methods explored the use of synthetic training data [47–49].  
In our transfer learning experiments, we follow the first category and predict 3D poses directly from images.   
However, we do not use any 2D annotations.

**Weakly supervised methods.**   
Much prior work has focused on reducing the need for 3D annotations.  
One approach is weak supervision, where only 2D annotation is used. These methods are typically based on minimizing the re-projection error of a predicted 3D pose [50–56].  
To resolve ambiguities and constrain the 3D, some methods require multi-view data [50–52], while others rely on unpaired 3D data used via adversarial training [54, 55].  
[53] solely relies on 2D annotation and uses an adversarial loss on random projections of the 3D pose.  
Our aim is not to rely on a weaker form of supervision (i.e., 2D annotations) and instead leverage multi-view data to learn a representation that can be transferred on a few annotated examples to the 3D estimation tasks with a good performance.

**Self-supervised methods for 3D human pose estimation.**
Here we consider methods that do not make use of any additional supervision, e.g., in the form of 2D pose annotation.  
These are methods that learn representations on unlabelled data and can be transferred via fine-tuning using a limited amount of 3D annotation. Rhodin et al. [57] learn a 3D representation via novel view synthesis, i.e., by reconstructing one view from another.  
Their method relies on synchronized multi-view data, knowledge of camera extrinsics and background images.  
Mitra et al. [58] use metric learning on multi-view data.  
The distance in feature space for images with the same pose, but different viewpoint is minimized while the distance to hard negatives sampled from the mini-batch is maximized.  
By construction, the resulting representation is view invariant (the local camera coordinate system is lost) and the transfer can only be performed in a canonical, rotation-invariant coordinate system.  
We also exploit multi-view data in our pre-training task. In contrast to [57], our task does not rely on knowledge of camera parameters and unlike [58] we successfully transfer our features to pose prediction in the local camera system.

## 3 Unsupervised learning of 3D pose-discriminative features

### 3.1 Classifying synchronized and flipped views

**Synchronized pairs.**  

**Unsynchronized pairs.**

**Flipped pairs.**

### 3.2 Static backgrounds introduce shortcuts

### 3.3 Implementation

### 3.4 Transfer to 3D human pose estimation

## 4 Experiments

**Dataset.**

**Metrics.**

### 4.1 Ablations

**(a)-(d) Influence of background removal:**

**(e)-(f) Combination of self-supervision signals:**

**(h)-(k) Fusion of features:**

**(l)-(m) Necessity of multi-view data:**

### 4.2 Comparison to prior work

### 4.3 Evaluation of the synchronization task

**Generalization across subjects.**

**Synchronization performance on test subjects.**

## 5 Conclusions
