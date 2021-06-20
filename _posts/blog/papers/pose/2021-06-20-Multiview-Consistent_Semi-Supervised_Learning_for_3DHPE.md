---
layout: post
bigtitle:  "Multiview-Consistent Semi-Supervised Learning for 3D Human Pose Estimation"
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



# Multiview-Consistent Semi-Supervised Learning for 3D Human Pose Estimation

Mitra, Rahul, et al. "Multiview-Consistent Semi-Supervised Learning for 3D Human Pose Estimation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.[paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Mitra_Multiview-Consistent_Semi-Supervised_Learning_for_3D_Human_Pose_Estimation_CVPR_2020_paper.html)

* toc
{:toc}

## Abstract

The best performing methods for 3D human pose estimation from monocular images require large amounts of in-the-wild 2D and controlled 3D pose annotated datasets which are costly and require sophisticated systems to acquire.  
To reduce this annotation dependency, we propose Multiview-Consistent Semi Supervised Learning (MCSS) framework that utilizes similarity in pose information from unannotated, uncalibrated but synchronized multi-view videos of human motions as additional weak supervision signal to guide 3D human pose regression.  
Our framework applies hard-negative mining based on temporal relations in multi-view videos to arrive at a multi-view consistent pose embedding.  
When jointly trained with limited 3D pose annotations, our approach improves the baseline by 25% and state-of-the-art by 8.7%, whilst using substantially smaller networks.  
Lastly, but importantly, we demonstrate the advantages of the learned embedding and establish view-invariant pose retrieval benchmarks on two popular, publicly available multi-view human pose datasets, Human 3.6M and MPI-INF-3DHP, to facilitate future research.

> monocular images에서 3D human pose estimation을 위한 best performing methods는 많은 양의 in-the-wild 2D와 controlled 3D pose annotated datasets를 필요로 하는데, 이 datasets는 비용이 많이 들고 정교한 시스템을 획득해야 한다.  
이러한 annotation 의존성을 줄이기 위해, 우리는 3D human pose regression를 guide하는 추가적인 weak supervision signal로 human motions의 unannotated되고, uncalibrated지만 synchronized된 multi-view videos의 pose information의 유사성을 활용하는 Multiview-Consistent Semi Supervised Learning (MCSS) framework를 제안한다.  
우리의 framework는 multi-view videos의 시간적 관계에 기초한 hard-negative mining을 적용하여 multi-view 일관된 pose embedding에 도달한다.  
limited 3D pose annotations과 jointly하게 훈련할 때, 우리의 approach는 훨씬 더 작은 networks를 사용하면서 기준을 25% 개선하고 state-of-the-art를 8.7% 향상시킨다.  
마지막으로, 그러나 중요한 것은 학습된 임베딩의 장점을 입증하고 향후 연구를 용이하게 하기 위해 공개 가능한 두 가지 인기 있는 multi-view human pose datasets, Human 3.6M 및 MPI-INF-3DHP에 대한 view-invariant pose retrieval benchmarks를 설정한다.


## Introduction

Over the years, the performance of monocular 3D human pose estimation has improved significantly due to increasingly sophisticated CNN models [55, 35, 46, 45, 30, 49].  
For training, these methods depend on the availability of large-scale 3D-pose annotated data, which is costly and challenging to obtain, especially under in-the-wild setting for articulated poses.  
The two most popular 3D-pose annotated datasets, Human3.6M [15] (3.6M samples) and MPIINF- 3DHP [29] (1.3M samples), are biased towards indoor like environment with uniform background and illumination.  
Therefore, 3D-pose models trained on these datasets don’t generalize well for real-world scenarios [8, 55].

> 몇 년 동안, 점점 더 정교해지는 CNN models 덕분에 monocular 3D human pose estimation의 성능이 크게 향상되었다[55, 35, 46, 45, 30, 49].  
훈련을 위해, 이러한 방법은 특히 articulated poses에 대한 in-the-wild setting에서 얻는 데 비용이 많이 들고 어려운 large-scale 3D-pose annotated data의 가용성에 따라 달라진다.  
가장 널리 사용되는 두 개의 most popular 3D-pose annotated datasets인 Human3.6M [15] (3.6M 샘플)과 MPIINF-3DHP [29] (1.3M 샘플)은 균일한 배경과 조도를 가진 실내와 같은 환경에 치우쳐 있다.  
따라서, 이러한 datasets에 대해 훈련된 3D 포즈 모델은 실제 시나리오에서는 잘 일반화되지 않는다 [8, 55].


Limited training data, or costly annotation, poses serious challenges to not only deep-learning based methods, but other machine-learning methods as well.  
Semi-supervised approaches [10, 22, 24, 14] have been extensively used in the past to leverage large-scale unlabelled datasets along with small labelled dataset to improve performance.  
Semi-supervised methods try to exploit the structure/invariances in the data to generate additional learning signals for training.  
Unlike classical machine-learning models that use fixed feature representation, deep-learning models can also learn a suitable feature representation from data as part of the training process.  
This unique ability calls for semi-supervised approaches to encourage better feature representation learning from large-scale unlabelled data for generalization.  
Intuitively it’s more appealing to leverage semi-supervised training signals that are more relevant to the final application.  
Therefore, given the vast diversity of computer vision tasks, it remains an exciting area of research to innovate novel semi-supervision signals.

> Limited training data 또는 costly annotation이 eep-learning based methods뿐만 아니라 다른  machine-learning methods에도 심각한 문제를 제기한다.  
Semi-supervised approaches [10, 22, 24, 14]은 과거에 성능을 향상시키기 위해 large-scale unlabelled datasets와 함께 널리 사용되었다.  
Semi-supervised methods은 data의 structure/invariances을 활용하여 훈련을 위한 additional learning signals를 생성하려고 한다.  
fixed feature representation을 사용하는 classical machine-learning models과 달리, deep-learning models은 training process의 part로 data로부터 적절한 feature representation을 학습할 수도 있다.  
이 unique ability는 일반화를 위해 large-scale unlabelled data에서 더 나은 feature representation learning을 장려하기 위한 semi-supervised approaches을 요구한다.  
직관적으로 최종 애플리케이션과 더 관련성이 높은 semi-supervised training signals를 활용하는 것이 더 매력적이다.  
따라서, computer vision tasks의 광범위한 다양성을 고려할 때, novel semi-supervision signals를 혁신하는 것은 여전히 흥미로운 연구 분야로 남아 있다.

To this end, we leverage projective multiview consistency to create a novel metric-learning based semi-supervised framework for 3D human-pose estimation.  
Multiview consistency has served as a fundamental paradigm in computer vision for more than 40 years and gave rise to some of the most used algorithms such as stereo [42], structure from motion [20], motion capture [32], simultaneous localization and mapping [4], etc.  
From human pose estimation perspective, the intrinsic 3D-pose of the human-body remains the same across multiple different views.  
Therefore, a deep-CNN should ideally be able to map 2D-images corresponding to a common 3D-pose, captured from different view points, to nearby points in an embedding space.  
Intuitively, such a deep-CNN is learning feature representations that are invariant to different views of the human-pose.  
Therefore, we posit that perhaps it can learn to project 2D images, from different viewpoints, into a canonical 3D-pose space in $$\mathcal{R}^N$$.  

> 이를 위해, 우리는 projective multiview consistency을 활용하여 3D human-pose estimation을 위한 novel metric-learning based semi-supervised framework를 만든다.  
Multiview consistency은 40년 이상 computer vision의 기본 패러다임으로 작용했으며 stereo[42], structure from motion[20], motion capture[32], simultaneous localization 및 mapping[4] 등과 같은 가장 많이 사용되는 algorithms 중 일부를 생성했다.  
human pose estimation perspective에서, human-body의 본질적인 3D-pose는 multiple different views에서 동일하게 유지된다.  
따라서 deep-CNN은 different view points에서 캡처된 common 3D-pose에 해당하는 2D-images를 embedding space의 nearby points에 매핑할 수 있는 이상적인 방법이 되어야 한다.  
직관적으로, 그러한 deep-CNN은 human-pose의 different views에 불변하는 feature representations을 학습하고 있다.  
따라서, 우리는 different viewpoints에서, $$\mathcal{R}^N$$의 canonical 3D-pose space에 2D images을 project하는 것을 배울 수 있다고 가정한다.

In Fig. 1b, we show a few embedding distances between different images from the Human 3.6M dataset [15] and provide empirical evidence to the aforementioned hypothesis via a novel cross view pose-retrieval experiment.  
Unfortunately, embedding vectors, $$x$$, from such a space do not translate directly to the 3D coordinates of human-pose.  
Therefore, we learn another transformation function from embedding to pose space and regress with small 3D-pose supervision while training.  
Since, the embedding is shared between the pose supervision and semi-supervised metric-learning, it leads to better generalizeable features for 3D-pose estimation.  
We name our proposed framework as Multi view Consistent Semi-Supervised learning, or MCSS for short.

> Fig. 1b에서는 Human 3.6M dataset [15]와 다른 images 사이의 few embedding distances를 보여주고 novel cross view pose-retrieval experiment을 통해 앞에서 언급한 가설에 대한 경험적 증거를 제공한다.  
불행히도, 그러한 space에서 vectors, $$x$$를 embedding하는것은 human-pose의 3D coordinates로 직접 translate되지 않는다.  
따라서 embedding에서 pose space에 이르는 another transformation function를 학습하고 훈련하는 동안 small 3D-pose supervision으로 regress한다.  
embedding은 pose supervision과 semi-supervised metric-learning 간에 공유되므로 3D-pose estimation을 위해 더 나은 generalizeable features으로 이어진다.  
우리는 제안된 framework를 Multi view Consistent Semi-Supervised learning 또는 줄여서 MCSS로 이름을 지었다.

The proposed framework fits really well with the practical requirements of our problem because it’s relatively easy to obtain real-world time-synchronized video streams of humans from multiple viewpoints vs. setting up capture rigs for 3D-annotated data out in-the-wild. An alternative approach could be to setup a calibrated multi-camera capture rig in-the-wild and use triangulation from 2D-pose annotated images to obtain 3D-pose. But, it still requires handannotated 2D-poses or an automated 2D-pose generation system . In [19], a pre-trained 2D-pose network has been used to generate pseudo 3D-pose labels for training a 3Dpose network. Yet another approach exploits relative camera extrinsics for cross-view image generation via a latent embedding [39]. We, on the other hand, don’t assume such requirements to yield a more practical solution for the limited data challenge.

We use MCSS to improve 3D-pose estimation performance with limited 3D supervision. In Sec. 5, we show the performance variation as 3D supervision is decreased. Sec. 6 demonstrates the richness of view-invariant MCSS embedding for capturing human-pose structure with the help of a carefully designed cross-view pose-retrieval task on Human3.6M and MPI-INF-3DHP to serve as a benchmark for future research in this direction. To summarize our contribution, we

+ Propose a novel Multiview-Consistent Semi-Supervised learning framework for 3D-human-pose estimation.

+ Achieve state-of-the-art performance on Human 3.6M dataset with limited 3D supervision.

+ Formulate a cross-view pose-retrieval benchmark on Human3.6M and MPI-INF-3DHP datasets.
