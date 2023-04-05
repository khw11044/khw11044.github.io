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

>  in-the-wild에서 monocular 3D human pose estimation을 위한 One major challenge는 정확한 3D poses로 annotated된 제약 없는 images을 포함하는 training data의 획득이다.  
본 논문에서는 3D annotations이 필요하지 않고 unlabeled multi-view data에서 3D poses를 estimate하는 방법을 학습하는 weakly-supervised approach을 제안함으로써 이 과제를 해결한다, 이는 in-the-wild 환경에서 쉽게 획득할 수 있다.  
multi-view 일관성을 사용하여 weakly-supervised training을 가능하게 하는 새로운 end-to-end learning framework를 제안한다.  
multi-view consistency은 degenerated solutions에 노출되기 쉽기 때문에 2.5D pose representation을 채택하고 훈련된 모델의 예측이 모든 camera views에서 일관되고 그럴듯할 때만 최소화할 수 있는 새로운 objective function를 제안한다.  
우리는 두 개의 대규모 데이터 세트(Human3.6M and MPII-INF-3DHP)에 대해 제안된 접근 방식을 평가한다, 그것은 semi-/weakly supervised methods 중 state-of-the-art 성능을 달성한다.

## 1. Introduction

Learning to estimate 3D body pose from a single RGB image is of great interest for many practical applications.  
The state-of-the-art methods [6,16,17,28,32,39–41,52,53] in this area use images annotated with 3D poses and train deep neural networks to directly regress 3D pose from images.  
While the performance of these methods has improved significantly, their applicability in in-the-wild environments has been limited due to the lack of training data with ample diversity.  
The commonly used training datasets such as Human3.6M [10], and MPII-INF-3DHP [22] are collected in controlled indoor settings using sophisticated multi-camera motion capture systems.  

> Learning to estimate 3D body pose from a single RGB image은 많은 practical application에 큰 관심이다.  
이 영역의 state-of-the-art methods[6,16,17,28,32,39–41,52,53]은 3D pose로 annotated된 images를 사용하고 이미지에서 직접 3D pose를 regress하기 위해 deep neural networks을 훈련시킨다.  
이러한 방법의 성능은 크게 향상되었지만, 충분한 다양성을 가진 training data의 부족으로 인해 in-the-wild 환경에서의 적용 가능성은 제한되었다.  
Human3.6M [10]와 MPII-INF-3DHP [22]같이 일반적으로 사용되는 training datasets는 정교한 멀multi-camera motion capture systems을 사용하여 제어된 실내 설정으로 수집된다.

While scaling such systems to unconstrained outdoor environments is impractical, manual annotations are difficult to obtain and prone to errors.  
Therefore, current methods resort to existing training data and try to improve the generalizabilty of trained models by incorporating additional weak supervision in form of various 2D annotations for in-the-wild images [27,39,52].  
While 2D annotations can be obtained easily, they do not provide sufficient information about the 3D body pose, especially when the body joints are foreshortened or occluded.  
Therefore, these methods rely heavily on the ground-truth 3D annotations, in particular, for depth predictions.

> 이러한 systems에 제한되지 않은 실외 환경으로 확장하는 것은 비현실적이지만 수동 annotations는 어렵고 오류가 발생하기 쉽다.  
따라서, 현재 방법은 기존 훈련 데이터에 의존하며, in-the-wild images에 대한 다양한 2D annotations 형태의 추가적인 weak supervision을 통합하여 훈련된 모델의 일반화를 개선하려고 한다[27,39,52].  
2D annotations은 쉽게 얻을 수 있지만, 특히 body joints이 foreshortened되거나 occluded되었을 때 3D body pose에 대한 충분한 정보를 제공하지 않는다.  
따라서 이러한 방법은 특히 깊이 예측을 위해 ground-truth 3D annotations에 크게 의존한다.

Instead of using 3D annotations, in this work, we propose to use unlabeled multi-view data for training.  
We assume this data to be without extrinsic camera calibration.  
Hence, it can be collected very easily in any in-the-wild setting.  
In contrast to 2D annotations, using multi-view data for training has several obvious advantages e.g., ambiguities arising due to body joint occlusions as well as foreshortening or motion blur can be resolved by utilizing information from other views.  
There have been only few works [14, 29, 33, 34] that utilize multi-view data to learn monocular 3D pose estimation models.  
While the approaches [29,33] need extrinsic camera calibration, [33,34] require at least some part of their training data to be labelled with ground-truth 3D poses.  
Both of these requirements are, however, very hard to acquire for unconstrained data, hence, limit the applicability of these methods to controlled indoor settings.  
In [14], 2D poses obtained from multiple camera views are used to generate pseudo ground-truths for training.  
However, this method uses a pre-trained pose estimation model which remains fixed during training, meaning 2D pose errors remain unaddressed and can propagate to the generated pseudo ground-truths.

> 본 연구에서는 3D annotations을 사용하는 대신 unlabeled multi-view data를 훈련에 사용할 것을 제안한다.  
우리는 이 data가 extrinsic camera calibration 없이 사용된다고 가정한다.  
따라서, 그것은 어떤 in-the-wild setting에서도 매우 쉽게 수집될 수 있다.  
2D annotations과 대조적으로, 훈련에 multi-view data를 사용하는 것은 몇 가지 분명한 이점을 가지고 있다, 예를 들어, body joint occlusions뿐만 아니라 foreshortening 또는 motion blur로 인해 발생하는 모호성은 다른 views의 정보를 활용하여 해결할 수 있다.  
multi-view data를 활용하여 monocular 3D pose estimation model을 학습하는 작업은 [14, 29, 33, 34] 거의 없었다.  
이 approaches [29,33]은 extrinsic camera calibration이 필요하고 [33,34]는 training data에 적어도 labelled된 일부 ground-truth 3D poses가 있어야한다.  
그러나 이 두 가지 요건 모두 구속되지 않은 data에 대해 획득하기가 매우 어렵기 때문에 controlled indoor settings에 대한 이러한 방법의 적용 가능성을 제한한다.  
[14]에서, multiple camera views에서 얻은 2D poses는 training을 위한 pseudo ground-truths을 생성하는 데 사용된다.  
그러나 이 방법은 training 중에는 고정된 상태로 유지되는 pre-trained pose estimation model을 사용하며, 이는 2D pose errors가 해결되지 않은 상태로 남아 생성된 pseudo ground-truths에 전파될 수 있음을 의미한다.

In this work, we present a weakly-supervised approach for monocular 3D pose estimation that does not require any 3D pose annotations at all.  
For training, we only use a collection of unlabeled multi-view data and an independent collection of images annotated with 2D poses.  
An overview of the approach can be seen in Fig. 1.  
Given an RGB image as input, we train the network to predict a 2.5D pose representation [12] from which the 3D pose can be reconstructed in a fully-differentiable way.  
Given unlabeled multi-view data, we use a multi-view consistency loss which enforces the 3D poses estimated from different views to be consistent up to a rigid transformation.  
However, naively enforcing multi-view consistency can lead to degenerated solutions.  
We, therefore, propose a novel objective function which is constrained such that it can only be minimized when the 3D poses are predicted correctly from all camera views.  
The proposed approach can be trained in a fully end-to-end manner, it does not require extrinsic camera calibration and is robust to body part occlusions and truncations in the unlabeled multi-view data.  
Furthermore, it can also improve the 2D pose predictions by exploiting multi-view consistency during training.

> 본 연구에서는 3D pose annotations이 전혀 필요하지 않은 monocular 3D pose estimation을 위해 weakly-supervised approach를 제시한다.  
훈련을 위해  unlabeled multi-view data 모음과 independent collection of images annotated with 2D poses만 사용한다.  
접근법의 개요는  Fig. 1에서 볼 수 있다.  
RGB image가 input으로 주어지면, network를 훈련시켜 3D pose가 fully-differentiable way로 reconstructed될 수 있는 2.5D pose representation[12]을 예측한다.  
unlabeled multi-view data가 주어지면, 우리는 multi-view consistency loss을 사용하여 서로 다른 views에서 estimated된 3D poses가 rigid transformation까지 일관되도록 한다.  
그러나 multi-view consistency을 기본적으로 적용하면 degenerated solutions될 수 있다.  
따라서, 우리는 모든 camera views에서 3D poses가 올바르게 예측될 때만 minimized될 수 있도록 제한된 새로운 objective function를 제안한다.  
proposed approach는 fully end-to-end 방식으로 훈련될 수 있으며, extrinsic camera calibration이 필요하지 않으며 unlabeled multi-view data의 body part occlusions 및 truncations에 robust하다.  
또한 훈련 중에 multi-view consistency을 활용하여 2D pose predictions을 개선할 수도 있다.

We evaluate our approach on two large scale datasets where it outperforms existing methods for semi/weakly supervised methods by a large margin.  
We also show that the MannequinChallenge dataset [18], which provides in-the-wild videos of people in static poses, can be effectively exploited by our proposed method to improve the generalizability of trained models, in particular, when their is a significant domain gap between the training and testing environments.

> 우리는 ,두 개의 large scale datasets에 대한, semi/weakly supervised methods에 대한 기존 methods을 큰 폭으로 능가하는 our approach을 평가한다.  
우리는 또한 static poses로 사람들의 in-the-wild 비디오를 제공하는 MannequinChallenge dataset [18]가 trained models의 generalizability을 개선하기 위해 제안된 방법에 의해 효과적으로 활용될 수 있음을 보여준다. 특히 training 환경과 testing 환경 사이의 상당한 도메인 격차가 있는 경우.

## 2. Related Work

We discuss existing methods for monocular 3D human pose estimation with varying degree of supervision.

> 우리는 다양한 수준의 supervision을 통해 monocular 3D human pose estimation을 위한 기존 방법을 논의한다.

**Fully-supervised methods**  
aim to learn a mapping from 2D information to 3D given pairs of 2D-3D correspondences as supervision.  
The recent methods in this direction adopt deep neural networks to directly predict 3D poses from images [16, 17, 41, 53].  
Training the data hungry neural networks, however, requires large amounts of training images with accurate 3D pose annotations which are very hard to acquire, in particular, in unconstrained scenarios.  
To this end, the approaches in [5, 35, 45] try to augment the training data using synthetic images, however, still need real data to obtain good performance.  
More recent methods try to improve the performance by incorporating additional data with weak supervision i.e., 2D pose annotations [6, 28, 32, 39, 40, 52], boolean geometric relationship between body parts [27, 31, 37], action labels [20], and temporal consistency [2].  
Adverserial losses during training [50] or testing [44] have also been used to improve the performance of models trained on fully-supervised data.

> Fully-supervised methods는 supervision으로서 2D-3D correspondences의 pairs가 주어진 2D information에서 3D로 mapping을 학습하는 것을 목표로 한다.  
이 방향으로의 최근의 methods은 images에서 3D poses를 직접 예측하는 deep neural networks을 채택한다[16, 17, 41, 53].  
그러나 데이터가 부족한 신경망을 훈련하려면 특히 제한되지 않은 시나리오에서 획득하기 매우 어려운 정확한 3D pose annotations을 가지는 많은 양의 training images가 필요하다.  
이를 위해 [5, 35, 45]의 approaches는 합성 이미지를 사용하여 훈련 데이터를 증강하려고 시도하지만, 우수한 성능을 얻기 위해서는 여전히 실제 데이터가 필요하다.  
보다 최근의 방법은 2D pose annotations[6, 28, 32, 39, 40, 52], body parts 사이의 boolean geometric relationship[27, 31, 37], action labels [20] 및 temporal consistency [2]과 같은 weak supervision을 가진 추가 data를 통합하여 성능을 개선하려고 한다.  
training [50] 또는 testing [44] 동안의 Adverserial losses도 fully-supervised data에 대해 훈련된 모델의 성능을 향상시키는 데 사용되었다.

Other methods alleviate the need of 3D image annotations by directly lifting 2D poses to 3D without using any image information e.g., by learning a regression network from 2D joints to 3D [9, 21, 24] or by searching nearest 3D poses in large databases using 2D projections as the query [3, 11, 31].  
Since these methods do not use image information for 3D pose estimation, they are prone to reprojection ambiguities and can also have discrepancies between the 2D and 3D poses.

> 다른 방법은 이미지 정보를 사용하지 않고 2D poses를 3D로 직접 lifting하거나, 2D joints에서 3D [9, 21, 24]로 regression network를 학습하거나, query [3, 11, 31]로 2D projections을 사용하여 대형 databases에서 가장 가까운 3D poses를 검색함으로써 3D image annotations의 필요성을 완화한다.  
이러한 방법은 image information을 3D pose estimation에 사용하지 않기 때문에 모호성을 reprojection하기 쉽고 2D poses와 3D poses 간에 불일치를 가질 수 있다.

In contrast, in this work, we present a method that combines the benefits of both paradigms i.e., it estimates 3D pose from an image input, hence, can handle the reprojection ambiguities, but does not require any images with 3D pose annotations.

> 대조적으로, 본 연구에서는 두 패러다임의 이점을 결합한 방법을 제시한다, 즉, image input에서 3D pose를 추정하므로 reprojection ambiguities을 처리할 수 있지만 3D pose annotations이 있는 이미지는 필요하지 않다.

**Semi-supervised methods**  
require only a small subset of training data with 3D annotations and assume no or weak supervision for the rest.  
The approaches [33,34,51] assume that multiple views of the same 2D pose are available and use multi-view constraints for supervision.  
Closest to our approach in this category is [34] in that it also uses multiview consistency to supervise the pose estimation model.  
However, their method is prone to degenerated solutions and its solution space cannot be constrained easily.  
Consequently, the requirement of images with 3D annotations is inevitable for their approach.  
In contrast, our method is weakly-supervised.  
We constrain the solution space of our method such that the 3D poses can be learned without any 3D annotations.  

> Semi-supervised methods는 3D annotations이 있는 training data의 small subset만 필요하며 나머지는 no supervision 또는 weak supervision이라고 가정한다.  
approaches [33,34,51]는 동일한 2D pose의 multiple views를 사용할 수 있다고 가정하고 supervision을 위해 multi-view constraints을 사용한다.  
이 범주에서 our approach에 가장 가까운 것은 pose estimation model을 supervise하기 위해 multiview consistency도 사용한다는 점에서 [34]이다.  
그러나 이러한 방법은 degenerated solutions를 초래하기 쉬우며 solution space를 쉽게 제한할 수 없다.  
따라서, 3D annotations을 가진 images의 요구사항은 접근에 불가피하다.  
대조적으로, 우리의 방법은 weakly-supervised이다.  
우리는 3D poses가 3D annotations 없이 학습될 수 있도록 our method의 solution space를 제한한다.  

In contrast to [34], our approach can easily be applied to in-the-wild scenarios as we will show in our experiments.  
The approaches [43, 48] use 2D pose annotations and re-projection losses to improve the performance of models pre-trained using synthetic data.  
In [19], a pre-trained model is iteratively improved by refining its predictions using temporal information and then using them as supervision for next steps.  
The approach in [30] estimates the 3D poses using a sequence of 2D poses as input and uses a re-projection loss accumulated over the entire sequence for supervision.  
While all of these methods demonstrate impressive results, their main limiting factor is the need of ground-truth 3D data.

> [34]와 대조적으로, our approach는 우리의 실험에서 보여줄 것처럼 in-the-wild scenarios에 쉽게 적용될 수 있다.  
approaches [43, 48]는 합성 데이터를 사용하여 pre-trained된 models의 성능을 향상시키기 위해 2D pose annotations과 re-projection losses을 사용한다.  
[19]에서 pre-trained model은 temporal information을 사용하여 예측을 정제하고 next steps에 대한 supervision으로 사용하여 반복적으로 개선된다.  
[30]의 approach는 sequence of 2D poses를 input으로 사용하여 3D poses를 추정하며 entire sequence에 걸쳐 누적된 re-projection loss을 supervision을 위해 사용한다.  
이러한 모든 방법은 인상적인 결과를 보여주지만, 주요 제한 요인은 ground-truth 3D data의 필요성이다.

**Weakly-supervised methods**  
do not require paired 2D-3D data and only use weak supervision in form of motion capture data [42], images/videos with 2D annotations [25, 47], collection of 2D poses [4, 7, 46], or multi-view images [14, 29].  
Our approach also lies in this paradigm and learns to estimate 3D poses from unlabeled multi-view data.  
In [42], a probabilistic 3D pose model learned using motion-capture data is integrated into a multi-staged 2D pose estimation model to iteratively refine 2D and 3D pose predictions.   
The approach [25] uses a re-projection loss to train the pose estimation model using images with only 2D pose annotations.  
Since re-projection loss alone is insufficient for training, they factorize the problem into the estimation of view-point and shape parameters and provide inductive bias via a canonicalization loss.  
Similar in spirit, the approaches [4,7,46] use collection of 2D poses with reprojection loss for training and use adversarial losses to distinguish between plausible and inplausible poses.   

> Weakly-supervised methods는 paired 2D-3D data를 필요로 하지 않으며 motion capture data[42], images/videos with 2D annotations [25, 47], collection of 2D poses [4, 7, 46] 또는 multi-view images [14, 29]의 form으로 weak supervision만 사용한다.
Our approach는 또한 이 패러다임에 있으며 unlabeled multi-view data에서 3D poses를 추정하는 방법을 배운다.
[42]에서 motion-capture data를 사용하여 학습한 probabilistic 3D pose model을 multi-staged 2D pose estimation model에 통합하여 2D 및 3D pose 예측을 반복적으로 세분화한다.
approach [25]는 re-projection loss을 사용하여 2D pose annotations만 있는 images를 사용하여 pose estimation model을 훈련시킨다.
re-projection loss만으로는 훈련에 충분하지 않기 때문에, 이들은 문제를 view-point 및 shape parameters의 estimation으로 인수하고 canonicalization loss를 통해 inductive bias를 제공한다.
비슷한 spirit으로, approaches[4,7,46]는 훈련을 위해 reprojection loss과 함께 collection of 2D poses을 사용하고 타당하지 않은 poses와 신뢰할 수 없는 poses를 구별하기 위해 adversarial losses을 사용한다.

In [47], non-rigid structure from motion is used to learn a 3D pose estimator from videos with 2D pose annotations.  
The closest to our work are the approaches of [14, 29] in that they also use unlabeled multi-view data for training.  
The approach of [29], however, requires calibrated camera views that are very hard to acquire in unconstrained environments.  
The approach [14] estimates 2D poses from multi-view images and reconstructs corresponding 3D pose using Epipolar geometry.  
The reconstructed poses are then used for training in a fully-supervised way.

> [47]에서는 non-rigid structure from motion를 사용하여 2D pose annotations이 있는 videos에서 3D pose estimator를 학습한다.  
작업에 가장 가까운 것은 [14, 29]의 접근 방식이며, 이 접근 방식은 unlabeled multi-view data도 훈련에 사용한다는 것이다.  
그러나 [29]의 approach는 제한되지 않은 환경에서 획득하기 매우 어려운 calibrated camera views를 필요로 한다.  
approach[14]는 multi-view images에서 2D poses를 추정하고 Epipolar geometry를 사용하여 해당 3D pose를 reconstructs한다.  
그런 다음 reconstructed poses는 fully-supervised way으로 훈련에 사용된다.

The main drawback of this method is that the 3D poses remain fixed throughout the training, and the errors in 3D reconstruction directly propagate to the trained models.  
This is, particularly, problematic if the multi-view data is captured in challenging outdoor environments where 2D pose estimation may fail easily.  
In contrast, in this work, we propose an end-to-end learning framework which is robust to challenges posed by the data captured in in-the-wild scenarios.  
It is trained using a novel objective function which can simultaneously optimize for 2D and 3D poses.  
In contrast to [14], our approach can also improve 2D predictions using unlabeled multi-view data.  
We evaluate our approach on two challenging datasets where it outperforms existing methods for semi/weakly supervised learning by a large margin.

> 이 method의 main drawback은 3D poses가 훈련 내내 고정된 상태로 유지되고 3D reconstruction에서 errors는 훈련된 모델에 직접 전파된다는 것이다.  
이는 특히 2D pose estimation이 쉽게 실패할 수 있는 까다로운 실외 환경에서 multi-view data를 캡처하는 경우 문제가 된다.  
이와는 대조적으로, 본 연구에서는 in-the-wild scenarios에서 캡처된 data에 의해 발생하는 문제에 robust한 end-to-end learning framework를 제안한다.  
2D 및 3D poses에 동시에 optimize할 수 있는 새로운 objective function를 사용하여 훈련된다.  
[14]와 대조적으로, our approach은 unlabeled multi-view data를 사용하여 2D predictions을 개선할 수도 있다.  
우리는 semi/weakly supervised learning에 대한 기존 방법을 큰 폭으로 능가하는 두 가지 까다로운 datasets에 대한 our approach를 평가한다.

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
