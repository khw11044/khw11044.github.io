---
layout: post
bigtitle:  "[Pose]Fast and Robust Multi-Person 3D Pose Estimation from Multiple Views"
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



# [Pose]Fast and Robust Multi-Person 3D Pose Estimation from Multiple Views

IEEE 2019 [paper]()

* toc
{:toc}

## Abstract

This paper addresses the problem of 3D pose estimation for multiple people in a few calibrated camera views.  
The main challenge of this problem is to find the cross-view correspondences among noisy and incomplete 2D pose predictions.  
Most previous methods address this challenge by directly reasoning in 3D using a pictorial structure model, which is inefficient due to the huge state space.  
We propose a fast and robust approach to solve this problem.  
Our key idea is to use a multi-way matching algorithm to cluster the detected 2D poses in all views.  
Each resulting cluster encodes 2D poses of the same person across different views and consistent correspondences across the keypoints, from which the 3D pose of each person can be effectively inferred.  
The proposed convex optimization based multi-way matching algorithm is efficient and robust against missing and false detections, without knowing the number of people in the scene.  
Moreover, we propose to combine geometric and appearance cues for cross-view matching.  
The proposed approach achieves significant performance gains from the state-of-the-art (96.3% vs. 90.6% and 96.9% vs. 88% on the Campus and Shelf datasets, respectively), while being efficient for real-time applications.

<details>

> 본 논문은 보정된 몇 개의 카메라 views에서 여러 사람에 대한 3D pose estimation 문제를 다룬다.  
이 문제의 주요 과제는 노이즈가 많고 불완전한 2D pose 예측에서 cross-view correspondences을 찾는 것이다.  
대부분의 이전 방법은 pictorial structure model을 사용하여 3D로 직접 추론함으로써 이 과제를 해결하는데, 이는 거대한 상태 공간으로 인해 비효율적이다.  
우리는 이 문제를 해결하기 위해 빠르고 강력한 접근법을 제안한다.  
우리의 핵심 아이디어는 다방향 매칭 알고리듬을 사용하여 모든 보기에서 탐지된 2D 포즈를 클러스터링하는 것이다.  
각 결과 클러스터는 서로 다른 관점에 걸쳐 동일인의 2D 포즈를 인코딩하고, 핵심 사항에 걸쳐 일관된 대응 방식을 통해 각 개인의 3D 포즈를 효과적으로 추론할 수 있다.  
제안된 볼록 최적화 기반 다방향 매칭 알고리듬은 장면의 사람 수를 모르는 상태에서 누락 및 거짓 탐지에 대해 효율적이고 강력하다.  
또한 교차 뷰 일치를 위해 지오메트릭과 외관 단서를 결합할 것을 제안한다.  
제안된 접근 방식은 실시간 애플리케이션에 효율적이면서 최첨단(캠퍼스 및 쉘프 데이터 세트에서 각각 96.3% 대 90.6%, 96.9% 대 88%)으로부터 상당한 성능 향상을 달성한다.

<br>
<br>

</details>

## 1. Introduction

Recovering 3D human pose and motion from videos has been a long-standing problem in computer vision, which has a variety of applications such as human-computer interaction, video surveillance and sports broadcasting.  
In particular, this paper focuses on the setting where there are multiple people in a scene, and the observations come from a few calibrated cameras (Figure 1).  
While remarkable advances have been made in multi-view reconstruction of a human body, there are fewer works that address a more challenging setting where multiple people interact with each other in crowded scenes, in which there are significant occlusions.

<details>

> 비디오에서 3D 인간 자세와 움직임을 복구하는 것은 인간과 컴퓨터의 상호 작용, 비디오 감시, 스포츠 방송과 같은 다양한 응용 프로그램을 가지고 있는 컴퓨터 비전에서 오랜 문제였다. 특히 본 논문은 한 장면에 여러 사람이 있는 설정에 초점을 맞추고 있으며, 관측치는 보정된 몇 대의 카메라에서 나온다(그림 1). 인체의 다중 뷰 재구성에서 주목할 만한 발전이 이루어졌지만, 복잡한 장면에서 여러 사람이 상호 작용하는 보다 어려운 설정을 다루는 작품은 적으며, 여기에는 상당한 혼선이 있다.

<br>
<br>

</details>

Existing methods typically solve this problem in two stages.  
The first stage detects human-body keypoints or parts in separate 2D views, which are aggregated in the second stage to reconstruct 3D poses.  
Given the fact that deep-learning based 2D keypoint detection techniques have achieved remarkable performance [8, 30], the remaining challenge is to find cross-view correspondences between detected keypoints as well as which person they belong to.  
Most previous methods [1, 2, 21, 12] employ a 3D pictorial structure (3DPS) model that implicitly solves the correspondence problem by reasoning about all hypotheses in 3D that are geometrically compatible with 2D detections.  
However, this 3DPS-based approach is computationally expensive due to the huge state space.  
In addition, it is not robust particularly when the number of cameras is small, as it only uses multi-view geometry to link the 2D detections across views, or in other words, the appearance cues are ignored.

<details>

> 기존 방법은 일반적으로 이 문제를 두 단계로 해결한다. 첫 번째 단계는 3D 포즈를 재구성하기 위해 두 번째 단계에서 집계된 인간-신체 키포인트 또는 부품을 별도의 2D 보기에서 감지한다. 딥 러닝 기반 2D 키포인트 감지 기술이 주목할 만한 성능을 달성했다는 사실을 고려할 때 [8, 30], 남은 과제는 감지된 키포인트뿐만 아니라 어떤 사람에 속하는지를 교차 조회하는 것이다. 대부분의 이전 방법[1, 2, 21, 12]은 2D 검출과 기하학적으로 호환되는 3D의 모든 가설을 추론하여 대응 문제를 암묵적으로 해결하는 3D 그림 구조(3DPS) 모델을 사용한다. 그러나 이 3DPS 기반 접근 방식은 거대한 상태 공간 때문에 계산 비용이 많이 든다. 또한 멀티 뷰 지오메트리만 사용하여 2D 탐지를 뷰 간에 연결하거나, 다시 말하면 외관 단서가 무시되므로 카메라 수가 적을 때는 특히 강력하지 않다.

<br>
<br>

</details>

In this paper, we propose a novel approach for multi-person 3D pose estimation.  
The proposed approach solves the correspondence problem at the body level by matching detected 2D poses among multiple views, producing clusters of 2D poses where each cluster includes 2D poses of the same person in different views.  
Then, the 3D pose can be inferred for each person separately from matched 2D poses, which is much faster than joint inference of multiple poses thanks to the reduced state space.

<details>

> 본 논문에서는 다중 사용자 3D 포즈 추정을 위한 새로운 접근법을 제안한다. 제안된 접근 방식은 여러 보기 간에 탐지된 2D 포즈를 일치시켜 신체 수준에서 대응 문제를 해결하여 각 클러스터가 서로 다른 보기에서 동일인의 2D 포즈를 포함하는 2D 포즈 클러스터를 생성한다. 그런 다음, 3D 포즈는 일치하는 2D 포즈와 별도로 각 사람에 대해 유추할 수 있으며, 이는 상태 공간이 줄어들었기 때문에 여러 포즈의 공동 추론보다 훨씬 빠르다.

<br>
<br>

</details>

However, matching 2D poses across multiple views is challenging.  
A typical approach is to use the epipolar constraint to verify if two 2D poses are projections of the same 3D pose for each pair of views [23].  
But this approach may fail for the following reasons.  
First, the detected 2D poses are often inaccurate due to heavy occlusion and truncation, as shown in Figure 2(b), which makes geometric verification difficult.  
Second, matching each pair of views separately may produce inconsistent correspondences which violate the cycle consistency constraint, that is, two corresponding poses in two views may be matched to different people in another view.  
Such inconsistency leads to incorrect multi-view reconstructions.  
Finally, as shown in Figure 2, different sets of people appear in different views and the total number of people is unknown, which brings additional difficulties to the matching problem.

<details>

> 그러나 여러 뷰에서 2D 포즈를 일치시키는 것은 어렵다. 일반적인 접근 방식은 두 개의 2D 포즈가 각 뷰 쌍에 대해 동일한 3D 포즈의 투영인지 확인하기 위해 후두경 구속조건을 사용하는 것이다 [23]. 그러나 이 접근법은 다음과 같은 이유로 실패할 수 있다. 첫째, 그림 2(b)와 같이 심한 폐색 및 잘림으로 인해 감지된 2D 포즈가 부정확한 경우가 많아 기하학적 검증이 어렵다. 둘째, 각 보기 쌍을 개별적으로 일치시키면 주기 일관성 제약 조건을 위반하는 일관되지 않은 대응성이 발생할 수 있으며, 즉 두 보기에서 두 개의 해당 포즈를 다른 보기의 다른 사용자와 일치시킬 수 있다. 이러한 불일치는 다중 뷰 재구성을 부정확하게 만듭니다. 마지막으로, 그림 2와 같이, 다른 세트의 사람들이 서로 다른 보기에 나타나며 총 인원수를 알 수 없어 매칭 문제에 추가적인 어려움을 가져온다.

<br>
<br>

</details>

We propose a multi-way matching algorithm to address the aforementioned challenges.  
Our key ideas are:   
(i) combining the geometric consistency between 2D poses with the appearance similarity among their associated image patches to reduce matching ambiguities,   
and (ii) solving the matching problem for all views simultaneously with a cycle-consistency constraint to leverage multi-way information and produce globally consistent correspondences.  
The matching problem is formulated as a convex optimization problem and an efficient algorithm is developed to solve the induced optimization problem.

<details

> 앞에서 언급한 과제를 해결하기 위한 다방향 매칭 알고리듬을 제안한다. 우리의 핵심 아이디어는 (i) 2D 포즈 사이의 기하학적 일관성과 연관된 이미지 패치 간의 외관 유사성을 결합하여 일치되는 모호성을 줄이고, (ii) 다중 방향 정보를 활용하고 전역적으로 일관된 코어를 생성하기 위한 주기 일관성 제약으로 모든 보기에 대한 일치 문제를 동시에 해결한다.애석한 생각 일치 문제는 볼록 최적화 문제로 공식화되고 유도 최적화 문제를 해결하기 위한 효율적인 알고리듬이 개발된다.

<br>
<br>

</details>

In summary, the main contributions of this work are:

- We propose a novel approach for fast and robust multi-person 3D pose estimation. We demonstrate that, instead of jointly inferring multiple 3D poses using a 3DPS model in a huge state space, we can greatly reduce the state space and consequently improve both efficiency and robustness of 3D pose estimation by grouping the detected 2D poses that belong to the same person in all views.

- We propose a multi-way matching algorithm to find the cycle-consistent correspondences of detected 2D poses across multiple views. The proposed matching algorithm is able to prune false detections and deal with partial overlaps between views, without knowing the true number of people in the scene.

- We propose to combine geometric and appearance cues to match the detected 2D poses across views. We show that the appearance information, which is mostly ignored by previous methods, is important to link the 2D detections across views.

- The proposed approach outperforms the state-of-theart methods by a large margin without using any training data from the evaluated datasets. The code is available at https://zju3dv.github.io/ mvpose/.

<details>

> 빠르고 강력한 다인 3D 포즈 추정을 위한 새로운 접근법을 제안한다. 우리는 거대한 상태 공간에서 3DPS 모델을 사용하여 여러 3D 포즈를 공동으로 추론하는 대신, 모든 보기에서 동일한 사람에 속하는 감지된 2D 포즈를 그룹화하여 상태 공간을 크게 줄이고 결과적으로 3D 포즈 추정의 효율성과 견고성을 모두 향상시킬 수 있음을 보여준다.
- 우리는 여러 뷰에서 탐지된 2D 포즈의 주기 일치 대응성을 찾기 위한 다방향 매칭 알고리듬을 제안한다. 제안된 일치 알고리듬은 씬(scene)의 실제 사람 수를 알지 못한 채 잘못된 탐지를 제거하고 보기 간의 부분 중복을 처리할 수 있다.
- 여러 뷰에서 감지된 2D 포즈와 일치하도록 기하학적 신호와 외관 신호를 결합할 것을 제안한다. 우리는 이전 방법에서 대부분 무시되는 외관 정보가 보기 간에 2D 탐지를 연결하는 데 중요하다는 것을 보여준다.
- 제안된 접근 방식은 평가된 데이터 세트의 훈련 데이터를 사용하지 않고 최첨단 방법을 큰 폭으로 능가한다. 코드는 https://zju3dv.github.io/ mvpose/에서 이용할 수 있다.

<br>
<br>

</details>

## 2.Related work

**Multi-view 3D human pose:** Markerless motion capture has been investigated in computer vision for a decade.  
Early works on this problem aim to track the 3D skeleton or geometric model of human body through a multi-view sequence [38, 43, 11].  
These tracking-based methods require initialization in the first frame and are prone to local optima and tracking failures.  
Therefore, more recent works are generally based on a bottom-up scheme where the 3D pose is reconstructed from 2D features detected from images [36, 6, 32].  
Recent work [22] shows remarkable results by combing statistical body models with deep learning based 2D detectors.

<details>
**멀티뷰 3D 휴먼 포즈:** 마커리스 모션 캡처는 컴퓨터 비전에서 10년 동안 조사되어 왔습니다.  
이 문제에 대한 초기 연구는 다중 뷰 시퀀스를 통해 인체의 3D 골격 또는 기하학적 모델을 추적하는 것을 목표로 한다 [38, 43, 11].  
이러한 추적 기반 방법은 첫 번째 프레임에서 초기화가 필요하며 로컬 최적 및 추적 실패가 발생하기 쉽다.  
따라서, 보다 최근의 연구는 일반적으로 3D 포즈가 영상에서 검출된 2D 기능에서 재구성되는 상향식 체계를 기반으로 한다 [36, 6, 32].  
최근 연구[22]는 통계 신체 모델과 딥 러닝 기반 2D 검출기를 결합하여 주목할 만한 결과를 보여준다.

<br>
<br>

</details>



In this work, we focus on the multi-person 3D pose estimation.  
Most previous works are based on 3DPS models in which nodes represent 3D locations of body joints and edges encode pairwise relations between them [1, 20, 2, 21, 12].  
The state space for each joint is often a 3D grid representing a discretized 3D space.  
The likelihood of a joint being at some location is given by a joint detector applied to all 2D views and the pairwise potentials between joints are given by skeletal constraints [1, 2] or body parts detected in 2D views [21, 12].  
Then, the 3D poses of multiple people are jointly inferred by maximum a posteriori estimation.

<details>

</details>

As all body joints for all people are considered simultaneously, the entire state space is huge, resulting in heavy computation in inference.  
Another limitation of this approach is that it only uses multi-view geometry to link 2D evidences, which is sensitive to the setup of cameras.  
As a result, the performance of this approach degrades significantly when the number of views decreases [21].   Recent work [23] proposes to match 2D poses between views and then reconstructs 3D poses from the 2D poses belonging to the same person.  
But it only utilizes epipolar geometry to match 2D poses for each pair of views and ignores the cycle consistency constraint among multiple views, which may result in inconsistent correspondences.

<details>

</details>

**Single-view pose estimation:** There is a large body of literature on human pose estimation from single images.  
Single-person pose estimation [41, 34, 42, 30, 17] localizes 2D body keypoints of a person in a cropped image.  
There are two categories of multi-person pose estimation methods: top-down methods [10, 17, 15, 13] that first detect people in the image and then apply single-person pose estimation to the cropped image of each person, and bottom-up methods [25, 29, 8, 35, 18] that first detect all keypoints and then group them into different people.  
In general, the top-down methods are more accurate, while the bottomup methods are relatively faster.   
In this work, We adopt the Cascaded Pyramid Network [10], a state-of-the-art approach for multi-person pose detection, as an initial step in our pipeline.  

<details>

</details>

The advances in learning-based methods also make it possible to recover 3D human pose from a single RGB image, either lifting the detected 2D poses into 3D [28, 47, 9, 27] or directly regressing 3D poses [40, 37, 39, 45, 31] and even 3D body shapes from RGB [4, 24, 33].  
But the reconstruction accuracy of these methods is not comparable with the multi-view results due to the inherit reconstruction ambiguity when only a single view is available.

<details>

</details>

**Person re-ID and multi-image matching:** Person re-ID aims to identify the same person in different images [44], which is used as a component in our approach.  
Multi-image matching is to find feature correspondences among a collection of images [16, 46].  
We make use of the recent results on cycle consistency [16] to solve the correspondence problem in multi-view pose estimation.

<details>

</details>

## 3. Technical approach

Figure 2 presents an overview of our approach.  
First, an off-the-shelf 2D human pose detector is adopted to produce bounding boxes and 2D keypoint locations of people in each view (Section 3.1).  
Given the noisy 2D detections, a multiway matching algorithm is proposed to establish the correspondences of the detected bounding boxes across views and get rid of the false detections (Section 3.2).  
Finally, the 3DPS model is used to reconstruct the 3D pose for each person from the corresponding 2D bounding boxes and keypoints (Section 3.3).

<details>

</details>

### 3.1. 2D human pose detection

We adopt the recently-proposed Cascaded Pyramid Network [10] trained on the MSCOCO [26] dataset for 2D pose detection in images.  
The Cascaded Pyramid Network consists of two stages: the GlobalNet estimates human poses roughly whereas the RefineNet gives optimal human poses.  
Despite its state-of-the-art performance on benchmarks, the detections may be quite noisy as shown in Figure 2(b).

<details>

우리는 영상에서 2D 포즈 탐지를 위해 MSCOCO [26] 데이터 세트에서 훈련된 최근 제안된 Cascaded Pyramid Network [10]을 채택한다.  
계단식 피라미드 네트워크는 두 단계로 구성된다: GlobalNet은 대략적으로 인간의 자세를 추정하는 반면, RefleaseNet은 최적의 인간 포즈를 제공한다.  
벤치마크에서 최첨단 성능에도 불구하고, 그림 2(b)와 같이 탐지가 상당히 시끄러울 수 있다.

<br>
<br>

</details>

<br>

### 3.2. Multiview correspondences

Before reconstructing the 3D poses, the detected 2D poses should be matched across views, i.e., we need to find in all views the 2D bounding boxes belonging to the same person.  
However, this is a challenging task as we discussed in the introduction.

<details>
3D 포즈를 재구성하기 전에 검출된 2D 포즈는 views 간에 일치되어야 한다.  
즉, 동일한 사람에 속하는 2D 경계 상자를 모든 보기에서 찾아야 한다.  
그러나 이것은 서론에서 논의한 바와 같이 어려운 과제이다.

<br>
<br>
</details>

<br>

To solve this problem, we need  
1) a proper metric to measure the likelihood that two 2D bounding boxes belong to the same person (a.k.a. affinity), and  
2) a matching algorithm to establish the correspondences of bounding boxes across multiple views.  
In particular, the matching algorithm should not place any assumption about the true number of people in the scene.  
Moreover, the output of the matching algorithm should be cycle-consistent, i.e. any two corresponding bounding boxes in two images should correspond to the same bounding box in another image.

<details>
이 문제를 해결하기 위해서는  
1) 두 개의 2D 경계 상자가 동일한 사람에 속할 가능성을 측정하기 위한 적절한 메트릭(즉, 친화력)  
2) 여러 뷰에서 경계 상자의 대응성을 설정하기 위한 일치 알고리즘.  
특히, 매칭 알고리즘은 장면에서 실제 사람 수에 대한 가정을 하지 않아야 한다.  
또한 일치 알고리듬의 출력은 사이클 일관성이 있어야 한다.  
즉, 두 이미지의 해당 경계 상자 2개는 다른 이미지의 동일한 경계 상자에 대응해야 한다.
<br>
<br>
</details>

<br>

**Problem statement:**

생략

$$0 <= P_{ij}1 <= 1, 0 <= P^T_{ij}1 <= 1. \; \; \; (1)$$

**Affinity matrix:** We propose to combine the appearance similarity and the geometric compatibility to calculate the affinity scores between bounding boxes.

<details>

<br>
<br>
</details>

<br>

First, we adopt a pre-trained person re-identification (re-ID) network to obtain a descriptor for a bounding box.  
The re-ID network trained on massive re-ID datasets is expected to be able to extract discriminative appearance features that are relatively invariant to illumination and viewpoint changes.   
Specifically, we feed the cropped image of each bounding box through the publicly available re-ID model proposed in [44] and extract the feature vector from the “pool5” layer as the descriptor for each bounding box.  
Then, we compute the Euclidean distance between the descriptors of a bounding box pair and map the distances to values in (0, 1) using the sigmoid function as the appearance affinity score of this bounding box pair.

<details>
먼저, 우리는 경계 상자에 대한 설명자를 얻기 위해 사전 훈련된 사람 재식별(re-ID) 네트워크를 채택한다.  
대규모 재ID 데이터 세트에 대해 훈련된 재ID 네트워크는 조명 및 관점 변화에 상대적으로 불변하는 차별적 외관 특징을 추출할 수 있을 것으로 기대된다.   
구체적으로, 우리는 [44]에서 제안된 공개적으로 사용 가능한 재ID 모델을 통해 각 경계 상자의 자른 이미지를 제공하고 "풀5" 계층에서 각 경계 상자의 설명자로 기능 벡터를 추출한다.  
그런 다음, 우리는 경계 상자 쌍의 설명자 사이의 유클리드 거리를 계산하고 시그모이드 함수를 이 경계 상자 쌍의 외관 선호도 점수로 사용하여 거리를 (0, 1)의 값으로 매핑한다.
<br>
<br>
</details>

<br>

Besides appearances, another important cue to associate two bounding boxes is that their associated 2D poses should
be geometrically consistent.  
Specifically, the corresponding 2D joint locations should satisfy the epipolar constraint, i.e. a joint in the first view should lie on the epipolar line associated with its correspondence in the second view

<details>

<br>
<br>
</details>

<br>

생략

**Multi-way matching with cycle consistency:** If there are only two views to match, one can simply maximize <$$P_{ij} ,A_{ij}$$ > and find the optimal matching by the Hungarian algorithm.  
But when there are multiple views, solving the matching problem separately for each pair of views ignores the cycle-consistency constraint and may lead to inconsistent results.  
Figure 3 shows an example, where the correspondences in red are inconsistent and the ones in green are cycle-consistent as they form a closed cycle.

<details>
Multi-way matching with cycle consistency: 일치시킬 보기가 두 개뿐이라면 <$$P_{ij},A_{ij}$$>를 최대화하고 헝가리 알고리즘으로 최적의 일치를 찾을 수 있다.  
그러나 보기가 여러 개 있는 경우 각 보기 쌍에 대해 일치 문제를 별도로 해결하는 것은 주기 일관성 제약 조건을 무시하고 일관되지 않은 결과를 초래할 수 있다.  
그림 3은 빨간색의 대응성이 일관성이 없고 녹색의 대응성이 닫힌 사이클을 형성할 때 일관성이 없는 예를 보여준다.
<br>
<br>
</details>

<br>

생략


### 3.3. 3D pose reconstruction

Given the estimated 2D poses of the same person in different views, we reconstruct the 3D pose.  
This can be simply done by triangulation, but the gross errors in 2D pose estimation may largely degrade the reconstruction.  
In order to fully integrate uncertainties in 2D pose estimation and incorporate the structural prior on human skeletons, we make use of the 3DPS model and propose an approximate algorithm for efficient inference.

<details>
서로 다른 보기에서 동일인의 추정 2D 포즈를 고려하여 3D 포즈를 재구성한다.  
이는 단순히 삼각 측량에 의해 수행될 수 있지만, 2D 포즈 추정의 총 오차는 재구성을 크게 저하시킬 수 있다.  
2D 포즈 추정의 불확실성을 완전히 통합하고 인간 골격에 앞서 구조적인 요소를 통합하기 위해, 우리는 3DPS 모델을 사용하고 효율적인 추론을 위한 근사 알고리듬을 제안한다.
<br>
<br>
</details>

<br>

**3D pictorial structure:**  

생략

**Inference:** The typical strategy to maximize $$p(T \vert I)$$ is first discretizing the state space as a uniform 3D grid, and applying the max-product algorithm [6, 32]. However, the complexity of the max-product algorithm grows fast with the dimension of the state space.

<details>
서로 다른 보기에서 동일인의 추정 2D 포즈를 고려하여 3D 포즈를 재구성한다.
이는 단순히 삼각 측량에 의해 수행될 수 있지만, 2D 포즈 추정의 총 오차는 재구성을 크게 저하시킬 수 있다.
2D 포즈 추정의 불확실성을 완전히 통합하고 인간 골격에 앞서 구조적인 요소를 통합하기 위해, 우리는 3DPS 모델을 사용하고 효율적인 추론을 위한 근사 알고리듬을 제안한다.
<br>
<br>
</details>

<br>

Instead of using grid sampling, we set the state space for each 3D joint to be the 3D proposals triangulated from all pairs of corresponding 2D joints.  
As long as a joint is correctly detected in two views, its true 3D location is included in the proposals.  
In this way, the state space is largely reduced, resulting in much faster inference without sacrificing the accuracy.

<details>
그리드 샘플링 대신, 우리는 각 3D 조인트의 상태 공간을 해당 2D 조인트의 모든 쌍에서 삼각 측량된 3D 제안으로 설정한다.  
두 뷰에서 조인트가 올바르게 감지되는 한, 조인트의 실제 3D 위치가 제안서에 포함됩니다.  
이렇게 하면 상태 공간이 크게 줄어들어 정확성을 희생하지 않고 훨씬 빠른 추론을 얻을 수 있다.
<br>
<br>
</details>

<br>

## 4.Empirical evaluation

We evaluate the proposed approach on three public datasets including both indoor and outdoor scenes and compare it with previous works as well as several variants of the proposed approach.

<details>

<br>
<br>
</details>

<br>

### 4.1. Datasets

The following three datasets are used for evaluation: Campus [1]: It is a dataset consisting of three people interacting with each other in an outdoor environment, captured with three calibrated cameras.  
We follow the same evaluation protocol as in previous works [1, 3, 2, 12] and use the percentage of correctly estimated parts (PCP) to measure the accuracy of 3D location of the body parts.

<details>

<br>
<br>
</details>

<br>

**Shelf [1]:** Compared with Campus, this dataset is more complex, which consists of four people disassembling a shelf at a close range.  
There are five calibrated cameras around them, but each view suffers from heavy occlusion.  
The evaluation protocol follows the prior work and the evaluation metric is also 3D PCP.  

<details>

<br>
<br>
</details>

<br>

CMU Panoptic [20]: This dataset is captured in a studio with hundreds of cameras, which contains multiple people engaging in social activities.  
For the lack of ground truth, we qualitatively evaluate our approach on the CMU Panoptic dataset.

<details>

<br>
<br>
</details>

<br>

### 4.2. Ablation analysis

We first give an ablation analysis to justify the algorithm design in the proposed approach.  
The Campus and Shelf datasets are used for evaluation.

<details>

<br>
<br>
</details>

<br>

**Appearance or geometry?**  
As described in section 3.2, our approach combines appearance and geometry information to construct the affinity matrix.  
Here, we compare it with the alternatives using appearance or geometry alone.  
The detailed results are presented in Table 1.

<details>

<br>
<br>
</details>

<br>

On the Campus, using appearance only achieves competitive results, since the appearance difference between actors is large.  
The result of using geometry only is worse because the cameras are far from the people, which degrades the discrimination ability of the epipolar constraint.  
On the Shelf, the performance of using appearance alone drops a lot.  
Especially, the result of actor 2 is erroneous, since his appearance is similar to another person.  
In this case, the combination of appearance and geometry greatly improve the performance.

<details>

<br>
<br>
</details>

<br>

**Direct triangulation or 3DPS?**  
Given the matched 2D poses in all views, we use a 3DPS model to infer the final 3D poses, which is able to integrate the structural prior on human skeletons.  
A simple alternative is to reconstruct 3D pose by triangulation, i.e., finding the 3D pose that has the minimum reprojection errors in all views.  
The result of this baseline method (‘NO 3DPS’) is presented in Table 1.

<details>

<br>
<br>
</details>

<br>

The result shows that when the number of cameras in the scene is relatively small, for example, in the Campus dataset (three cameras), using 3DPS can greatly improve the performance.  
When a person is often occluded in many views, for example, actor 2 in the Shelf dataset, the 3DPS model can also be helpful.

<details>

<br>
<br>
</details>

<br>


**Matching or no matching?**  
Our approach first matches 2D poses across views and then applies the 3DPS model to each cluster of matched 2D poses.  
An alternative approach in most previous works [2, 21] is to directly apply the 3DPS model to infer multiple 3D poses from all detected 2D poses without matching.

<details>

<br>
<br>
</details>

<br>

Here, we give a comparison between them. As Belagiannis et al. [2] did not use the most recent CNN-based keypoint detectors and Joo et al. [21] did not report results on public benchmarks, we re-implement their approach with the state-of-the-art 2D pose detector [8] for a fair comparison.  
The implementation details are given in the supplementary materials.  
Table 1 shows that the 3DPS without matching obtained decent results on the Self dataset but performed much worse on the Campus dataset, where there are only three cameras.  
The main reason is that the 3DPS model implicitly uses multi-view geometry to link the 2D detections across views but ignores the appearance cues.  
When using a sparse set of camera views, the multiview geometric consistency alone is sometimes insufficient to differentiate the correct and false correspondences, which leads to false 3D pose estimation.  
This observation coincides with the other results in Table 1 as well as the observation in [21].  
The proposed approach explicitly leverage the appearance cues to find cross-view correspondences, leading to more robust results.  
Moreover, the matching step significantly reduces the size of state space and makes the 3DPS model inference much faster.

<details>

<br>
<br>
</details>

<br>

### 4.3. Comparison with stateoftheart

We compare with the following baseline methods. Belagiannis et al. [1, 3] were among the first to introduce 3DPS model-based multi-person pose estimation and their method was extended to the video case to leverage temporal consistency [2].  
Ershadi-Nasab et al. [12] is a very recent method that proposed to cluster the 3D candidate joints to reduce the state space.  

<details>

<br>
<br>
</details>

<br>

The results on the Campus and Shelf datasets are presented in Table 2. Note that the 2D pose detector [10] and the reID network [44] used in our approach are the released pre-trianed models without any fine-tuning on the evaluated datasets.  
Even with the generic models, our approach outperforms the state-of-the-art methods by a large margin.  
In particular, our approach significantly improves the performance on Actor 3 in the Campus dataset and Actor 2 in the Shelf dataset, which suffer from severe occlusion.  
We also include our results without the 3DPS model but using triangulation to reconstruct 3D poses from matched 2D poses.  
Due to the robust and consistent matching, direct triangulation also obtains better performance than previous methods.

<details>

<br>
<br>
</details>

### 4.4. Qualitative evaluation

Figure 4 shows some representative results of the proposed approach on the Shelf and CMU Panoptic dataset.  
Taking inaccurate 2D detections as input, our approach is able to establish their correspondences across views, identify the number of people in the scene automatically, and finally reconstruct their 3D poses.  
The final 2D pose estimates obtained by projecting the 3D poses back to 2D views are also much more accurate than the original detections.

<details>

<br>
<br>
</details>

### 4.5. Running time

We report running time of our algorithm on the sequences with four people and five views in the Shelf dataset, tested on a desktop with an Intel i7 3.60 GHz CPU and a GeForce 1080Ti GPU.  
Our unoptimized implementation on average takes 25 ms for running reID and constructing affinity matrices, 20 ms for the multi-way matching algorithm, and 60 ms for 3D pose inference.  
Moreover, the results in Table 2 show that our approach without the 3DPS model also obtains very competitive performance, which is able to achieve real-time performance at > 20fps.

<details>

<br>
<br>
</details>

## 5. Summary

In this paper, we propose a novel approach to multi-view 3D pose estimation that can fastly and robustly recover 3D poses of a crowd of people with a few cameras. Compared with the previous 3DPS based methods, our key idea is to use a multi-way matching algorithm to cluster the detected 2D poses to reduce the state space of the 3DPS model and thus improves both efficiency and robustness.  
We also demonstrate that the 3D poses can be reliably reconstructed from clustered 2D poses by triangulation even without using the 3DPS model.  
This shows the effectiveness of the proposed multi-way matching algorithm, which leverages the combination of geometric and appearance cues as well as the cycle-consistency constraint for matching 2D poses across multiple views.

<details>

<br>
<br>
</details>

**Acknowledgements:**
