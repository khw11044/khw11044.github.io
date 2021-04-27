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


First, we introduce a cross-view fusion scheme into CNN to jointly estimate 2D poses for multiple views.  
Consequently, the 2D pose estimation for each view already benefits from other views.  
Second, we present a recursive Pictorial Structure Model to recover the 3D pose from the multi-view 2D poses.  
It gradually improves the accuracy of 3D pose with affordable computational cost.  
We test our method on two public datasets H36M and Total Capture.  
The Mean Per Joint Position Errors on the two datasets are 26mm and 29mm, which outperforms the state-of-the-arts remarkably (26mm vs 52mm, 29mm vs 35mm).

> 첫째, 우리는 multiple views에 대한 2D poses를 jointly로 estimate하기 위해 CNN에 cross-view fusion scheme를 도입한다.  
결과적으로 각 view에 대한 2D pose estimation은 이미 다른 뷰에서 이익을 얻는다.  
둘째, multi-view 2D poses에서 3D pose를 recover하기 위한 recursive Pictorial Structure Model을 제시한다.  
저렴한 계산 비용으로 3D 포즈의 정확도를 단계적으로 향상시킨다.  
우리는 두 개의 공개 데이터 세트 H36M과 Total Capture에서 우리의 방법을 테스트한다.  
두 데이터 세트의 Mean Per Joint Position Errors는 26mm와 29mm로 state-of-the-arts 상태(26mm vs 52mm, 29mm vs 35mm)를 크게 능가한다.


## 1. Introduction
The task of 3D pose estimation has made significant progress due to the introduction of deep neural networks.  
Most efforts [16, 13, 33, 17, 23, 19, 29, 28, 6] have been devoted to estimating relative 3D poses from monocular images.  
The estimated poses are centered around the pelvis joint thus do not know their absolute locations in the environment (world coordinate system).  

>3D pose estimation 작업은 deep neural networks의 도입으로 인해 상당한 진전을 이루었다.  
대부분의 노력[16, 13, 33, 17, 23, 19, 29, 28, 6]은 단안 영상에서 상대적인 3D poses를 추정하는 데 사용되었다.  
추정된 포즈는 골반 관절의 중심에 있으므로 환경(world coordinate system)에서 절대적인 위치를 알 수 없다.

In this paper, we tackle the problem of estimating absolute 3D poses in the world coordinate system from multiple cameras [1, 15, 4, 18, 3, 20].  
Most works follow the pipeline of first estimating 2D poses and then recovering 3D pose from them.  
However, the latter step usually depends on the performance of the first step which unfortunately often has large errors in practice especially when occlusion or motion blur occurs in images.  
This poses a big challenge for the final 3D estimation.  

> 본 논문에서는 여러 카메라[1, 15, 4, 18, 3, 20]를 사용한 world coordinate system에서  absolute 3D poses를 추정하는 문제를 다룬다.  
대부분의 작업은 먼저 2D 포즈를 추정한 다음 3D 포즈를 복구하는 파이프라인을 따른다.  
그러나 후자는 일반적으로 첫 번째 단계의 성능에 따라 달라지는데, 안타깝게도 첫 번째 단계의 성능은 특히 영상에 occlusion 또는 motion blur 현상이 발생할 때 실제로 큰 오류를 범하는 경우가 많다.  
이는 최종 3D estimation에 큰 문제를 제기한다.

On the other hand, using the Pictorial Structure Model (PSM) [14, 18, 3] for 3D pose estimation can alleviate the influence of inaccurate 2D joints by considering their spatial dependence.  
It discretizes the space around the root joint by an N ×N ×N grid and assigns each joint to one of the $$N^3$$ bins (hypotheses).  
It jointly minimizes the projection error between the estimated 3D pose and the 2D pose, along with the discrepancy of the spatial configuration of joints and its prior structures.  
However, the space discretization causes large quantization errors.  
For example, when the space surrounding the human is of size 2000mm and N is 32, the quantization error is as large as 30mm. We could reduce the error by increasing N, but the inference cost also increases at $$O(N^3)$$, which is usually intractable.  

> 반면, 3D pose estimation을 위해 Pictorial Structure Model (PSM)[14, 18, 3]을 사용하면 공간 의존성을 고려하여 부정확한 2D joints의 영향을 완화할 수 있다.  
N × N × N grid에 의해 뿌리 관절 주변의 공간을 이산화하고 각 관절을 $$N^3$$ bins 중 하나에 할당한다.(hypotheses)  
관절의 공간 구성과 이전 구조의 불일치와 함께, estimated 3D pose와 2D pose 사이의 projection error를 jointly로 최소화한다.  
그러나 공간 이산화는 큰 양자화 오류를 일으킨다.  
예를 들어 인간을 둘러싼 공간이 2000mm이고 N이 32인 경우 정량화 오류는 30mm만큼 크다.  
우리는 N을 증가시켜 오차를 줄일 수 있지만, 추론 비용 또한 대개 다루기 힘든 O($$N^3$$)에서 증가한다.

Second, we present Recursive Pictorial Structure Model (RPSM), to recover the 3D pose from the estimated multi-view 2D pose heatmaps.  
Different from PSM which directly discretizes the space into a large number of bins in order to control the quantization error, RPSM recursively discretizes the space around each joint location (estimated in the previous iteration) into a finer-grained grid using a small number of bins.  
As a result, the estimated 3D pose is refined step by step.  
Since N in each step is usually small, the inference speed is very fast for a single iteration.  
In our experiments, RPSM decreases the error by at least 50% compared to PSM with little increase of inference time.  

> 둘째, estimated multi-view 2D pose heatmaps에서 3D pose를 recover하기 위해 Recursive Pictorial Structure Model (RPSM)을 제시합니다.  
quantization error를 제어하기 위해 공간을 직접 많은 수의 bins으로 이산시키는 PSM과는 달리, RPSM은 적은 수의 bins을 사용하여 각 관절 위치(estimated in the previous iteration) 주변의 공간을 finer-grained grid로 재귀적 이산화한다.

For 2D pose estimation on the H36M dataset [11], the average detection rate over all joints improves from 89% to 96%.  
The improvement is significant for the most challenging “wrist” joint.  
For 3D pose estimation, changing PSM to RPSM dramatically reduces the average error from 77mm to 26mm.  
Even compared with the state-of-the-art method with an average error 52mm, our approach also cuts the error in half.  
We further evaluate our approach on the Total Capture dataset [27] to validate its generalization ability.  
It still outperforms the state-of-the-art [26].

> H36M 데이터 세트에 대한 2D 포즈 추정의 경우 [11], 모든 관절에 대한 평균 감지 속도는 89%에서 96%로 향상된다.  
개선은 가장 도전적인 "손목" 관절에 중요하다.  
3D 포즈 추정의 경우 PSM을 RPSM으로 변경하면 평균 오차가 77mm에서 26mm로 크게 줄어든다.  
평균 오차 52mm의 최첨단 방법과 비교해도 우리의 접근 방식은 오류를 절반으로 줄인다.  
우리는 총 캡처 데이터 세트[27]에 대한 접근 방식을 추가로 평가하여 일반화 능력을 검증한다.  
그것은 여전히 최첨단[26]을 능가한다.

## 2. Related Work
We first review the related work on multi-view 3D pose estimation and discuss how they differ from our work.  
Then we discuss some techniques on feature fusion.

> 먼저 multi-view 3D pose estimation에 대한 관련 작업을 검토하고 작업과 어떻게 다른지 논의한다.  
그런 다음 feature fusion에 대한 몇 가지 기술에 대해 논의한다.

**Multi-view 3D Pose Estimation**   
Many approaches [15, 10, 4, 18, 3, 19, 20] are proposed for multi-view pose estimation.  
They first define a body model represented as simple primitives, and then optimize the model parameters to align the projections of the body model with the image features.  
These approaches differ in terms of the used image features and optimization algorithms.

> multi-view pose estimation을 위해 많은 접근 방식[15, 10, 4, 18, 3, 19, 20]이 제안된다.  
이들은 먼저 simple primitives로 표현되는 body model을 정의한 다음, body model의 projections을 image features과 정렬하도록  model parameters를 최적화한다.  
이러한 접근 방식은 사용된 image features과 최적화 알고리즘 측면에서 다르다.

We focus on the Pictorial Structure Model (PSM) which is widely used in object detection [8, 9] to model the spatial dependence between the object parts.  
This technique is also used for 2D [32, 5, 1] and 3D [4, 18] pose estimation where the parts are the body joints or limbs.  
In [1], Amin et al. first estimate the 2D poses in a multi-view setup with PSM and then obtain the 3D poses by direct triangulation.  
Later Burenius et al. [4] and Pavlakos et al. [18] extend PSM to multi-view 3D human pose estimation.  
For example, in [18], they first estimate 2D poses independently for each view and then recover the 3D pose using PSM.  
Our work differs from [18] in that we extend PSM to a recursive version, i.e. RPSM, which efficiently refines the 3D pose estimations step by step.  
In addition, they [18] do not perform cross-view feature fusion as we do.

> 우리는 object detection [8, 9]에서 object parts 사이의 공간 의존성을 모델링하기 위해 널리 사용되는 Pictorial Structure Model (PSM)에 초점을 맞춘다.  
이 기술은 또한 신체 관절이나 팔다리가 있는 부분이 2D [32, 5, 1] 및 3D [4, 18] pose estimation에도 사용된다.  
[1]에서 Amin 등은 PSM이 있는 multi-view 설정에서 먼저 2D poses를 추정한 다음 직접 삼각측정을 통해 3D poses를 얻는다.  
나중에 Burenius 외 [4]와 Pavlakos 외 [18]는 PSM을 multi-view 3D human pose estimation으로 확장한다.  
예를 들어, [18]에서는 먼저 각 view에 대해 독립적으로 2D poses를 추정한 다음 PSM을 사용하여 3D pose를 recover한다.  
우리의 연구는 PSM을 recursive version, 즉 RPSM으로 확장한다는 점에서 [18]과 다르다. RPSM은3D pose estimations을 단계별로 효율적으로 세분화한다.  
또한 [18]은(는) cross-view feature fusion을 수행하지 않는다.

**Multi-image Feature Fusion**   
Fusing features from different sources is a common practice in the computer vision literature.  
For example, in [34], Zhu et al. propose to warp the features of the neighboring frames (in a video sequence) to the current frame according to optical flow in order to robustly detect the objects.  
Ding et al. [7] propose to aggregate the multi-scale features which achieves better segmentation accuracy for both large and small objects.  

> 다른 sources의 features들을 융합하는 것은 computer vision literature에서 흔한 관행이다.   
예를 들어, [34]에서 Zhu 등은 objects를 robustly하게 detect하기 위해 optical flow에 따라 인접 프레임의 features(비디오 시퀀스로)을 현재 프레임으로 왜곡할 것을 제안한다.  
Ding 등. [7]은 큰 객체와 작은 객체 모두에 대해 더 나은 segmentation accuracy를 달성하는 multi-scale features을 집계할 것을 제안한다.

Amin et al. [1] propose to estimate 2D poses by exploring the geometric relation between multi-view images.  
It differs from our work in that it does not fuse features from other views to obtain better 2D heatmaps.  
Instead, they use the multi-view 3D geometric relation to select the joint locations from the “imperfect” heatmaps.  
In [12], multi-view consistency is used as a source of supervision to train the pose estimation network.  
To the best of our knowledge, there is no previous work which fuses multi-view features so as to obtain better 2D pose heatmaps because it is a challenging task to find the corresponding features across different views which is one of our key contributions of this work.

> Amin 외. [1]은 multi-view images 간의 geometric 관계를 탐색하여 2D poses를 추정하는 것을 제안한다.  
더 나은 2D heatmaps을 얻기 위해 다른 views와 features을 융합하지 않는다는 점에서 우리의 작업과 다르다.  
대신 multi-view 3D geometric 관계를 사용하여 "불완전한" heatmaps에서 joint 위치를 선택한다.  
[12]에서 multi-view consistency은 pose estimation network를 훈련시키기 위한 supervision의 source로 사용된다.  
우리가 아는 한, 이 작업의 key contributions 중 하나인 다른 views에서 해당 features을 찾는 것은 어려운 작업이기 때문에 더 나은 2D pose heatmaps을 얻기 위해 multi-view features을 융합한 previous work는 없다.

## 3. Cross View Fusion for 2D Pose Estimation

![Fig1](/assets/img/Blog/papers/Pose/CrossViewFusion/Fig1.JPG)

Our 2D pose estimator takes multi-view images as input, generates initial pose heatmaps respectively for each, and then fuses the heatmaps across different views such that the heatmap of each view benefits from others.  
The process is accomplished in a single CNN and can be trained end-to-end.  
Figure 1 shows the pipeline for two-view fusion.  
Extending it to multi-views is trivial where the heatmap of each view is fused with the heatmaps of all other views.  
The core of our fusion approach is to find the corresponding features between a pair of views.  

> 우리의 2D pose estimator는 multi-view images을 입력으로 가져가고 각각에 대해 초기 pose heatmaps을 생성한 다음 각 view의 heatmap이 다른 views로부터 benefits을 얻도록 여러 views에 걸쳐 heatmaps을 융합한다.  
이 프로세스는 단일 CNN에서 수행되며 end-to-end 훈련을 받을 수 있다.  
Figure 1은 two-view fusion을 위한 pipeline이다.  
각 view의 heatmap가 다른 모든 views의 heatmaps와 융합되는 경우 multi-views로 확장하는 것은 trivial이다.
우리의 fusion 접근법의 핵심은 한 쌍의 views 사이에서 상응하는 features을 찾는 것이다.

![Fig2](/assets/img/Blog/papers/Pose/CrossViewFusion/Fig2.JPG)

Suppose there is a point $$P$$ in 3D space.  
See Figure 2.  
Its projections in view $$u$$ and $$v$$ are $$Y_P^u \in \mathcal{Z}^u$$ and $$Y^v_P \in Z^v$$,
respectively where $$\mathcal{Z}^u$$ and $$\mathcal{Z}^v$$ denote all pixel locations in the two views, respectively.  
The heatmaps of view u and v are $$\mathcal{F}^u = {x_1^u, ... , x^u_{\vert \mathcal{Z}^u \vert}$$ and $$\mathcal{F}^v = {x_1^v, ... , x^v_{\vert \mathcal{Z}^v \vert}$$.
