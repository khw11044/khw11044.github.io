---
layout: post
bigtitle:  "The Progress of Human Pose Estimation:"
subtitle:   " A Survey and Taxonomy of Models Applied in 2D Human Pose Estimation"
categories:
    - blog
    - papers
    - pose-estimation
tags:
    - pose
comments: true
published: true
---



# The Progress of Human Pose Estimation : Survey-2020

IEEE 2020 [paper](https://ieeexplore.ieee.org/abstract/document/9144178)

* toc
{:toc}

## ABSTRACT

Human pose estimation localizes body keypoints to accurately recognizing the postures of individuals given an image.  
This step is a crucial prerequisite to multiple tasks of computer vision which include human action recognition, human tracking, human-computer interaction, gaming, sign languages, and video surveillance.  
Therefore, we present this survey article to fill the knowledge gap and shed light on the researches of 2D human pose estimation.  
A brief introduction is followed by classifying it as a single or multi-person pose estimation based on the number of people needed to be tracked.  
Then gradually the approaches used in human pose estimation are described before listing some applications and also flaws facing in pose estimation.  
Following that, a center of attention is given on briefly discussing researches with a significant effect on human pose estimation and examine the novelty, motivation, architecture, the procedures (working principles) of each model together with its practical application and drawbacks, datasets implemented, as well as the evaluation metrics used to evaluate the model.  
This review is presented as a baseline for newcomers and guides researchers to discover new models by observing the procedure and architecture flaws of existing researches.

> Human pose estimation은 이미지가 주어지고 개인의 자세를 정확하게 인식하기 위해 body keypoints를 위치시킨다.  
이 단계는 human action recognition, human tracking, human-computer interaction, gaming, sign languages, 그리고 video surveillance를 포함하는 컴퓨터 비전의 여러 작업에 대한 중요한 전제 조건이다.  
따라서, 우리는 지식 격차를 메우기 위해 이 survey article를 제시하고 2D human pose estimation의 연구를 조명한다.  
간단한 소개에 이어 추적해야 할 사람수를 기준으로 single-person pose estimation 또는 multi-person pose estimation으로 분류한다.  
이 후 일부 applications과 pose estimatio이 직면한 결함을 설명하기 전에 human pose estimation에 사용되는 접근법을 설명한다.
그 후, human pose estimation에 상당한 영향을 미치는 연구를 간략하게 논의하고 각 모델의 novelty, motivation, architecture procedures(작업 원리)와 application 및 단점, 구현된 datasets, 그리고 model을 평가하는데 사용된 evaluation metrics를 함께 검토하는 데 집중한다.   
이 review는 newcomers의 baseline으로 제시하며, 기존 연구의 절차와 architecture 결함을 관찰하여 연구자가 새로운 모델을 발견하도록 안내한다.

## INDEX TERMS
Human pose estimation, pose estimation and action recognition, pose estimation survey, single and multi-person pose estimation.

## I. INTRODUCTION

![Fig1](/assets/img/Blog/papers/Pose/survey_2020/Fig1.JPG)

Human pose estimation is one of the challenging fields of study in computer vision which aims in determining the position or spatial location of body keypoints (parts/joints) of a person from a given image or video [1], [2], as shown in Fig.1.  
Thus, pose estimation obtains the pose of an articulated human body, which consists of joints and rigid parts using image-based observations [3].  
Human pose estimation refers to the process of inferring poses in an image and these estimations are performed in either 3D or 2D [4].  
To solve this problem, several approaches in the literature have been proposed.  
Early works introduced the classical approaches to articulated human pose estimation called the pictorial structures [5]-[8].  
In these models, the spatial correlations of the body parts are demonstrated as a tree-structured graphical model and they are very successful when the limbs are visible however faced problems when the tree-structured fails capturing the correlation between variables.  
Hand-crafted features such as edges, contours, the histogram of oriented gradients (HOG) features, and color histograms have also been used in early works for human pose estimation [9]-[13].  
These models have shown bad generalization performance which faced problems in detecting the accurate location of the body parts.

> Human pose estimation은 Fig.1에 표시된 것처럼 주어진 image 또는 video에서 사람의 body keypoints (parts/joints)의 position 또는 spatial location를 결정하는 것을 목표로 하는 computer vision 연구의 어려운 분야 중 하나이다. [1], [2]  
따라서 pose estimation은 image-based 관찰을 사용하여 joints과 rigid parts로 구성된 articulated human body의 pose를 얻는다 [3].  
Human pose estimation은 image에서 poses를 유추(infer)하는 과정을 의미하며 이러한 estimations은 3D 또는 2D에서 수행된다 [4].  
이 문제를 해결하기 위해, 문헌에서 몇 가지 접근법이 제안되었다.  
초기 works들은 pictorial structures라고 불리는 articulated human pose estimation에 대한 고전적인 접근 방식을 도입했다 [5]-[8].  
이러한 모델에서 body parts의 spatial 상관 관계는 tree-structured graphical model로 입증되며, 팔다리(limbs)가 보일 때 매우 성공적이지만 tree-structured가 variables 간의 상관 관계를 capturing하지 못할 때 문제에 직면한다.  
edges, contours(윤곽선), the histogram of oriented gradients (HOG) features 와 color histograms과 같은 Hand-crafted features도 초기 works에는 human pose estimation을 위해 사용되었다 [9]-[13].  
이러한 모델은 body parts의 정확한 location를 감지하는 데 문제가 발생한 좋지 않은 generalization performance를 보여주었다.

**Contributions:**  
Solving the problems and challenges related to human pose estimation has been advanced and progressed remarkably with the help of deep learning and publicly available datasets.  
This survey provides a summary of these works comprehending up to date information and points the future research directions.  
Like some remarkable surveys [14]-[18], this paper also provides a general concept of human pose estimation.  
It can be used as a guideline for people who are new to this concept and helps them to define noble models by combining the network structures of the existing models.  
Additionally, it helps researchers to compare their work with significant models based on deep learning.  
Besides, here are some specific main contributions of this review:

> **Contributions:**   
human pose estimation과 관련된 challenges 및 문제 해결은 deep learning과 publicly available datasets의 도움을 받아 매우 진전되었다.  
본 survey는 이러한 연구의 요약본으로 최신 정보를 파악하고 향후 연구 방향을 제시한다.  
일부 주목할 만한 surveys[14]-[18]와 마찬가지로, 본 논문에서는 human pose estimation의 일반적인 개념도 제공한다.  
이 개념을 처음 접하는 사람들을 위한 guideline으로 활용할 수 있으며, 기존 모델의 network 구조를 결합해 noble models을 정의할 수 있도록 돕는다.  
또한, 그것은 연구자들이 deep learning을 기반으로 한 중요한 models과 그들의 연구를 비교할 수 있도록 돕는다.  
또한, 다음은 이 review의 몇 가지 주요 contributions이다 :

- Provides a summary of preferred backbone architectures and loss functions used in addition to the overview of evaluation metrics implied and datasets employed.  
- Provides an overview of recent models on 2D human pose estimation
- Limitations of each model's work and open issues are presented

> - 암시된 evaluation metrics와 채택된 datasets의 개요와 함께 사용되는 기본 backbone architectures 와 loss functions의 summary를 제공.
- 2D human pose estimation에 대한 최신 모델의 개요를 제공합니다.
- 모델별 work 및 open issues의 한계 제시

### A. RELATED WORKS

Other survey papers related to pose estimation have been released in the past years.  
For example, two survey papers [14], [15] published in 2016 have extensively surveyed models on human pose estimation which did not implement deep learning-based approaches.  
Then [16] presented a survey of deep learning, pose estimation, and application of deep learning for computer vision.  
A review on hand pose estimation is presented by [19] whereas [20] provided a survey on head pose estimation.

> pose estimation과 관련된 다른 survey papers은 지난 몇 년 동안 발표되었다.  
예를 들어, 2016년에 발표된 두 개의 survey papers [14], [15]은 deep learning-based approaches을 구현하지 않은 human pose estimation에 대한 models을 광범위하게 조사했다.  
그리고 [16]은 computer vision에 대한 deep learning, pose estimation 및 application에 대한 survey를 제시하였다.  
hand pose estimation에 대한 review [19]는 에 의해 제시되며 [20]은 head pose estimation에 대한 survey를 제공한다.

One of the recent surveys on 2D human pose estimation based on deep learning is [17].  
This review started by categorizing pose estimation as a single person and multi-person pipeline and in each category created sub-categories.  
Another survey on deep-learning-based pose estimation has just come out [18] on both 2D and 3D pose estimation.  
2D human pose estimation is categorized as [17] while 3D human pose estimation is categorized as model-free and model-based and the approaches are discussed based on these categories in both cases.

> 2D human pose estimation based on deep learning에 대한 최근 surveys 중 하나는 [17]이다.
이 review는 pose estimation을 single person와 multi-person pipeline으로 분류하고 생성된 각 범주에서 하위 범주로 분류함으로써 시작되었다.
deep-learning-based pose estimation에 대한 그리고 2D 및 3D pose estimation에 대한 또 다른 survey[18]가 최근 나왔다.
2D human pose estimation은 [17]로 분류되는 반면, 33D human pose estimation은  model-free 과 model-based로 분류되며, 두 경우 모두 이러한 범주에 기초하여 접근법이 논의된다.

This survey paper presents different deep learning-based 2D human pose estimation models.  
The backbone architecture used, loss functions, the datasets used, as well as evaluation metrics implied are discussed and evaluated.  
The main objective of this paper is to provide a detailed analysis of mostly known effective models used, provide readers with various opportunities in mixing architecture of different models so that to come up with better human pose estimation models using better evaluation metrics or efficient backbone architecture.

> 본 survey paper는 다양한 deep learning-based 2D human pose estimation models을 제시한다.  
사용된 backbone architecture, loss functions, 사용된 datasets 및 암시된 evaluation metrics이 논의되고 평가된다.  
본 논문의 주요 목표는 주로 사용되는 것으로 알려진 effective models에 대한 상세한 분석을 제공하고, 독자에게 다양한 모델의 architecture를 혼합하는 다양한 기회를 제공하여 더 나은 evaluation metrics 또는 효율적인 backbone architecture를 사용하는 더 나은 human pose estimation models을 고안하는 것이다.

### B. BASIC STEPS IN POSE ESTIMATION

![Fig2](/assets/img/Blog/papers/Pose/survey_2020/Fig2.JPG)

The main process of human pose estimation is boiled into two basic steps:  
i) localizing human body joints/keypoints and;  
ii) grouping those joints into valid human pose configuration [7], [8].   
In the first step, the main focus is on finding the location of each keypoints of human beings as displayed in Fig.2. E.g. Head, shoulder, arm, hand, knee, ankle.

> human pose estimation의 주요 과정은 두 가지 기본 단계로 나뉜다.  
i) human body joints/keypoints 위치 파악 및  
ii) 이 joints를 유효한 인체 포즈 구성으로 그룹화[7], [8]   

_In the first step_, the main focus is on finding the location of each keypoints of human beings as displayed in Fig.2. E.g. Head, shoulder, arm, hand, knee, ankle.  

Collecting and identifying these joints can be done through any of the different popular dataset formats; such that the way keypoints are stored in the selected dataset.  
As shown in Fig.2, different platforms can result in different dataset output formats for the same image of body joints.  
For instance, COCO [21], dataset provide 17 body joints whereas MPII [22] provides 14 body joints.  
Table 1 displays the outputs dataset for the two platforms.

> 첫 번째 단계에서 주요 초점은 머리, 어깨, 팔, 손, 무릎, 발목과 같은 Fig.2에 표시된 인간의 각 keypoints의 location을 끝내는 것이다.  
이러한 joints를 수집하고 식별하는 것은 ;keypoints가 선택한 dataset에 저장되는 방법 등과 같이; 널리 사용되는 여러 dataset formats을을 통해 수행할 수 있다.  
Fig.2에서 보듯이, 다른 플랫폼들은 동일한 body joints 이미지에 대해 다른 dataset output formats을 초래할 수 있다.  
예를 들어, COCO [21] dataset는 17개의 body joints를 제공하는 반면, MPII [22]는 14개의 body joints를 제공한다.  
Table 1에는 두 플랫폼에 대한 outputs dataset를 표시하고 있다.

_The second step_ is grouping those joints into valid human pose configuration which determines the pairwise terms between body parts as seen in Fig.3.  
Different techniques have been applied in joining the keypoint candidates [23], [24].

> 두 번째 단계는 Fig.3에서 보는 바와 같이 body parts 사이의 pairwise terms을 결정하는 유효한 인체 자세 구성으로 이 joints을 그룹화하는 것이다.  
keypoint 후보군에 합류하는 데는 다양한 기술이 적용되었다 [23], [24].

The rest of this paper is organized as follows.  
Section II describes the category of pose estimation based on the number of people needed to track, approaches used in pose estimation, application of pose estimation, and flaws/challenges in pose estimation.  
Section III started by the introduction of backbone architectures used, the loss functions, dataset implied, and finally common evaluation metrics used to evaluate models.  
In section IV, a detailed discussion of each model's network procedures is discussed.  
Section V summarizes the models in short as a table and opens discussion based on presented in this article and finally section VI concludes the paper's works.

> 본 논문의 나머지 부분은 다음과 같이 구성되어 있다.  
Section II에서는 tracking에 필요한 pose estimation based on the number of people의 category, pose estimation에 사용된 approaches, pose estimation의 application, pose estimation의 flaws/challenges를 설명한다.  
Section III는 사용된 backbone architectures, loss functions, 암시된 dataset, 마지막으로 models 평가에 사용되는 공통 evaluation metrics의 도입으로 시작되었다.  
Section IV에서는 각  model's network procedures에 대한 자세한 discussion이 논의된다.  
Section V는 models을 간략하게 table로 요약하고 이 글에 제시된 내용을 기반으로 discussion을 열고  
마지막으로 Section VI는 본 논문의 작업을 마무리한다.

![Table1](/assets/img/Blog/papers/Pose/survey_2020/Table1.JPG)

## II. POSE ESTIMATION PRELIMINARY

This section discusses the general classification of pose estimation based on the number of people to track, introduce the most popular approaches, application of pose estimation, and finally, the challenges that still require new as well as innovative approaches.

> 이 section에서는 tracking을 위한 pose estimation based on the number of people의 일반적인 분류, 가장 인기 있는 approaches 소개, pose estimation의 application, 마지막으로 여전히 새롭고 혁신적인 approaches이 필요한 challenges에 대해 설명다.

![Fig3](/assets/img/Blog/papers/Pose/survey_2020/Fig3.JPG)

### A. SINGLE/MULTI-PERSON POSE ESTIMATION

Based on the number of individuals being estimated given an image, pose estimation is classified as single-person and/or multi-person pose estimation.  
Single-person pose estimation is much easier compared to multi-person, to estimate pose for a single person from a given image which may contain (usually does) more than a single person.  
On the other hand, multi-person pose estimation determines the pose of all individuals available in the image [25].  
Fig.4 shows the approach of a single person and multi-person pose estimation applied in the given images.

> Based on the number of individuals being estimated given an image, pose estimation은 single-person and/or multi-person pose estimation으로 분류된다.  
Single-person pose estimation은 multi-person에 비해 훨씬 쉬우며, single person이상이 포함된 image에서 single person의 pose를 estimate한다.
반면에,  multi-person pose estimation은 이미지에서 사용 가능한 모든 개인의 pose를 결정한다 [25].  
Fig.4는 주어진 영상에 적용된 single person 과 multi-person pose estimation의 approach를 보여준다.

![Fig4](/assets/img/Blog/papers/Pose/survey_2020/Fig4.JPG)

The use and introduction of deep learning-based architectures [36]-[39] and the availability of large-scale datasets such as MPII human pose dataset [22], COCO [21], and LSP [40] both single and multi-person pose estimation problems have lately been getting attention more and more.

> deep learning-based architectures [36]-[39]의 사용 및 도입과  MPII human pose dataset [22], COCO[21], LSP[40]와 같은 large-scale datasets의 가용성이 최근 single 및 multi-person pose estimation 문제에 점점 더 주목을 받고 있다.

### B. APPROACHES IN POSE ESTIMATION

![Fig5](/assets/img/Blog/papers/Pose/survey_2020/Fig5.JPG)

Two common approaches are employed in estimating the poses of individuals in a given image.  
1) Top-down approaches, the processing is done from low to high resolutions, follow the detection of the individual instances in the image first using a bounding box object detector and then focus on determining their poses next [26], [27], [29], as shown in Fig.5.

> given image에서 개인의 poses를 추정하는 데 두 가지 일반적인 approaches가 사용된다.  
1) Top-down approaches은 low resolutions에서 high resolutions로 처리되며, 먼저 bounding box object detector를 사용하여 image의 개별 instances 탐지를 수행한 후 Fig.5와 같이 다음의 poses를 결정하는 데 초점을 맞춘다 [26], [27], [29].

These approaches always suffer from early commitment, which means if the detection of individuals fails, there is no possibility of recovering.  
Also, it is vulnerable when multiple individuals are nearby.  
Furthermore, the computational cost depends on the number of people in the image, the more the people the more the computational cost.  
Hence, the run-time of these approaches is directly proportional to the number of people: means for every detection, a single-person pose estimator is run.

> 이러한 approaches는  early commitment로 인해 어려움을 겪는데, 이는 개인 탐지가 실패할 경우 recovering될 가능성이 없다는 것을 의미한다.(= detect를 못하면 pose estimation을 전혀 할수 없음)  
또한 여러 사람이 근처에 있을 때 취약하다.  
더욱이, 계산 비용은 image의 사람 수에 따라 달라지며, 사람이 많을수록 계산 비용이 증가한다.  
따라서 이러한 approaches의 run-time은 사람 수에 정비례한다: 모든 탐지에 대해 평균은 single-person pose estimator가 실행된다.

2) The bottom-up approaches [31]-[34] processing is done from high to low resolutions.  
It starts by localizing identity-free semantic entities and then grouping them into person instance.  
Bottom-up approaches overcame early commitment and showed detached run-time complexity from the number of people in the image as shown in Fig.6.  
In addition to that, some researches using bottom-up approaches use the global contextual cues from other body parts and other people.  
However, bottom-up approaches face challenges in grouping body parts when there is a large overlap between people.

> 2) bottom-up approaches [31]-[34] processing은 high resolutions에서 low resolutions으로 수행된다.
그것은 identity-free semantic entities를 지역화한 다음 그것들을  person instance로 그룹화하는 것으로 시작한다.
ttom-up approaches은 early commitment을 극복하고 Fig.6에 표시된 것처럼 image의 사람 수로부터 분리된 run-time 복잡성을 보여주었다.
또한 bottom-up approaches를 사용한 일부 연구는 다른 body parts 및 다른 사람의 global contextual 단서를 사용한다.
그러나 bottom-up approaches은 사람 간에 큰 겹침이 있을 때 body parts를 그룹화하는 데 있어 어려움에 직면한다.

![Fig6](/assets/img/Blog/papers/Pose/survey_2020/Fig6.JPG)

### C. APPLICATIONS OF POSE ESTIMATION

Earlier human pose estimation application areas such as action recognition, human tracking, animation, and gaming [41], [42] are mentioned.  
Video surveillance, assisted living, and advanced driver assistance systems (ADAS) [43], [44] are also included.  
Furthermore, it may also provide game analysis in sports by describing the players' movement [45].  
Pose estimation is also applicable in Sign languages to help disabled people.  
Some of the most common current applications of pose estimation are depicted in Fig.7.

> 이에 앞서 action recognition, human tracking, animation, gaming와 같은 human pose estimation 적용분야[41], [42]가 언급되어 있다.  
Video surveillance, assisted living, advanced driver assistance systems (ADAS)가 [43], [44]에 포함되어 있다.  
게다가, 그것은 또한 players' movement [45]를 설명함으로써 스포츠에 게임 분석을 제공할 수 있다.  
또한 Pose estimation은 Sign languages로 장애인들을 돕는 것에도 적용가능하다.
어떤 pose estimation의 가장 흔한 현재 적용은 Fig.7를 참고.

### D. CHALLENGES FACING

Principally every state-of-the-art (SOTA) pose estimation model includes a component that detects body joints or estimates their position and making pairwise terms between body part hypotheses which assist categorizing the pairwise terms into valid human pose configurations.  
In doing so, some challenges are faced.  
Such as position and scale of each person in the image; barely visible joints; interactions between people, which brings complex spatial interference due to clothing, lighting changes, contact, occlusion of individual parts by clothes, backgrounds, and limb articulations which makes the association of parts difficult.  
As the cost of 3D depth-sensing camera decreases and the machine learning algorithms to process the datasets of such technologies improve, we believe that it would bring new approaches to solve current challenges.

> 기본적으로 모든 최신 (SOTA) pose estimation model에는 body joints을 detect하거나 position을 estimate하는 구성 요소(component) 그리고 pairwise terms를 valid human pose configurations로 분류하는 데 도움이 되는 body part hypotheses간에 pairwise terms을 만드는것을 포함한다.
그렇게함으로써 몇 가지 도전에 직면하게된다.
image에서 각 사람의 position와 scale과 같은; 간신히 보이는 joints, 옷 때문에 복잡한 공간 간섭을 가져오는 사람들간의 상호작용, 조명 변화, 접촉, 옷에 의한 ndividual parts의 occlusion, 배경, 팔다리 articulations이 parts 연결을 어렵게 만든다.
3D depth-sensing camera의 비용이 감소하고 이러한 기술의 datasets를 처리하는 machine learning algorithms이 개선됨에 따라 현재 문제를 해결하기위한 새로운 approaches를 가져올 것이라고 믿는다.

## III. MAIN COMPONENTS OF POSE ESTIMATION

Before diving into the details of each research model, better to explore first the main components of the pose estimation research's fundamentals such as backbone architecture, pose loss functions inhabited, the dataset used, and also evaluation metrics applied.

> 각 연구 model의 세부 정보를 살펴보기 전에, 먼저 backbone architecture, 사용되는 pose loss functions, 사용된 dataset, 적용된 evaluation metrics과 같은 pose estimatio 연구의 기초의 main components를 탐색하는 것이 좋다.

### A. BACKBONE ARCHITECTURE

DeepPose [46] is the first significant research article that applied deep learning to human pose estimation.  
The authors have implemented the network architecture of AlexNet [37] as backbone architecture which consists of five convolution layers, two fully connected layers, and a softmax classifier.  
After the introduction of AlexNet, other machine learning algorithms such as R-CNN [47], Fast R-CNN [48], FPN [49], Faster R-CNN [50] and Mask R-CNN [39] have been used as backbone architecture for other human pose estimation researches [51], [32] and [52].  
The second most popular backbone architecture is VGG [36] which has been used in [29], [34].  
Although AlexNet and VGG have been in use for a while, most of the recent researches in human pose estimation [26], [27], [31], [32], [35], [53], have been using ResNet [38] as a backbone architecture.

> DeepPose[46]는 deep learning을 human pose estimation에 최초로 적용한 중요 연구이다.  
저자들은 다섯 개의 convolution layers, 두 개의 fully connected layers, 그리고 softmax classifier로 구성된 backbone architecture로서 AlexNet [37]의 network architecture를 구현했다.  
AlexNet의 도입 이후, R-CNN[47], Fast R-CNN[48], FPN[49], Fast R-CNN[50] 및 Mask R-CNN[39]과 같은 다른 머신 러닝 알고리즘은 다른 human pose estimation 연구[51], [32], [52]의 backbone architecture로 사용되어 왔다.   
두 번째로 널리 사용되는 backbone architecture는 [29], [34]에서 사용된 VGG[36]이다.  
비록 AlexNet과 VGG가 한동안 사용되어 왔지만, human pose estimation [26], [27], [31], [32], [35], [53]의 최근 연구는 ResNet[38]을 backbone architecture로 사용하고 있다.

### B. LOSS FUNCTIONS

As one part of machine learning, human pose estimation models learn by loss functions.  
Loss functions evaluate how well a specific algorithm models the given dataset.  
It reduces the error in the prediction process [54], [55].  
Largely three kinds of loss functions applied in human pose estimation models, namely, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Cross-Entropy loss.

> 머신 러닝의 한 부분으로서, human pose estimation models은 loss functions를 통해 학습한다.  
Loss functions는 특정 알고리즘이 주어진 데dataset를 얼마나 잘 모델링하는지 평가한다.  
이렇게 하면 prediction process [54], [55]의 오차가 줄어든다.  
human pose estimation models에 적용되는 loss functions의 종류는 크게 세 가지로, 즉, Mean Absolute Error (MAE), Mean Squared Error (MSE), Cross-Entropy loss이다.  

MAE or L1 loss function is calculated as the average of sums of all absolute differences between true and predicted values.  
L1 loss function does not consider the direction, but only measures the magnitude of the error.  
The L1 loss function is not sensitive to outliers thus it is robust.  
But it is very hard to regress precisely which brings complexity for machine learning [56].

$$L_1 = \frac{1}{n} \sum^n_{i=1} \vert y_i - f(x_i)\vert , \; \; \; (1)$$

> MAE 또는 L1 loss function는 true 값과 예측 값 사이의 모든 절대 차이 합계의 평균으로 계산된다.  
L1 loss function는 방향을 고려하지 않고 오차의 크기만 측정한다.  
L1 loss function는 특이치에 민감하지 않으므로 robust하다.  
그러나  machine learning에 복잡성을 가져오는 정확한 regress은 매우 어렵다 [56].


MSE also called L2 loss or Quadratic loss function is calculated as the average of the squared differences between true and predicted values.  
Like L1 loss, the L2 loss function measures the magnitude of error without considering the direction.  
L2 loss function provides an easier way to calculate gradients due to its mathematical properties.  
But, the L2 loss function is very sensitive to outliers, unlike L1 because of its usage of squaring when predicted values and true values are very far away occasionally [56].

$$L_2 = \frac{1}{n} \sum^n_{i=1} (y_i - f(x_i))^2$$

> L2 loss 또는 Quadratic loss function라고도 하는 MSE는 true 값과 예측 값 사이의 제곱 차이의 평균으로 계산된다.  
L1 loss과 마찬가지로, L2 loss function는 방향을 고려하지 않고 오차 크기를 측정한다.  
L2 loss function는 수학적 특성으로 인해 gradients를 계산하기 더 쉬운 방법을 제공한다.  
그러나, L2 loss function는 L1과 달리 특이치에 매우 민감한데, 이는 예측 값과 true 값이 때때로 매우 멀리 있을 때 제곱을 사용하기 때문이다 [56].

In Cross-Entropy loss (Negative Log-Likelihood or Log loss), each predicted probability is compared to the actual class output value (0 or 1) and a score is calculated that penalizes the probability based on the distance from the expected value [54].  
The penalty is logarithmic, offering a small score for small differences (0.1 or 0.2) and an enormous score for a large difference (0.9 or 1.0) [54].  
This means an algorithm with smaller Cross-Entropy loss is preferable, and if it has 0.0 Log loss, then it predicts perfect probability.

$$\text{Log}_{loss} = -(y_i \text{log}(f(x_i)) + (1-y_i)\text{log}(1-f(x_i))) \; \; \; (3)$$

> Cross-Entropy loss (Negative Log-Likelihood or Log loss)에서 각 예측 확률을 실제 클래스 출력 값(0 또는 1)과 비교하고 예상 값으로부터의 거리에 기초한 확률에 penalty를 주는 점수로 계산합니다 [54].  
penalty는 로그로 작은 차이(0.1 또는 0.2)에 작은 점수(0.9 또는 1.0)를 제공하고 큰 차이(0.9 또는 1.0)에 큰 점수를 제공합니다 [54].  
이는 더 작은 Cross-Entropy loss를 가지는 알고리즘이 바람직하다는 것을 의미하며, 0.0 Log loss이 있으면 완벽한 확률을 예측한다.

### C. DATASET

Researchers in human pose estimation have been mainly using the following four datasets which are freely available to the public: FLIC, LSP, MPII Human Pose, and COCO.  
Less known datasets such as Pascal VOC [57], SURREAL [58] for single-person in both 2D and 3D pose estimation, HumanEva, Human3.6 dataset, CrowdPose [59], and JTA [60] have also been used in human pose estimation.

> human pose estimation의 연구자들은 대중이 자유롭게 사용할 수 있는 다음과 같은 4개의 datasets를 주로 사용해 왔다: FLIC, LSP, MPII Human Pose, COCO.  
Pascal VOC [57], 2D 및 3D pose estimation 모두에서 single-person을 위한 SERPRIC [58], HumanEva, Human 3.6 dataset, CrowdPose [59], JTA [60]과 같은 덜 알려진 datasets도  human pose estimation에 사용되었다.

Frames Labeled In Cinema (FLIC) [61] dataset consists of a total of 5,003 images of which 80% (around 4,000 images) are used as training and 20% (around 1016 images) are used as testing dataset.  
FLIC dataset is acquired from popular 30 movies in Hollywood by running a person detector SOTA model on every tenth frame of 30 movies.  
These images contain individuals in different kinds of poses with different kinds of clothing.  
From the dataset, each individual is labeled with 10 body joints.  
In most cases, the FLIC dataset has been used for a single person and multi-person pose estimation models.

Leeds Sports Pose dataset (LSP) [40] and LSP Ext (LSP extension or sometimes expressed as LSPe) contain a combination of 11,000 training and 1,000 testing images from Flickr.  
These images are mostly from sports activities which make it very challenging in their appearance terms.  
In addition to that, most individuals in the image have scaled to roughly 150 pixels in length.  
In the LSP dataset, each individual's full body is labeled with a total of 14 joints which shows an increased number of joints compared to FLIC. To be specific, the LSP dataset has a total of 2,000 annotated images whereas LSP Ext has a total of 10,000 images.  
In most cases, both datasets have been used for single person pose estimation models.

Max Planck Institute for Informatics (MPII) Human Pose dataset [22] contains around 25,000 images from which composed of more than 40,000 individuals with annotated body joints.  
These images are collected on the purpose to show human activities every day.  
In MPII human pose dataset, each individual's body is labeled with 15 body joints as mentioned in the introduction section.  
As FLIC dataset, MPII Human pose dataset has also been used for a single person and multiperson pose estimation models.

Finally, the MS-COCO dataset has got huge attention for multi-person pose estimation models.  
MS-COCO or usually called COCO (Common Objects in Context) is a product of Microsoft (MS) [21].  
COCO dataset is a collection of a very large dataset with annotation types of object detection, keypoint detection, stuff segmentation, panoptic segmentation, and image captioning.  
A JSON file is used to store annotations.  
COCO dataset brought to the table a very interesting mix of data, with various human poses used in different body scales, also containing occlusion patterns, with unconstrained environments.  
COCO dataset contains a total of 200,000 images and these contain 250,000 people with keypoints from which each individual's instance is labeled with 17 joints.  
COCO dataset has been producing dataset starting from 2014 with a large amount.

### D. COMMON EVALUATION METRICS

Similar to any other research, human pose estimation also uses evaluation metrics to compare and contrast one model from the other.  
Some researchers claim the superiority of their model based on a metric they developed, which could lead to false performance improvement.   This section glances some of the commonly used evaluation metrics in human pose estimation research, such that a consistent result is presented in the field and also researchers new to the field can easily adapt to these metrics.

> 다른 연구와 마찬가지로, human pose estimation도 평가 지표를 사용하여 한 모델을 다른 모델과 비교하고 대조한다.  
일부 연구자들은 자신들이 개발한 metric를 기반으로 모델의 우수성을 주장하며, 이는 잘못된 성능 향상으로 이어질 수 있다.    
이 절에서는 human pose estimation 연구에서 일반적으로 사용되는 일부 evaluation metrics을 살펴본다.  
따라서 현장에서 일관된 결과가 제시되고 해당 분야에 새로 온 연구자도 이러한 metrics에 쉽게 적응할 수 있다.

1) Percentage of Correct Parts (PCP): this metric measures the detection rate of limbs.  
A limb (body part) is considered detected if the distance between the two predicted joint locations and the true limb joint locations is less than half of the limb length [46].  
PCP commonly referred also as PCP@0.5. Recently, PCP has not been preferred as an evaluation metric even though it was initially regarded as the go-to metric.  
This is because PCP penalizes shorter limbs. The higher the PCP the better the model.

2) Percentage of Detected Joints (PDJ): this metric is proposed to address the limitations observed in PCP.  
This evaluation metric defines a joint correctly detected if the distance between the predicted joint location and the true joint location is within a certain fraction of the torso diameter (the distance between the right hip and left shoulder).  
For instance, for PDJ@0.2, it means the distance between the predicted joint location and the true joint location should be less than 0.2 * torso diameter.  
By changing this fraction, detection rates are obtained for different degrees of localization precision.

3) Percentage of Correct Key-points (PCK): this also measures the distance between the predicted joint location and the true joint location.  
The PCK evaluation metric measures the body joints' localization accuracy.  
The criteria of PCK and PDJ are very similar except to the fact that the torso diameter is replaced with the maximum side length (or threshold) of the external rectangle of ground truth body joints [62].  
Thus, detecting a joint is considered correct if the distance between the predicted joint and the true joint is within a certain fraction of the specified threshold.  
Again, the higher the PCK the better the model.

4) PCKh is a modified version of PCK.  
PCKh's matching threshold is 50% of the head segment length (a portion of the head length is used as a reference at 50%).  
PCKh is also defined as the head-normalized probability of the correct keypoint metric [63].  
In PCKh, joint detection is considered correct if the predicted joint location is with a certain threshold from the true joint location.  
But the threshold should be adaptively selected based on the individual's size.  
It should fall within l pixels of the ground-truth position, where $$\alpha$$is a constant and $$l$$ is the head size that corresponds to 60% of the diagonal length of the ground-truth head bounding box.  
The PCKh@0.5 ($$ \alpha = 0.5$$) score is reported [63].  
To make the metric articulation independent, one probably better chooses to use the head size.

5) Area Under the Curve (AUC) measures the different range PCK thresholds (E.g., when $$\alpha$$ varies from 0 to 0.5) entirely.  
It informs howthe model is capable of distinguishing each body's joints.  
The higher the AUC, the better the model.

6) Object Keypoint Similarity (OKS) gives a measure of how a predicted keypoint is close to ground truth.  
OKS is much similar to IoU (Intersection over Union) in keypoint detection performance.  
So, when a model gets higher OKS, it means the overlap between the predicted keypoint and the truth is higher.

Besides the above evaluation metrics, Average Precision (AP) and mean Average Precision (mAP) are also used [64], [65].  
In addition to deep learning being advanced and having very large datasets, non-linear jumping systems discussed in [66]-[68] can also help to improve the efficiency of different algorithms in deep learning models.

## IV. MAJOR RESEARCHES IN HUMAN POSE ESTIMATION

We will now dive into some unique and most effective pose estimation models' network flow.  
While discussing each approach we will explain, howtheir architecture is organized? How CNN architecture got deeper [69]? Is it a single/multiperson model? what kind of loss functions the models are using? dataset implied and how evaluated the work?

### A. DeepPose

![Fig8](/assets/img/Blog/papers/Pose/survey_2020/Fig8.JPG)

DeepPose [46], a single person pose estimation model published in 2014, formulated body joints as a problem of a CNN-based regression (which is a class of DNN-based regression).  
The authors have used AlexNet [37] as a backbone architecture, to analyze the effects of jointly training a multi-staged architecture with repeated intermediate supervision.  
DeepPose refines the coarse pose to get better estimation using a cascade of regressors which output coordinates (x, y) of each joint.  
When joints are predicted in DeepPose cascaded regressors, images are cropped around that joint to feed for the next stage.  
This allows the subsequent regressors to learn features for finer scales because a higher resolution images guide them to better precision.

> 2014년에 발표된 single person pose estimation model인 DeepPose[46]는 CNN-based regression(DNN-based regression의 class)의 문제로 신체 관절을 공식화했다.  
저자들은 AlexNet[37]을 backbone architecture로 사용하여 반복적인 intermediate supervision으로 multi-staged architecture를 공동으로(jointly) 훈련하는 효과를 분석하였다.  
DeepPose는 각 joint의 좌표(x, y)를 출력하는 계단식(cascade) 회귀 분석기를 사용하여 더 나은 estimation을 얻기 위해 거친 포즈(coarse pose)를 정제한다.  
DeepPose cascaded regressors에서 joints가 예측되면 다음 단계를 위해 images이 해당 joint 주위에 crop된다.  
따라서 subsequent regressors에서는 고해상도 이미지가 더 정밀하게 유도되므로 더 미세한 scales를 위한 features를 학습할 수 있다.

DeepPose has a total of 3-stages cascade of regressors to estimate the pose of an individual in a given image.  
The Overall network structure of DeepPose is shown in Fig.8, in which the blue color shows the convolutional layers while the green shows the fully connected layers.  
The left schematic view shows the initial stage which contains the DNN-based regressor for the coarse pose.  
When joints are predicted at this stage, the image is cropped around the coordinates of the detected joint then passed to the next stage called the DNN-based refiner (on the right side of Fig.8) as an input.

> DeepPose는 주어진 이미지에서 개인의 pose를 estimate하기 위해 총 3-stages cascade of regressors를 가지고 있다.  
DeepPose의 전체 network 구조는 Fig.8에 나타나 있으며, 파란색은 convolutional layers를 나타내고 녹색은 fully connected layers를 보여준다.  
왼쪽 개략도는 coarse(거친) pose에 대한 DNN-based regressor를 포함하는 초기 단계를 보여준다.  
이 단계에서 joints가 예측되면 영상이 검출된 joint의 좌표 주변을 자른 다음 DNN 기반 정제기(그림 8의 오른쪽)라고 하는 다음 단계로 입력으로 전달됩니다.

The performance of this model is evaluated on two datasets (FLIC and LSP) using evaluation metrics of PCK and PCP.  
This model outperformed the previous SOTA works in most cases.

> 이 모델의 성능은 PCK 및 PCP의 평가 지표를 사용하여 두 개의 datasets(FLIC, LSP)에서 평가된다.  
이 모델은 대부분의 경우 이전 SOTA 작업보다 성능이 우수하다.

Even though producing the first CNN based human pose estimation model is very significant, the work has some limitations.  
The main problem was regressing to a location is very difficult.  
This increased the complexity of the learning which weakened generalization.  
Thus, DeepPose performed very poorly in some regions.  
However, it has been very helpful for recent SOTA researches to change the challenge to the problem of estimating heatmaps for available joints or keypoints.

> first CNN based human pose estimation model을 생산하는 것은 매우 중요하지만, 이 작업은 몇 가지 한계를 가지고 있다.  
가장 큰 문제는 location로 regressing하는 것이 매우 어렵다는 것이었다.  
이것은 일반화를 약화시키는 학습의 복잡성을 증가시켰다.  
따라서 DeepPose는 일부 regions에서 매우 낮은 성능을 보였다.  
그러나 최근의 SOTA 연구에서 사용 가능한 joints 또는 keypoints에 대한 heatmaps 추정 문제로 문제를 변경하는 것은 매우 도움이 되었다.

### B. ConvNet POSE: EFFICIENT OBJECT LOCALIZATION USING CONVOLUTIONAL NETWORKS

![Fig9](/assets/img/Blog/papers/Pose/survey_2020/Fig9.JPG)

In this paper [70], ConvNet architecture, multi-resolution CNN architecture is proposed to generate discrete heatmaps instead of continuous regression that predicts the probability of the location of individual joints in monocular RGB images.  
In ConvNet pose, different scale features are captured simultaneously using multiple resolution CNN architectures in parallel.

> 본 논문에서는 단안 RGB 이미지에서 개별 접합의 위치 확률을 예측하는 연속 회귀 분석 대신 이산 히트 맵을 생성하기 위해 다중 해상도 CNN 아키텍처를 제안한다.
ConvNet 포즈에서는 여러 해상도 CNN 아키텍처를 병렬로 사용하여 다양한 스케일 기능이 동시에 캡처된다.

This model implements a sliding window detector that produces a coarse heatmap output and this coarse heatmap is refined by 'pose refinement' ConvNet to get better localization which improves in recovering the spatial accuracy lost due to pooling in the initial model.  
This means the model contains a module (a convolutional network) for coarse localization, a module for sampling and cropping the features of ConvNet for each joint at a specified location (x, y), and also a module for fine-tuning as shown in Fig.9 which displays the model's overall network structure.

This model has shown the use of a ConvNet and a graphical model jointly.  
The spatial relationship between the joints is typically learned by the graphical model [71].  
The performance of the model is evaluated using PCK and PCKh@0.5 on FLIC [61] and MPII [22] dataset respectively in which outperformed the previous SOTA models.

This model implemented the joint use of a convolutional network and graphical model.  
Also, it revealed heatmaps are preferable than direct joint regression.  
Human poses are structured because of physical connections (like knees are rigidly related to hips and ankles), body part proportions, joint limits (like knees do not bend forward), left-right symmetries, interpenetration constraints, and others.  
Thus, modeling this structure realizes that detecting visible keypoints is easier and this directs on estimating the occluded keypoints which are very hard to detect.  
However, this model lacks structure modeling.

### C. CPM: CONVOLUTIONAL POSE MACHINES

![Fig10](/assets/img/Blog/papers/Pose/survey_2020/Fig10.JPG)

CPM [29] consists of a sequence of convolutional networks that produce a 2D belief map for the location of each keypoint.  
The sequential prediction framework provided by CPM helps them to learn rich implicit spatial information and feature representation of images at the same time.  
CPM is completely differentiable and the multi-stage architecture can be trained end-to-end.  
Thus, the image features and belief maps produced by the previous stage are given as input for the next stage (except the first stage) in CPM.  
One of the basic motivations for CPM is learning long-range spatial relationships and this is done using large receptive fields.  
Also, CPM used intermediate supervision after each stage to avoid the problem of vanishing gradients.

The overall network structure and receptive field of CPM are shown in Fig.10. CPM network is divided into multiple stages (the stage is used as hyper-parameter, usually D3) and at each stage, the confidence (belief) map of each keypoint is computed. Fig.10, (a) and (b) show the structures in the pose machine, (c) and (d) show the corresponding convolutional networks respectively, while (e) shows the receptive fields at different stages.

At the first stage a basic convolutional network, a classic VGG structure represented by X predicts the belief maps of each keypoint from the original input image. This leads to the condition that if the individual in the image has p joint points, then the belief map has p layers with each layer representing the joint point heatmap. Each layer's loss is added up as a total loss to achieve intermediate supervision which helped them in vanishing gradients.

For subsequent stages, stage $$>= 2$$, the structure is the same except the input to the network are two data: a belief map output from the previous stage and the result of the original image passed through X'. In addition to that, CPM showed that increasing the receptive field increases the accuracy of the prediction of keypoints. Furthermore, CPM implemented intermediate supervision after each stage to solve the vanishing gradients.

CPM implemented their model on three known datasets: MPII, LSP, and FLIC using evaluation metrics of PCK@0.1, PCK@0.2, and PCKh@0.5. It is noteworthy to mention that CPM achieved a PCKh@0.5 score of 10.76% higher than the previous SOTA on the most challenging part, the ankle.

CPM is the integration of the convolutional network to pose machines to learn image features and image-dependant spatial models to estimate human poses. Nevertheless, this work implemented a top-down approach on single person pose estimation, which leads to known errors and complexities of the top-down approach discussed earlier.

### D. STACKED HOURGLASS NETWORKS FOR HUMAN POSE ESTIMATION

![Fig11](/assets/img/Blog/papers/Pose/survey_2020/Fig11.JPG)

Stacked hourglass network [27], exactly lookalike of an hourglass stacked which are composed as steps of pooling and upsampling layers, is on the basic motivation of capturing information at every scale. In human pose estimation: an individual's orientation, limb arrangements, the relationship between adjacent joints, and other many cues that are best identified at different scales in a given image.

Thus, the stacked hourglass network is performing a repeated use of bottom-up (from high resolution to low resolution using pooling), top-down (from low resolution to high resolution using upsampling), and intermediate supervision to improve the network performance. The overall network structure of stacked hourglass modules is given in Fig.12. The hourglass stacked helps them to capture information on every scale means both global and local information is captured. It means skip connections are used to preserve spatial information in every resolution and pass it for upsampling.

Fig.11 displays a single hourglass module in which a single box represents a residual module. The primary module in an hourglass structure, residual or recurrent learning, is used for the bypass addition structure. This residual learning, composed of three convolutional layers with different scales in which batch normalization and ReLu inserted between them, extracts higher-level features while maintaining the primary level of information. The second path skips the path and contains only one kernel, A convolution layer with a scale of 1. Thus, only the data depth is changed not the data size.

For each hourglass module, a fourth-order residual module is used. The 4th-order Hourglass sub-network extracts features from the original scale to the 1/16 scale. It does not change the data size, only the data depth. The hourglass module is used to capture local information contained in pictures at different scales. At different scales, it may contain a lot of useful information, such as the position of the human body, the movements of the limbs, the relationship between adjacent joint points, and so on. First, the Conv layer and Max Pooling layer are used to scale features to a small resolution. At each Max Pooling (down-sampling), the network forks (branches) and convolves the features with the original pre-pooled resolution; After getting the lowest resolution features, the network starts up-sampling, and gradually combines feature information of different scales. The lower resolution here uses the nearest neighbor upsampling method, and two different feature sets are added element by element (which performs two different feature sets Add elements).

In stacked hourglass down-sampling uses max pooling, and up-sampling uses nearest-neighbor interpolation. The original image is down-sampled and input into the Hourglass sub-net. The output of Hourglass goes through two linear modules to get the final response graph. During this period, the Residual module and the convolutional layer are used to gradually extract features. The secondary used network centered around two Hourglass and repeats the second (latter) half of the primary network. The input of the second Hourglass contains three channels (paths): the input data of the first Hourglass, the output data of the first Hourglass, and the first-level prediction result. These three channels of data are fused by concat and add, and their scales are different, which reflects the currently popular idea of skip-level (jump) structure.

In the Stacked hourglass network both high-resolution to low-resolution processing and low-resolution to highresolution processing are symmetrical. The stacked hourglass was tested on MPII and FLIC dataset benchmarks using PCK@0.2 and PCKh@0.5 evaluation metrics in which surpassed all previous SOTA performance. In addition, this work has improved accuracy from 4-5%. on the joints difficult to detect (knees and ankles).

### E. DeeperCut: A DEEPER, STRONGER, AND FASTER MULTI-PERSON POSE ESTIMATION MODEL

![Fig12](/assets/img/Blog/papers/Pose/survey_2020/Fig12.JPG)

DeeperCut [33] is a more similar and an upgrade version of the approach presented in DeepCut [52]. DeeperCut has implied a strong body part detectors to generate effective bottom-up proposals for body joints and adapted the extremely deep Residual Network (ResNet [38]) for human body detection whereas DeepCut adapted Fast R-CNN [48] for the task. The proposed keypoints are assembled into a variable number of consistent body part configurations using image-conditioned pairwise terms.

Unlike DeepCut, DeeperCut used an incremental optimization strategy that explores the search space more efficiently which leads to both better performance and speed-up factors. Adapting ResNet allowed this work to tackle the problem of vanishing gradients because ResNet tackles the problem bypassing the state though identity layers and modeling residual functions.

Similar to DeepCut, DeeperCut jointly estimates the poses of every individual appeared in an image by minimizing a joint objective based on Integer Linear Programming (ILP). The authors started by making a set of body joint candidates (D) generated by body part detectors and a set of body joint classes(C) such as head, shoulder, and knee in which each candidate's joint has a unary score for every joint class. Adapting ResNet to the fully convolutional model for the sliding window-based body part detection usually brings a stride of 32px which is too coarse for effective joint localization. The authors showed by reducing the stride from 32px to 8px. Besides, to tackling the problem of vanishing gradients in adapting ResNet, DeeeperCut also achieves a large receptive field size which allows them to incorporate context when predicting locations of individual body joints.

After detecting the keypoints, DeeperCut implemented image-conditioned pairwise terms on proposed keypoints. First, an individual in the image is randomly selected and then the location is fixed for each keypoint at its ground truth location using the learned regression. Second, an individual pairwise score-maps will be done and this gets the shape of a cone which extends to the direction of the correct location, but these are visually fuzzy. Finally, by applying an incremental optimization strategy that uses a branch-and-cut algorithm to incrementally solve several pairwise instances to have a valid human pose configuration.

DeeperCut employed the model on LSP, MPII, and COCO dataset using an evaluation metric of AP and mAP which outperformed most of the previous SOTA models except CPM [29] which got similar performance in some cases. Evaluation is done on both single and multi-person pose estimation.

DeeperCut has introduced novel image-conditioned pairwise terms but still needs several minutes per given image (around 4 min/image). However, the pairwise representations are very hard to regress precisely. Additionally, it implemented the model with a batch size of 1 which increases the instability of the model.

### F. IEF: HUMAN POSE ESTIMATION WITH ITERATIVE ERROR FEEDBACK

![Fig13](/assets/img/Blog/papers/Pose/survey_2020/Fig13.JPG)

IEF human pose estimation [72] basically motivated on the concept of prediction, identify what is wrong on this prediction, and correct them iteratively, which is done by a top-down feedback mechanism. IEF employed a framework that extends the hierarchical feature extractor (ConvNet) to include both input and output spaces. In IEF, Error predictions are fed to the initial solution repeatedly and progressively by a self-correcting model as a replacement of directly identifying the keypoints in one go. This framework is called Iterative Error Feedback (IEF) and Fig.13 shows the implementation of IEF for human pose estimation.

On the left side of Fig.13, there is an input composed of the image I and the initially guessed keypoints $$y_0$$ (representation of the previous output $$y_{t-1}$$). Assume three keypoints to the head (red), the right wrist (green), and the left wrist (blue). Then, define input $$X_t = I \oplus g(y_{t-1})$$, where I represents the image and $$y_{t-1}$$ is the previous output. The function $$f(X_t)$$, modeled as a ConvNet, produces the correction $$\varepsilon_t$$ as output and this output is added to the current output $$y_t$$ to produce $$y_{t+1}$$ which means the correction is considered. The function $$g(y_{t+1})$$ converts every keypoint position into one Gaussian heatmap channel so that it can be part of the input with the image for the next iteration. This procedure is done repeatedly and progressively T times until getting a refined $$y_{t+1}$$ which is very close to the ground truth.

IEF human pose estimation evaluated their performance on two datasets (LSP and MPII) using a single evaluation metric PCKh@0.5. IEF introduced novelty and good work. The functions used, both $$f$$ and $$g$$, are learnable and also, $$f$$ is a ConvNet. This means $$f$$ has the ability to learn features over the joint input-output space.

### G. REALTIME MULTI-PERSON2D POSE ESTIMATION USING PART AFFINITY FIELDS

![Fig14](/assets/img/Blog/papers/Pose/survey_2020/Fig14.JPG)

Realtime multi-person 2D pose estimation [34] proposed a novelty approach to connect human body parts using Part Affinity Fields (PAF), a non-parametric method, to achieve bottom-up multi-person pose estimation model. The main motivation of this research is identifying the difficulties faced on detecting individual body joints involving multi-person such as the number of people in the image (infinity), the interaction between these people, irregular scale for each individual, increasing complexity, and others.

The overall pipeline and architecture of this model are shown in Fig.14. For a given input image (Fig.14. a), the location of each joint is determined by part confidence maps (Fig.14. b), and the location and orientation of the body parts are determined by PAF (Fig.14. c) a 2D vector that represents the degree of association between the body parts. These body part candidates are associated with the parsing step to perform a set of bipartite matching as shown in Fig.14 (d) and finally, assembled full-body pose because of parsing results in (e).

The two-branch multi-stage CNN network shown in Fig.14 receives an input of a feature map F of an image initialized by the first 10 layers of VGG architecture. The feed-forward model simultaneously predicts confidence maps S (shown in beige) for predicting the location of joints with J confidence maps for each joint $$(S = S_1, S_2, . . . , S_J)$$ and affinity fields L or a set of 2D vector fields (shown in blue) for encoding parts/limbs association which has C vectors corresponding to each limb $$(L = L_1, L_2, . . . , L_C)$$.

Thus, at the end of the first stage, the network outputs a set of detection confidence maps and part affinity fields. For the consecutive stages, the inputs will be the combination of the two previous stage outputs and the feature map F. Both the confidence maps and the part affinity fields are passed by the greedy inference to have the 2D keypoints for every individual in the image, called Bipartite matching. Furthermore, this work implemented intermediate supervision after each stage to solve the vanishing gradient's problems by restoring the gradients periodically.

This work is evaluated on COCO and MPII dataset using AP, mAP, and PCKh@0.5 evaluation metrics to achieve the best results compared to the existing SOTA models in terms of performance and efficiency.

### H. CPN: CASCADED PYRAMID NETWORK FOR MULTI-PERSON POSE ESTIMATION

![Fig15](/assets/img/Blog/papers/Pose/survey_2020/Fig15.JPG)

Cascaded Pyramid Network (CPN) for multi-person pose estimation model [32] is motivated with the concept of facing the challenging problems which are called "hard keypoints". These include occlusion of keypoints (Occluded by clothes or another person), invisible keypoints, complex backgrounds, etc. The authors proposed a top-down model for multiperson pose estimation with CPN network structure as shown in Fig.15. This CPN network structure is composed of two stages: GlobalNet and RefineNet. Relatively easy keypoints are estimated by the GlobalNet while the hard keypoints estimation is done by RefineNet using online hard keypoint mining loss.

CPN network structure uses a CNN model to identify some human keypoints called Visible easy keypoints, which are relatively easy to detect; for instance Nose, Left elbow, and right hand in the image below. Visible easy keypoints have somewhat a fixed shape and this helps in obtaining texture information which makes it easy to get contextual information around the location of the joints. Then there are visible hard keypoints that are obscured by clothes such as the left knee, right knee, and left hip. Additionally, some joints are hidden and hard to distinguish, not only obscured by clothes, such as the right shoulder in the image shown below. For such hard keypoints, which have no contextual information, increasing the local receptive field is required such that the context information can be further refined. Based on this concept CPN roughly categorized the human body joints into simple parts and difficult parts.

GlobalNet, composed of a forward CNN, is a simple regression model that focuses on easy to detect human keypoints usually eyes, arms, and other easy to detect parts. The purpose of RefineNet is to detect difficult-to-recognize human keypoints, called hard keypoints. RefineNet integrates multiple receptive fields information with the feature maps of the pyramid model generated by GlobalNet. Then finally all feature maps with the same size are concatenated such that a correction for ambiguous keypoints is obtained. RefineNet applies two things to mine the difficult keypoints 1) concat when using features of multiple layers and 2) online hard keypoints mining technology for the second-level network. In general, RefineNet combines low-level features and highlevel features through convolution operations.

By the use of RefineNet plus online hard keypoints mining, the model outperformed the previous SOTA models when implementing the model on the COCO dataset using AP and OKS evaluation metrics. CPN exhibits similar properties as stacked hourglass being symmetrical in both processing of high-to-low resolution and low-to-high resolution. It is easy to observe processing from high-resolution to low-resolution as part of a classification network and that it is heavy. Nevertheless, other-way processing (low-resolution to high Resolution) is relatively light.

### I. SIMPLE BASELINES FOR HUMAN POSE ESTIMATION AND TRACKING

![Fig16](/assets/img/Blog/papers/Pose/survey_2020/Fig16.JPG)

The main motivation behind simple baselines for human pose estimation and tracking [31] is that most of the recent models on human pose estimation are very complex and look different in structure but achieving very close results. simple baselines proposed a relatively simplified and intuitive model that consists of a few deconvolutional layers at the end of ResNet to estimate the keypoints heatmap. While most human pose estimation models like stacked hourglass [27] and CPN [32] use the structure composed of upsampling and convolution to increase the low-resolution feature map, simple baselines inserts several layers of deconvolution in ResNet which is a very simple way to expand the feature map to the size of the original image to generate the keypoints heatmap as shown in Fig.16.

In this article, both pose estimation and pose tracking are discussed, but our discussion focused on the former.
As mentioned earlier this model's network structure is straightforward: add several layers of deconvolution after ResNet to generate a heatmap for the individual keypoints. The takeaway from this work is that the more the deconvolution layers, the greater the resolution of the generated heatmap.

Simple baselines achieved better performance compared to the previous works with the COCO dataset using AP evaluation metrics simply and easily. Similar to CPN, highresolution to low-resolution processing is viewed as part of a classification network (such as ResNet and VGGNet), and this is heavy while processing low-resolution to highresolution is comparatively light.

### J. HRNet: DEEP HIGH-RESOLUTION REPRESENTATION LEARNING FOR HUMAN POSE ESTIMATION

![Fig17](/assets/img/Blog/papers/Pose/survey_2020/Fig17.JPG)

The usual trend applied in human pose estimation is downsampling high-resolution feature maps to low-resolution and then trying to recover a high-resolution value from low-resolution feature maps. Based on this motivation, this research proposed an intuitive and different model called High-Resolution Net (HRNet) to maintain a high-resolution representation throughout the process [35]. In Stacked hourglass [27] both high-to-low resolution and low-to-high resolution processes are symmetrical. Processing from high-resolution to low-resolution in both CPN [32] and simple baselines [31] considered as part of a classification network by the backbone architecture which is heavy, but the reverse process is relatively light.

There is a high-resolution sub-network at the first stage of this network architecture, as shown in Fig.17. Then gradually a high-to-low resolution sub-networks are added one by one to acquire the output of multiple stages. Finally, the output of multiple resolution sub-networks in parallel are connected. It performs repeated multi-scale fusions such that each high-resolution to low-resolution feature map representation can receive information from other parallel representation branches, again and again, to obtain a more informative high-resolution representation. In the end, the keypoints heatmap of the network output and the spatial resolution are more accurate. Because of repeated multi-scale fusions, HRNet does not need to use intermediate heatmap supervision, unlike the previous works.

HRNet consists of parallel high-to-low resolution subnetworks with repeated information exchange across multiresolution sub-networks (multi-scale fusion). The horizontal and vertical directions correspond to the depth of the network and the scale of the feature maps, respectively. There are three scale branches in total. The resolution of the feature map will not change during the forward propagation of each scale branch. Even though there will be information exchange between each scale branch, the three branches are different. For instance, in the forward process, branch 1 (the top branch in the figure) will downsample its feature map and then transfer it to branch 2. Branch 2 will also send the enlarged feature to branch 1 through upsampling. Two operations can be performed in the same stage.

HRNet is evaluated on COCO and MPII dataset using AP, mAP, PCKh@0.5 evaluation metrics to achieve better performance. HRNet introduced the connection of the outputs of high-to-low resolution sub-networks in parallel rather than the usual serial connection. This means it does not require to restore the resolution because high-resolution representations are maintained always.

### K. CFA: CASCADE FEATURE AGGREGATION FOR HUMAN POSE ESTIMATION

![Fig18](/assets/img/Blog/papers/Pose/survey_2020/Fig18.JPG)

CFA proposed a cascaded multiple hourglass and aggregates low, medium, and high-level features to better capture local detailed information and global semantic information [26]. The motivation behind CFA network architecture is combining the concept implied in the network architecture of Stacked hourglass [27], CPN [32], and HRNet [35].

The overall network structure of CFA is displayed in Fig.18. CFA consists of multiple hourglass networks that are summed up by elements. Feature aggregation shows that the hourglass network at each stage will predict the feature map, and the output of the previous layer is used as an input to the next stage at the same time.

In each stage of CFA, ResNet based hourglass network is applied, which is an encoder-decoder model designed based on the hourglass. The basic structure used for the encoder part is ResNet and the connection employed from the encoder to the decoder is highway connection. There are three different feature aggregations in all stages of the CFA model: Lowlevel feature aggregation, middle-level feature aggregation, and high-level feature aggregation.

Fig.19 briefly describes the feature aggregation between different stages of CFA. Detailed local information is accommodated in low-level features which help them in localizing the exact location of human joints. On the other side, to refine the localization in case of complex backgrounds and partial occlusions, there are high-level features that contain semantic information. Finally, all these different feature aggregations are forwarded as input for the next stage which brings prediction more stable.

CFA evaluated their model on the LIP and MPII datasets using only PCKh@0.5 evaluation metrics. This paper is currently at the top of the 2019 CVPR article based on the MPII dataset PCKh@0.5 evaluation index in the field of single person pose estimation.

![Fig19](/assets/img/Blog/papers/Pose/survey_2020/Fig19.JPG)

### L. OccNet: HUMAN POSE ESTIMATION FOR REAL-WORLD CROWDED SCENARIOS

![Fig20](/assets/img/Blog/papers/Pose/survey_2020/Fig20.JPG)

This model proposed in the motivation of estimating the pose of individuals in real-world crowded areas [73]. The challenges of estimating poses in such densely populated areas include people close to each other, mutual occlusions, and partial visibility. The method is a two-stage, top-down approach that localizes the individual first and then performs a single-person pose estimation for every detected person. This model proposed two occlusion detection networks Occlusion Net (OccNet) and Occlusion Net Cross Branch (OccNetCB) as shown in Fig.20, the backbone network is ResNet shown in beige.

In OccNet, to learn a joint representation in the previous layers the network splits after two transposed convolutions. OccNet produces two sets of heatmaps for the location of keypoints per pose: a heatmap for visible keypoints and a heatmap for occluded keypoints. The other architecture, OccNetCB, splits after only one transposed convolution. In OccNetCB, both branches have the opportunity to get information extracted by one another because in OccNetCB the output from both layers is shared.

The model has been evaluated on two datasets annotated on the crowded real-world situation: CrowdPose and JTA datasets using OKS and AP evaluation metrics.

### M. DarkPose: DISTRIBUTION-AWARE COORDINATE REPRESENTATION FOR HUMAN POSE ESTIMATION

![Fig21](/assets/img/Blog/papers/Pose/survey_2020/Fig21.JPG)

The main motivation behind the Distribution-Aware Coordinate Representation of Keypoint (DarkPose) is that the coordinate representation of the heatmap [30]. The assumption is that heatmap is never systematically investigated. Based on this concept, the authors have shown design limitations on the existing standard coordinate decoding method, and propose a principled distribution-aware decoding method. In addition to that, an accurate heatmap distribution for the unbiased model training instead of the usual coordinate encoding process is generated (i.e. transforming ground-truth coordinates to heatmaps).

Standard label representation in existing methods is coordinate heatmap as a 2-dimensional Gaussian distribution centered at the labeled coordinate of each keypoint of an individual. According to this work, the major obstacle in heatmap label representation and that is quadratic function's computational cost of the input image resolution which restrains CNN based models from processing the typically high-resolution raw imagery data. Hence, there is a need to down-sample all the person bounding box images into a small resolution then fed them to human pose estimation model to predict the location of each keypoint in the original image coordinate space which needs to transform to the original coordinate space, and this brings the problem of sub-pixel localization. Coordinate decoding from heatmap to coordinate is the last prediction of the location with the maximal activation. The network structure of DarkPose is shown in Fig.21.

Coordinate representation, the problem of coordinate encoding and decoding, focused on predicting joint coordinates in a given image. Coordinate decoding is a process of translating a predicted heatmap of each individual's joint into a coordinate in the original image space. Unlike the standard method of considering the second maximum activation to upsample the heatmaps to the original image resolution, DarkPose introduced the heatmap distributional statistics for disclosing the underlying maximum more accurately as shown in Fig.21 and this is employed using Taylor-expansion way. The heatmaps, predicted by a human pose estimation model, usually present multiple peaks around the maximum activation which causes negative effects on the performance of the decoding method. To overcome this issue, DarkPose inserted modulating the heatmap distribution before resolution recovery. In coordinate decoding method, three steps employed: heatmap distribution modulation, distributionaware joint localization by Taylor-expansion as sub-pixel accuracy, and resolution recovery to the original coordinate space. A limitation similar to Coordinate decoding is also observed in coordinate encoding in reducing the resolution. Some of the existing methods start by downsampling given the original image to the model input size. Therefore, in this case, transforming the ground-truth joint coordinates accordingly was necessary before generating heatmaps and this is done by using unbiased sub-pixel centered coordinate encoding.

DarkPose has come up with the concept of problems facing in coordinate representation and the model was evaluated on COCO and MPII using evaluation metrics of PCK and OKS.

## V. SUMMARY AND DISCUSSION

![Table2](/assets/img/Blog/papers/Pose/survey_2020/Table2.JPG)

This paper reviewed the progress made in pose estimations for human beings with selected and most notable researches made to our knowledge. This discussion started from Deep- Pose [46], the first well known and has been as reference for most researches in pose estimation progress. Then models have been selected based on their novelty, innovation, the influence made by the model, and other criteria. Table 2 summarizes the models with some criteria.

Human pose estimation deals with the process of inferring poses in an image [4]. To accomplish this objective different kind of techniques have been employed in each model. The techniques can be evaluated with criteria such as the backbone architecture, approaches followed, tracking single or multi-person, the dataset used, loss functions, and evaluation metrics employed.

As shown in Table 2, ResNet [38] nowadays is a default pick as backbone architecture in most models because of its property of solving the problem of vanishing gradients in addition to its great accuracy.

In tracking the number of people in a given image, models are classified as single or multi-person pose estimation. Substantial researches have been carried out in a single person pose estimation with very good results. Even though multiperson pose estimation getting attention, the challenges are still there. These challenges come from the position of each person in the image, visibility of the joints, scale difference, interaction between people, occlusion of joints by clothes, and others.

As shown in Table 2, researchers are preferring Top-down approach instead of bottom-up in most cases. There are also models using both approaches simultaneously.

In datasets selection, COCO and MPII are default picks in recent cases. Especially, COCO is a famous dataset by its property of having very wide human poses and an enormous number of images. LSP and FLIC datasets are also used next to COCO and MPII.

Even though the $$L_1$$ loss is not sensitive to outliers, the L2 loss function is applied in most models to evaluate their learning process. Finally, PCKh@0.5 is the number one evaluation metrics in human pose estimation before mAP and AP.

This article reviewed models focused on determining the full body's pose of individuals. Fascinating researches are also available in discovering only some parts of a human being. For instance: hand pose, head pose, upper body pose, and so on. Additionally, estimating the pose of only children is also presented in [74] research.

## VI. CONCLUSION

This paper presented a review of the most outstanding and influential models in human pose estimation progress. As introduced early a 2D human pose estimation has been a fundamental yet challenging problem in computer vision. The main objective of human pose estimation is to localize human anatomical keypoints (e.g., head, shoulder, elbow, wrist, etc.) or joints. This article started by introducing human pose estimation, then classified pose estimation based on tracing the number of people as a single or multi-person. Furthermore, approaches used in pose estimation are explored before discussing its applications and flaws. Finally, some significant papers on pose estimation in both cases of single or multi-person are briefly discussed.

Thus, this article provides a guideline for newreaders about human pose estimation. Furthermore, this paper can be a base for research to innovate new models by combining the techniques used in different papers mentioned above. This can be done by changing the backbone architecture or combining the two or three models to create new, or adding new architecture on one of the mentioned papers.

There are very large datasets publicly available on the net. Using these datasets, we have seen substantial progress in 2D human pose estimation with deep learning. However, in addition to the issues discussed in the summary and discussion section, some challenges remain to be addressed in the near future works. Such as i) occlusion of body parts by clothes and other people, ii) interactions between people, iii) human body structure constraints, and iv) barely visible joints are some of the prominent issues that need immense attention to be resolved in the coming works.
