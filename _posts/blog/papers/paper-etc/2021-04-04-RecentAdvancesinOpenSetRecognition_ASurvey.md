---
layout: post
bigtitle:  "Recent Advances in Open Set Recognition: A Survey"
subtitle:   "servey"
categories:
    - blog
    - papers
    - paper-etc
tags:
    - servey
comments: true
published: true
related_posts:
---
# Recent Advances in Open Set Recognition: A Survey

2020 [paper](https://arxiv.org/pdf/1811.08581.pdf)

Chuanxing Geng,  
Sheng-Jun Huang and  
Songcan Chen

---

* toc
{:toc}

## ABSTRACT

In real-world recognition/classification tasks, limited by various objective factors, it is usually difficult to collect training samples to exhaust all classes when training a recognizer or classifier.  
A more realistic scenario is open set recognition (OSR), where incomplete knowledge of the world exists at training time, and unknown classes can be submitted to an algorithm during testing, requiring the classifiers to not only accurately classify the seen classes, but also effectively deal with unseen ones.  
This paper provides a comprehensive survey of existing open set recognition techniques covering various aspects ranging from related definitions, representations of models, datasets, evaluation criteria, and algorithm comparisons.  
Furthermore, we briefly analyze the relationships between OSR and its related tasks including zero-shot, one-shot (few-shot) recognition/learning techniques, classification with reject option, and so forth.  
Additionally, we also review the open world recognition which can be seen as a natural extension of OSR.  
Importantly, we highlight the limitations of existing approaches and point out some promising subsequent research directions in this field.

> 다양한 객관적 요인에 의해 제한되는, real-world의 recognition/classification tasks에서 recognizer 또는 classifier를 훈련시킬 때 모든 클래스를 사용하기 위해 훈련 샘플을 수집하기는 대개 어렵다.  
보다 현실적인 시나리오는 label이 불완전하게 되어있는 실세계 training 데이터인 open set recognition (OSR)이다, 그리고 test 중에 unknown classes를 algorithm에 제출할 수 있어, classifiers가 seen classes를 정확하게 분류할 뿐만 아니라 unseen classes를 효과적으로 처리해야 한다.  
본 논문은 관련 정의, 모델의 representations, datasets, 평가 기준, algorithm 비교에서 다양한 측면을 다루는 기존의 open set recognition techniques에 대한 포괄적인 조사를 제공한다.  
또한 zero-shot, one-shot(few-shot) recognition/learning techniques, reject option과의 분류 등을 포함한 OSR과 관련 tasks 간의 관계를 간략하게 분석한다.  
또한 OSR의 자연스러운 확장으로 볼 수 있는 open world recognition도 검토한다.  
중요하게, 우리는 기존 approaches의 한계를 강조하고 이 분야에서 유망한 후속 연구 방향을 지적한다.


**Index Terms**—Open set recognition/classification, open world recognition, zero-short learning, one-shot learning.

## 1 INTRODUCTION

UNDER a common closed set (or static environment) assumption: the training and testing data are drawn from the same label and feature spaces, the traditional recognition/classification algorithms have already achieved significant success in a variety of machine learning (ML) tasks.  
However, a more realistic scenario is usually open and non-stationary such as driverless, fault/medical diagnosis, etc., where unseen situations can emerge unexpectedly, which drastically weakens the robustness of these existing methods.  
To meet this challenge, several related research topics have been explored including  
lifelong learning [1], [2],  
transfer learning [3], [4], [5],  
domain adaptation [6], [7],  
zero-shot [8], [9], [10],  
one-shot (few-shot) [11], [12], [13], [14], [15], [16], [17], [18], [19], [20]  
recognition/learning, open set recognition/classification [21], [22], [23], and so forth.

> [1] “Lifelong robot learning,” Robotics and Autonomous Systems, 1995.
[2] “A pac-bayesian bound for lifelong learning,” 2014.
[3]  “A survey on transfer learning,” 2010.
[4] “A survey of transfer learning,” 2016.
[5]  “Transfer learning for visual categorization: A survey,” 2015.
[6] “Visual domain adaptation: A survey of recent advances,” 2015.
[7]  “Domain adaptation for structured regression,” 2014.
[8] “Zero-shot learning with semantic output codes,” 2009.
[9] “Learning to detect unseen object classes by between-class attribute transfer,” 2009.
[10] “Recent advances in zero-shot recognition: Toward data-efficient understanding of visual content,” 2018.
[11] “One-shot learning of object categories,” 2006.

Based on Donald Rumsfeld’s famous “There are known knowns” statement [24],  
we further expand the basic recognition categories of classes asserted by [22], where we restate that recognition should consider four basic categories of classes as follows:

1) known known classes (KKCs), i.e., the classes with distinctly labeled positive training samples (also serving as negative samples for other KKCs), and even have the corresponding side-information like semantic/attribute information, etc;   
2) known unknown classes (KUCs), i.e., labeled negative samples, not necessarily grouped into meaningful classes, such as the background classes [25], the universum classes [26], etc;   
3) unknown known classes (UKCs), i.e., classes with no available samples in training, but available side-information (e.g., semantic/attribute information) of them during training;  
4) unknown unknown classes (UUCs), i.e., classes without any information regarding them during training: not only unseen but also having not side-information (e.g., semantic/attribute information, etc.) during training.

> 1) known known classes (KKCs), 즉 분명하게 레이블이 지정된 positive training samples이 있는 classes(다른 KKC에 대한 negative samples 역할도 함), 심지어 semantic/attribute information등과 같은 해당 side-information를 가지고 있다.   
2) known unknown classes (KUCs), 즉 배경 클래스 [25], universum classes [26] 등과 같은것들인, 의미 있는 클래스로 분류될 필요는 없는, 레이블이 지정된 negative samples.   
3) unknown known classes (UKCs), 즉 훈련 중에 사용 가능한 샘플이 없지만 훈련 중에 이용할 수 있는 side-information(예: semantic/attribute information)가 있는 클래스  
4)  unknown unknown classes (UUCs), 즉 훈련 중에 관련 정보가 없는 클래스: 훈련 중에 보이지 않을 뿐만 아니라 side-information(예: semantic/attribute information 등)도 없다.

![Fig1](/assets/img/Blog/papers/survet_recentadvances/Fig1.JPG)

Fig. 1 gives an example of visualizing KKCs, KUCs, and UUCs from the real data distribution using t-SNE [27].  
Note that since the main difference between UKCs and UUCs lies in whether their side-information is available or not, we here only visualize UUCs.  
Traditional classification only considers KKCs, while including KUCs will result in models with an explicit "other class," or a detector trained with unclassified negatives [22].  

Unlike traditional classification, zero-shot learning (ZSL) focuses more on the recognition of UKCs.  
As the saying goes: prediction is impossible without any assumptions about how the past relates to the future. ZSL leverages semantic information shared between KKCs and UKCs to implement such a recognition [8], [9]. In fact, assuming the test samples only from UKCs is rather restrictive and impractical, since we usually know nothing about them either from KKCs or UKCs. On the other hand, the object frequencies in natural world follow long-tailed distributions [28], [29], meaning that KKCs are more common than UKCs. Therefore, some researchers have begun to pay attention to the more generalized ZSL (GZSL) [30], [31], [32], [33], where the testing samples come from both KKCs and UKCs. As a closely-related problem to ZSL, one/few-shot learning (FSL) can be seen as natural extensions of zero-shot learning when a limited number of UKCs’ samples during training are available [11], [12], [13], [14], [15], [16], [17], [18], [19], [20]. Similar to G-ZSL, a more realistic setting for FSL considering both KKCs and UKCs in testing, i.e., generalized FSL (G-FSL), is also becoming more popular [34]. Compared to (G-)ZSL and (G- )FSL, open set recognition (OSR) [21], [22], [23] probably faces a more serious challenge due to the fact that only KKCs are available without any other side-information like attributes or a limited number of samples from UUCs.

![Fig2](/assets/img/Blog/papers/survet_recentadvances/Fig2.JPG)

Open set recognition [21] describes such a scenario where new classes (UUCs) unseen in training appear in testing, and requires the classifiers to not only accurately classify KKCs but also effectively deal with UUCs. Therefore, the classifiers need to have a corresponding reject option when a testing sample comes from some UUC. Fig. 2 gives a comparative demonstration of traditional classification and OSR problems. It should be noted that there have been already a variety of works in the literature regarding classification with reject option [35], [36], [37], [38], [39], [40], [41], [42], [43], [44], [45]. Although related in some sense, this task should not be confused with open set recognition since it still works under the closed set assumption, while the corresponding classifier rejects to recognize an input sample due to its low confidence, avoiding classifying a sample of one class as a member of another one.

In addition, the one-class classifier [46], [47], [48], [49], [50], [51], [52], [53] usually used for anomaly detection seems suitable for OSR problem, in which the empirical distribution of training data is modeled such that it can be separated from the surrounding open space (the space far from known/training data) in all directions of the feature space. Popular approaches for one-class classification include one-class SVM [46] and support vector data description (SVDD) [48], [54], where one-class SVM separates the training samples from the origin of the feature space with a maximum margin, while SVDD encloses the training data with a hypersphere of minimum volume. Note that treating multiple KKCs as a single one in the one-class setup obviously ignores the discriminative information among these KKCs, leading to poor performance [23], [55]. Even if each KKC is modeled by an individual one-class classifier as proposed in [37], the novelty detection performance is rather low [55]. Therefore, it is necessary to rebuild effective classifiers specifically for OSR problem, especially for multiclass OSR problem.

![Table1](/assets/img/Blog/papers/survet_recentadvances/Table1.JPG)

As a summary, Table 1 lists the differences between open set recognition and its related tasks mentioned above. In fact, OSR has been studied under a number of frameworks, assumptions, and names [56], [57], [58], [59], [60], [61]. In a study on evaluation methods for face recognition, Phillips et al. [56] proposed a typical framework for open set identity recognition, while Li and Wechsler [57] again viewed open set face recognition from an evaluation perspective and proposed Open Set TCM-kNN (Transduction Confidence Machine-k Nearest Neighbors) method. It is Scheirer et al. [21] that first formalized the open set recognition problem and proposed a preliminary solution—1-vs- Set machine, which incorporates an open space risk term in modeling to account for the space beyond the reasonable support of KKCs. Afterwards, open set recognition attracted widespread attention. Note that OSR has been mentioned in the recent survey on ZSL [10], however, it has not been extensively discussed. Unlike [10], we here provide a comprehensive review regarding OSR.

The rest of this paper is organized as follows. In the next three sections, we first give the basic notation and related definitions (Section 2). Then we categorize the existing OSR technologies from the modeling perspective, and for each category, we review different approaches, given in Table 2 in detail (Section 3). Lastly, we review the open world recognition (OWR) which can be seen as a natural extension of OSR in Section 4. Furthermore, Section 5 reports the commonly used datasets, evaluation criteria, and algorithm comparisons, while Section 6 highlights the limitations of existing approaches and points out some promising research directions in this field. Finally, Section 7 gives a conclusion.

## 2 BASIC NOTATION AND RELATED DEFINITION
