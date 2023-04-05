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
As the saying goes: prediction is impossible without any assumptions about how the past relates to the future.  
ZSL leverages semantic information shared between KKCs and UKCs to implement such a recognition [8], [9].  
In fact, assuming the test samples only from UKCs is rather restrictive and impractical, since we usually know nothing about them either from KKCs or UKCs.  
On the other hand, the object frequencies in natural world follow long-tailed distributions [28], [29], meaning that KKCs are more common than UKCs.  
Therefore, some researchers have begun to pay attention to the more generalized ZSL (GZSL) [30], [31], [32], [33], where the testing samples come from both KKCs and UKCs.  

As a closely-related problem to ZSL, one/few-shot learning (FSL) can be seen as natural extensions of zero-shot learning when a limited number of UKCs’ samples during training are available [11], [12], [13], [14], [15], [16], [17], [18], [19], [20].  
Similar to G-ZSL, a more realistic setting for FSL considering both KKCs and UKCs in testing, i.e., generalized FSL (G-FSL), is also becoming more popular [34].  
Compared to (G-)ZSL and (G-)FSL, open set recognition (OSR) [21], [22], [23] probably faces a more serious challenge due to the fact that only KKCs are available without any other side-information like attributes or a limited number of samples from UUCs.

![Fig2](/assets/img/Blog/papers/survet_recentadvances/Fig2.JPG)

<br>

Open set recognition [21] describes such a scenario where new classes (UUCs) unseen in training appear in testing, and requires the classifiers to not only accurately classify KKCs but also effectively deal with UUCs.  
Therefore, the classifiers need to have a corresponding reject option when a testing sample comes from some UUC.  

Fig. 2 gives a comparative demonstration of traditional classification and OSR problems.  
It should be noted that there have been already a variety of works in the literature regarding classification with reject option [35], [36], [37], [38], [39], [40], [41], [42], [43], [44], [45].  

Although related in some sense, this task should not be confused with open set recognition since it still works under the closed set assumption, while the corresponding classifier rejects to recognize an input sample due to its low confidence, avoiding classifying a sample of one class as a member of another one.

<br>

In addition, the one-class classifier [46], [47], [48], [49], [50], [51], [52], [53] usually used for anomaly detection seems suitable for OSR problem, in which the empirical distribution of training data is modeled such that it can be separated from the surrounding open space (the space far from known/training data) in all directions of the feature space.

Popular approaches for one-class classification include one-class SVM [46] and support vector data description (SVDD) [48], [54], where one-class SVM separates the training samples from the origin of the feature space with a maximum margin, while SVDD encloses the training data with a hypersphere of minimum volume.  

Note that treating multiple KKCs as a single one in the one-class setup obviously ignores the discriminative information among these KKCs, leading to poor performance [23], [55].  
Even if each KKC is modeled by an individual one-class classifier as proposed in [37], the novelty detection performance is rather low [55].  
Therefore, it is necessary to rebuild effective classifiers specifically for OSR problem, especially for multiclass OSR problem.

![Table1](/assets/img/Blog/papers/survet_recentadvances/Table1.JPG)

<br>

As a summary, Table 1 lists the differences between open set recognition and its related tasks mentioned above.  
In fact, OSR has been studied under a number of frameworks, assumptions, and names [56], [57], [58], [59], [60], [61].  
In a study on evaluation methods for face recognition, Phillips et al [56] proposed a typical framework for open set identity recognition, while Li and Wechsler [57] again viewed open set face recognition from an evaluation perspective and proposed Open Set TCM-kNN (Transduction Confidence Machine-k Nearest Neighbors) method.  

It is Scheirer et al [21] that first formalized the open set recognition problem and proposed a preliminary solution—1-vs-Set machine, which incorporates an open space risk term in modeling to account for the space beyond the reasonable support of KKCs.  
Afterwards, open set recognition attracted widespread attention.  
Note that OSR has been mentioned in the recent survey on ZSL [10], however, it has not been extensively discussed.  
Unlike [10], we here provide a comprehensive review regarding OSR.

<br>

The rest of this paper is organized as follows.  
In the next three sections, we first give the basic notation and related definitions (Section 2).  
Then we categorize the existing OSR technologies from the modeling perspective, and for each category, we review different approaches, given in Table 2 in detail (Section 3).  
Lastly, we review the open world recognition (OWR) which can be seen as a natural extension of OSR in Section 4.  
Furthermore, Section 5 reports the commonly used datasets, evaluation criteria, and algorithm comparisons, while Section 6 highlights the limitations of existing approaches and points out some promising research directions in this field.  
Finally, Section 7 gives a conclusion.

## 2 BASIC NOTATION AND RELATED DEFINITION

This part briefly reviews the formalized OSR problem described in [21].  
As discussed in [21], the space far from known data (including KKCs and KUCs) is usually considered as _open space_ $$\mathcal{O}$$.   
So labeling any sample in this space as an arbitrary KKC inevitably incurs risk, which is called open _space risk_ $$R_{\mathcal{O}}$$.  
As UUCs are agnostic in training, it is often difficult to quantitatively analyze open space risk.   Alternatively, [21] gives a qualitative description for $$R_{\mathcal{O}}$$, where it is formalized as the relative measure of open space $$\mathcal{O}$$ compared to the overall measure space $$S_o$$:

![(1)](/assets/img/Blog/papers/survet_recentadvances/(1).JPG)

where $$f$$ denotes the measurable recognition function.  
$$f(x) = 1$$ indicates that some class in KKCs is recognized, otherwise $$f(x) = 0$$.  
Under such a formalization, the more we label samples in open space as KKCs, the greater $$R_{\mathcal{O}}$$ is.  

Further, the authors in [21] also formally introduced the concept of openness for a particular problem or data universe.

<br>

**Definition 1.**  
(The openness defined in [21]) Let C<sub>TA</sub>, C<sub>TR</sub>, and C<sub>TE</sub> respectively represent the set of classes to be recognized, the set of classes used in training and the set of classes used during testing.  
Then the openness of the corresponding recognition task
$$O$$ is:
![(2)](/assets/img/Blog/papers/survet_recentadvances/(2).JPG)
where $$ \vert . \vert $$ denotes the number of classes in the corresponding set.

<br>

Larger openness corresponds to more open problems, while the problem is completely closed when the openness equals 0.  

Note that [21] does not explicitly give the relationships among C<sub>TA</sub>, C<sub>TR</sub>, and C<sub>TE</sub>.  

In most existing works [22], [67], [90], [91], the relationship, $$C_{TA} = C_{TR} \subseteq C_{TE}$$, holds by default.   

Besides, the authors in [82] specifically give the following relationship: $$C_{TA} \subseteq C_{TR} \subseteq C_{TE}$$, which contains the former case.  

However, such a relationship is problematic for Definition 1. Consider the following simple case: $$C_{TA} \subseteq C_{TR} \subseteq C_{TE}$$, and $$\vert C_{TA} \vert = 3, \vert C_{TR} \vert = 10; \vert C_{TE} \vert = 15$$.  
Then we will have $$O < 0$$, which is obviously unreasonable.  

In fact, C<sub>TA</sub> should be a subset of C<sub>TR</sub>, otherwise it would make no sense because one usually does not use the classifiers trained on C<sub>TR</sub> to identify other classes which are not in C<sub>TR</sub>.  

Intuitively, the openness of a particular problem should only depend on the KKCs’ knowledge from $$C_{TR}$$ and the UUCs’ knowledge from $$C_{TE}$$ rather than $$C_{TA}$$, $$C_{TR}$$, and $$C_{TE}$$ their three.  
Therefore, in this paper, we recalibrate the formula of _openness_:
![(3)](/assets/img/Blog/papers/survet_recentadvances/(3).JPG)

<br>

Compared to Eq. (2), Eq. (3) is just a relatively more reasonable form to estimate the _openness_.  
Other definitions can also capture this notion, and some may be more precise, thus worth further exploring.  
With the concepts of _open space risk_ and _openness_ in mind, the definition of **OSR** problem can be given as follows:

**Definition 2.**  
(The Open Set Recognition Problem [21]) Let $$V$$ be the training data, and let $$R_{\mathcal{O}}$$, $$R_{\mathcal{\epsilon}}$$ respectively denote the open space risk and the empirical risk.  
Then the goal of open set recognition is to find a measurable recognition function $$f \in \mathcal{H}$$, where $$f(x) > 0$$ implies correct recognition, and f is defined by minimizing the following $$Open Set Risk$$:  
![(3)](/assets/img/Blog/papers/survet_recentadvances/(4).JPG)
where $$\lambda_{r}$$ is a regularization constant.

<br>

The open set risk denoted in formula (4) balances the empirical risk and the open space risk over the space of allowable recognition functions.  
Although this initial definition mentioned above is more theoretical, it provides an important guidance for subsequent OSR modeling, leading to a series of OSR algorithms which will be detailed in the following section.

## 3 A CATEGORIZATION OF OSR TECHNIQUES

![(3)](/assets/img/Blog/papers/survet_recentadvances/Table2.JPG)

Although Scheirer et al. [21] formalized the OSR problem, an important question is how to incorporate Eq. (1) to modeling.  
There is an ongoing debate between the use of generative and discriminative models in statistical learning [92], [93], with arguments for the value of each.  

However, as discussed in [22], open set recognition introduces such a new issue, in which neither discriminative nor generative models can directly address UUCs existing in open space unless some constraints are imposed.  

Thus, with some constraints, researchers have made the exploration in modeling of OSR respectively from the discriminative and generative perspectives. Next, we mainly review the existing OSR models from these two perspectives.

<br>

According to the modeling forms, these models can be further categorized into four categories (Table 2): Traditional ML (TML)-based and Deep Neural Network (DNN)- based methods from the discriminative model perspective; Instance and Non-Instance Generation-based methods from the generative model perspective.

For each category, we review different approaches by focusing on their corresponding representative works.  
Moreover, a global picture on how these approaches are linked is given in Fig. 3, while several available software packages’ links are also listed (Table 3) to facilitate subsequent research by relevant researchers.  

Next, we first give a review from the discriminative model perspective, where most existing OSR algorithms are modeled from this perspective.

![(3)](/assets/img/Blog/papers/survet_recentadvances/Fig3.JPG)

![(3)](/assets/img/Blog/papers/survet_recentadvances/Table3.JPG)

### 3.1 Discriminative Model for OSR

#### 3.1.1 Traditional ML Methods-based OSR Models

As mentioned above, traditional machine learning methods (e.g., SVM, sparse representation, Nearest Neighbor, etc.) usually assume that the training and testing data are drawn from the same distribution.  
However, such an assumption does not hold any more in OSR.  
To adapt these methods to the OSR scenario, many efforts have been made [21], [22], [23], [62], [63], [64], [65], [66], [67], [68], [69], [70], [71], [72], [73], [74].

> Table 2 참조

<br>

**SVM-based**: The Support Vector Machine (SVM) [94] has been successfully used in traditional classification/ recognition task.  
However, when UUCs appear during testing, its classification performance will decrease significantly since it usually divides over-occupied space for KKCs under the closed set assumption.  
As shown in Fig. 2(b), once the UUCs’ samples fall into the space divided for some KKCs, these samples will never be correctly classified.  
To overcome this problem, many SVM-based OSR methods have been proposed.

<br>

Using Definition 2, Scheirer et al. [21] proposed 1-vs-Set machine, which incorporates an open space risk term in modeling to account for the space beyond the reasonable support of KKCs.  
Concretely, they added another hyperplane parallelling the separating hyperplane obtained by SVM in score space, thus leading to a slab in feature space.  
The open space risk for a linear kernel slab model is defined as follows:

![(3)](/assets/img/Blog/papers/survet_recentadvances/(5).JPG)

where $$\delta_{A}$$ and $$\delta_{\Omega}$$ denote the marginal distances of the corresponding hyperplanes, and $$\delta^{+}$$ is the separation needed to account for all positive data.

Moreover, user-specified parameters $$p_A$$ and $$p_{\Omega}$$ are given to weight the importance between the margin spaces $$w_A$$ and $$w_{\Omega}$$.  

In this case, a testing sample that appears between the two hyperplanes would be labeled as the appropriate class.  
Otherwise, it would be considered as non-target class or rejected, depending on which side of the slab it resides.  

Similar to 1-vs-Set machine, Cevikalp [62], [63] added another constraint on the samples of positive/target class based on the traditional SVM, and proposed the Best Fitting Hyperplane Classifier (BFHC) model which directly formed a slab in feature space.  
In addition, BFHC can be extended to nonlinear case by using kernel trick, and we refer reader to [63] for more details.

<br>

Although the slab models mentioned above decrease the KKC’s region for each binary SVM, the space occupied by each KKC remains unbounded. Thus the open space risk still exists. To overcome this challenge, researchers further seek new ways to control this risk [22], [23], [64], [65], [66].

<br>

Scheirer et al. [22] incorporated non-linear kernels into a solution that further limited open space risk by positively labeling only sets with finite measure. They formulated a compact abating probability (CAP) model, where probability of class membership abates as points move from known data to open space. Specifically, a Weibull-calibrated SVM (W-SVM) model was proposed, which combined the statis-tical extreme value theory (EVT) [95]<sup>3</sup> for score calibration with two separated SVMs.

The first SVM is a one-class SVM CAP model used as a conditioner: if the posterior estimate $$P_{O}(y\vert x)$$ of an input sample $$x$$ predicted by one-class SVM is less than a threshold $$\delta_{\mathcal{T}}$$, the sample will be rejected outright.

Otherwise, it will be passed to the second SVM.  

The second one is a binary SVM CAP model via a fitted Weibull cumulative distribution function, yielding the posterior estimate $$P_{\eta}(y \vert x)$$ for the corresponding positive KKC.  

Furthermore, it also obtains the posterior estimate $$P_{\psi}(y\vert x)$$ for the corresponding negative KKCs by a reverse Weibull fitting. Defined an indicator variable: $$\iota_{y} = 1$$ if $$P_{O}(y\vert x) > \delta_{\mathcal{T}}$$ and $$\iota_{y} = 0$$ otherwise, the W-SVM recognition for all KKCs $$\mathcal{Y}$$ is:

![(3)](/assets/img/Blog/papers/survet_recentadvances/(6).JPG)

where $$\delta_{R}$$ is the threshold of the second SVM CAP model.  
The thresholds $$\delta_{\mathcal{T}}$$ and $$\delta_{R}$$ are set empirically, e.g., $$\delta_{\mathcal{T}}$$ is fixed to 0.001 as specified by the authors, while $$\delta_{R}$$ is recommended to set according to the openness of the specific problem

![(3)](/assets/img/Blog/papers/survet_recentadvances/(7).JPG)

<br>

Besides, W-SVM was further used for open set intrusion recognition on the KDDCUP’99 dataset [96]. More works on intrusion detection in open set scenario can be found in [97].  
Intuitively, we can reject a large set of UUCs (even under an assumption of incomplete class knowledge) if the positive data for any KKCs is accurately modeled without overfitting.  
Based on this intuition, Jain et al[23] invoked EVT to model the positive training samples at the decision boundary and proposed the $$P_I$$-SVM algorithm.  
$$P_I$$-SVM also adopts the threshold-based classification scheme, in which the selection of corresponding threshold takes the same strategy in W-SVM.

<br>

Note that while both W-SVM and $$P_I$$-SVM effectively limit the open space risk by the threshold-based classification schemes, their thresholds’ selection also gives some caveats.  

First, they are assumed that all KKCs have equal thresholds, which may be not reasonable since the distributions of classes in feature space are usually unknown.  

Second, the reject thresholds are recommended to set according to the problem openness [22]. However, the openness of the corresponding problem is usually unknown as well.

<br>

To address these caveats, Scherreik et al. [64] introduced the probabilistic open set SVM (POS-SVM) classifier which could empirically determine unique reject threshold for each KKC under Definition 2.  

Instead of defining $$R_{\mathcal{O}}$$ as relative measure of open and class-defined space, POS-SVM chooses probabilistic representations respectively for open space risk $$R_{\mathcal{O}}$$ and empirical risk $$R_{\varepsilon}$$" (details c.f. [64]).  

Moreover, the authors also adopted a new OSR evalution metric called Youden’s index which combines the true negative rate and recall, and will be detailed in subsection 5.2.  

Recently, to address sliding window visual object detection and open set recognition tasks, Cevikalp and Triggs [65], [66] used a family of quasi-linear “polyhedral conic” functions of [98] to define the acceptance regions for positive KKCs.  

This choice provides a convenient family of compact and convex region shapes for discriminating relatively well localized positive KKCs from broader negative ones including negative KKCs and UUCs.

<br>

**Sparse Representation-based:** In recent years, the sparse representation-based techniques have been widely used in computer vision and image processing fields [99], [100], [101].  
In particular, sparse representation-based classifier (SRC) [102] has gained a lot of attentions, which identifies the correct class by seeking the sparsest representation of the testing sample in terms of the training.  
SRC and its variants are essentially still under the closed set assumption, so in order to adapt SRC to an open environment, Zhang and Patel [67] presented the sparse representation-based open set recognition model, briefly called SROSR.

<br>

SROSR models the tails of the matched and sum of nonmatched reconstruction error distributions using EVT due to the fact that most of the discriminative information for OSR is hidden in the tail part of those two error distributions.  

This model consists of two main stages.  
One stage reduces the OSR problem into hypothesis testing problems by modeling the tails of error distributions using EVT, and the other first calculates the reconstruction errors for a testing sample, then fusing the confidence scores based on the two tail distributions to determine its identity.

<br>

As reported in [67], although SROSR outperformed many competitive OSR algorithms, it also contains some limitations. For example, in the face recognition task, the SROSR would fail in such cases that the dataset contained extreme variations in pose, illumination or resolution, where the self expressiveness property required by the SRC do no longer hold. Besides, for good recognition performance, the training set is required to be extensive enough to span the conditions that might occur in testing set. Note that while only SROSR is currently proposed based on sparse representation, it is still an interesting topic for future work to develop the sparse representation-based OSR algorithms.

**Distance-based**: Similar to other traditional ML methods mentioned above, the distance-based classifiers are usually no longer valid under the open set scenario. To meet this challenge, Bendale and Boult [68] established a Nearest Non-Outlier (NNO) algorithm for open set recognition by extending upon the Nearest Class Mean (NCM) classifier [103], [104]. NNO carries out classification based on the distance between the testing sample and the means of KKCs, where it rejects an input sample when all classifiers reject it. What needs to be emphasized is that this algorithm can dynamically add new classes based on manually labeled data. In addition, the authors introduced the concept of open world recognition, which details in Section 4.

Further, based on the traditional Nearest Neighbor classifier, Junior et al. [69] introduced an open set version of Nearest Neighbor classifier (OSNN) to deal with the OSR problem. Different from those works which directly use a threshold on the similarity score for the most similar class, OSNN applies a threshold on the ratio of similarity scores to the two most similar classes instead, which is called Nearest Neighbor Distance Ratio (NNDR) technique.

Specifically, it first finds the nearest neighbor $$t$$ and $$u$$ of the testing sample $$s$$, where $$t$$ and $$u$$ come from different classes, then calculates the ratio

![(3)](/assets/img/Blog/papers/survet_recentadvances/(8).JPG)

where $$d(x, x')$$ denotes the Euclidean distance between sample $$x$$ and $$x'$$ in feature space.

If the Ratio is less than or equal to the pre-set threshold, $$s$$ will be classified as the same label
of $$t$$.  
Otherwise, it is considered as the UUC.

OSNN is inherently multiclass, meaning that its efficiency will not be affected as the number of available classes for training increases. Moreover, the NNDR technique can also be applied effortlessly to other classifiers based on the similarity score, e.g., the Optimum-Path Forest (OPF) classifier [105]. Other metrics could be used to replace Euclidean metric as well, and even the feature space considered could be a transformed one, as suggested by the authors. Note that one limitation of OSNN is that just selecting two reference samples coming from different classes for comparison makes OSNN vulnerable to outliers [91].

Margin Distribution-based: Considering that most existing OSR methods take little to no distribution information of the data into account and lack a strong theoretical foundation, Rudd et al. [70] formulated a theoretically sound classifier—the Extreme Value Machine (EVM) which stems from the concept of margin distributions. Various definitions and uses of margin distributions have been explored [106], [107], [108], [109], involving techniques such as maximizing the mean or median margin, taking a weighted combination margin, or optimizing the margin mean and variance. Utilizing the marginal distribution itself can provide better error bounds than those offered by a soft-margin SVM, which translates into reduced experimental error in some cases.

As an extension of margin distribution theory from a perclass formulation [106], [107], [108], [109] to a sample-wise formulation, EVM is modeled in terms of the distribution of sample half-distances relative to a reference point. Specifically, it obtains the following theorem:

**Theorem 1.** Assume we are given a positive sample xi and sufficiently many negative samples $$x_i$$ drawn from well-defined class distributions, yielding pairwise margin estimates $$m_{ij}$$. Assume a continuous non-degenerate margin distribution exists. Then the distribution for the minimal values of the margin distance for $$x_i$$ is given by a Weibull distribution.

As Theorem 1 holds for any point $$x_i$$, each point can estimate its own distribution of distance to the margin, thus yielding:

**Corollary 1.** ($$\Psi$$ **Density Function**) Given the conditions for the Theorem 1, the probability that $x'$$ is included in the boundary estimated by $$x_i$$ is given by

![(3)](/assets/img/Blog/papers/survet_recentadvances/(9).JPG)

where $$\| x_i - x'\| \vert$$ is the distance of $$x'$$ from sample $$x_i$$, and $$\kappa_i, \lambda_i$$ are Weibull shape and scale parameters respectively obtained from fitting to the smallest $$m_{ij}$$.

**Prediction:** Once EVM is trained, the probability of a new sample $$x'$$ associated with class $$\mathcal{C}_l$$, i.e., $$\hat{P}(\mathcal{C}_l\vert x')$$, can be obtained by Eq. (9), thus resulting in the following decision function

![(3)](/assets/img/Blog/papers/survet_recentadvances/(10).JPG)

where $$M$$ denotes the number of KKCs in training, and $$\delta$$ represents the probability threshold which defines the boundary between the KKCs and unsupported open space.

Derived from the margin distribution and extreme value theories, EVM has a well-grounded interpretation and can perform nonlinear kernel-free variable bandwidth incremental learning, which is further utilized to explore the open set face recognition [110] and the intrusion detection [111]. Note that it also has some limitations as reported in [71], in which an obvious one is that the use of geometry of KKCs is risky when the geometries of KKCs and UUCs differ. To address these limitations, Vignotto and Engelke [71] further presented the GPD and GEV classifiers relying on approximations from EVT.

**Other Traditional ML Methods-based:** Using centerbased similarity (CBS) space learning, Fei and Liu [72] proposed a novel solution for text classification under OSR scenario, while Vareto et al. [73] explored the open set face recognition and proposed HPLS and HFCN algorithms by combining hashing functions, partial least squares (PLS) and fully connected networks (FCN). Neira et al. [74] adopted the integrated idea, where different classifiers and features are combined to solve the OSR problem. We refer the reader to [72], [73], [74] for more details. As most traditional machine learning methods for classification currently are under closed set assumption, it is appealing to adapt them to the open and non-stationary environment.

#### 3.1.2 Deep Neural Network-based OSR Models

Thanks to the powerful learning representation ability, Deep Neural Networks (DNNs) have gained significant benefits for various tasks such as visual recognition, natural language processing, text classification, etc. DNNs usually follow a typical SoftMax cross-entropy classification loss, which inevitably incurs the normalization problem, making them inherently have the closed set nature. As a consequence, DNNs often make wrong predictions, and even do so too confidently, when processing the UUCs’ samples. The works in [112], [113] have indicated that DNNs easily suffer from vulnerability to ’fooling’ and ’rubbish’ images which are visually far from the desired class but produce high confidence scores. To address these problems, researchers have looked at different approaches [25], [75], [76], [77], [78], [79], [80], [81], [82], [83], [84], [85].

Replacing the SoftMax layer in DNNs with an OpenMax layer, Bendale and Boult [75] proposed the OpenMax model as a first solution towards open set Deep Networks. Specifically, a deep neural network is first trained with the normal SoftMax layer by minimizing the cross entropy loss. Adopting the concept of Nearest Class Mean [103], [104], each class is then represented as a mean activation vector (MAV) with the mean of the activation vectors (only for the correctly classified training samples) in the penultimate layer of that network. Next, the training samples’ distances from their corresponding class MAVs are calculated and used to fit the separate Weibull distribution for each class. Further, the activation vector’s values are redistributed according to the Weibull distribution fitting score, and then used to compute a pseudo-activation for UUCs. Finally, the class probabilities of KKCs and (pseudo) UUCs are computed by using SoftMax again on these new redistributed activation vectors.

As discussed in [75], OpenMax effectively addressed the recognition challenge for fooling/rubbish and unrelated open set images, but it fails to recognize the adversarial images which are visually indistinguishable from training samples but are designed to make deep networks produce high confidence but incorrect answers [113], [114]. Rozsa et al. [76] also analyzed and compared the adversarial robustness of DNNs using SoftMax layer with OpenMax: although OpenMax provides less vulnerable systems than SoftMax to traditional attacks, it is equally susceptible to more sophisticated adversarial generation techniques directly working on deep representations. Therefore, the adversarial samples are still a serious challenge for open set recognition. Furthermore, using the distance from MAV, the cross entropy loss function in OpenMax does not directly incentivize projecting class samples around the MAV. In addition to that, the distance function used in testing is not used in training, possibly resulting in inaccurate measurement in that space [77]. To address this limitation, Hassen and Chan [77] learned a neural network based representation for open set recognition. In this representation, samples from the same class are closed to each other while the ones from different classes are further apart, leading to larger space among KKCs for UUCs’ samples to occupy.

Besides, Prakhya et al. [78] continued to follow the technical line of OpenMax to explore the open set text classification, while Shu et al. [79] replaced the SoftMax layer with a 1-vs-rest final layer of sigmoids and presented Deep Open classifier (DOC) model. Kardan and Stanley [80] proposed the competitive overcomplete output layer (COOL) neural network to circumvent the overgeneralization of neural networks over regions far from the training data. Based on an elaborate distance-like computation provided by a weightless neural network, Cardoso et al. [81] proposed the tWiSARD algorithm for open set recognition, which is further developed in [82]. Recently, considering the available background classes (KUCs), Dhamija et al. [25] combined SoftMax with the novel Entropic Open-Set and Objectosphere losses to address the OSR problem. Yoshihashi et al. [83] presented the Classification-Reconstruction learning algorithm for open set recognition (CROSR), which utilizes latent representations for reconstruction and enables robust UUCs’ detection without harming the KKCs’ classification accuracy. Using class conditioned auto-encoders with novel training and testing methodology, Oza and Patel [85] proposed C2AE model for OSR. Compared to the works described above, Shu et al. [84] paid more attention to discovering the hidden UUCs in the reject samples. Correspondingly, they proposed a joint open classification model with a sub-model for classifying whether a pair of examples belong to the same class or not, where the sub-model can serve as a distance function for clustering to discover the hidden classes in the reject samples.

**Remark:** From the discriminative model perspective, almost all existing OSR approaches adopt the thresholdbased classification scheme, where recognizers in decision either reject or categorize the input samples to some KKC using empirically-set threshold. Thus the threshold plays a key role. However, at the moment, the selection for it usually depends on the knowledge from KKCs, which inevitably incurs risks due to lacking available information from UUCs [91]. In fact, as the KUCs’ data is often available at hand [25], [115], [116], we can fully leverage them to reduce such a risk and further improve the robustness of these methods for UUCs. Besides, effectively modeling the tails of the data distribution makes EVT widely used in existing OSR methods. However, regrettably, it provides no principled means of selecting the size of tail for fitting. Further, as the object frequencies in visual categories ordinarily follow long-tailed distribution [29], [117], such a distribution fitting will face challenges once the rare classes in KKCs and UUCs appear together in testing [118].

### 3.2 Generative Model for Open Set Recognition

In this section, we will review the OSR methods from the generative model perspective, where these methods can be further categorized into Instance Generation-based and Non-Instance Generation-based methods according to their modeling forms.

#### 3.2.1 Instance Generation-based OSR Models

The adversarial learning (AL) [119] as a novel technology has gained the striking successes, which employs a generative model and a discriminative model, where the generative model learns to generate samples that can fool the discriminative model as non-generated samples. Due to the properties of AL, some researchers also attempt to account for open space with the UUCs generated by the AL technique [86], [87], [88], [89], [90].

Using a conditional generative adversarial network (GAN) to synthesize mixtures of UUCs, Ge et al. [86] proposed the Generative OpenMax (G-OpenMax) algorithm, which can provide explicit probability estimation over the generated UUCs, enabling the classifier to locate the decision margin according to the knowledge of both KKCs and generated UUCs. Obviously, such UUCs in their setting are only limited in a subspace of the original KKCs’ space. Moreover, as reported in [86], although G-OpenMax effectively detects UUCs in monochrome digit datasets, it has no significant performance improvement on natural images .

Different from G-OpenMax, Neal et al. [87] introduced a novel dataset augmentation technique, called counterfactual image generation (OSRCI). OSRCI adopts an encoderdecoder GAN architecture to generate the synthetic open set examples which are close to KKCs, yet do not belong to any KKCs. They further reformulated the OSR problem as classification with one additional class containing those newly generated samples. Similar in spirit to [87], Jo et al. [88] adopted the GAN technique to generate fake data as the UUCs’ data to further enhance the robustness of the classifiers for UUCs. Yu et al[89] proposed the adversarial sample generation (ASG) framework for OSR.

ASG can be applied to various learning models besides neural networks, while it can generate not only UUCs’ data but also KKCs’ data if necessary. In addition, Yang et al. [90] borrowed the generator in a typical GAN networks to produce synthetic samples that are highly similar to the target samples as the automatic negative set, while the discriminator is redesigned to output multiple classes together with an UUC. Then they explored the open set human activity recognition based on micro-Doppler signatures.

**Remark:** As most Instance Generation-based OSR methods often rely on deep neural networks, they also seem to fall into the category of DNN-based methods. But please note that the essential difference between these two categories of methods lies in whether the UUCs’ samples are generated or not in learning. In addition, the AL technique does not just rely on deep neural networks, such as ASG [89].

#### 3.2.2 Non-Instance Generation-based OSR Models

Dirichlet process (DP) [120], [121], [122], [123], [124] considered as a distribution over distributions is a stochastic process, which has been widely applied in clustering and density estimation problems as a nonparametric prior defined over the number of mixture components. This model does not overly depend on training samples and can achieve adaptive change as the data changes, making it naturally adapt to the OSR scenario.

With slight modification to hierarchical Dirichlet process (HDP), Geng and Chen [91] adapted HDP to OSR and proposed the collective decision-based OSR model (CDOSR), which can address both batch and individual samples. CD-OSR first performs a co-clustering process to obtain the appropriate parameters in the training phase. In testing phase, it models each KKC’s data as a group of CD-OSR using a Gaussian mixture model (GMM) with an unknown number of components/subclasses, while the whole testing set as one collective/batch is treated in the same way. Then all of the groups are co-clustered under the HDP framework. After co-clustering, one can obtain one or more subclasses representing the corresponding class. Thus, for a testing sample, it would be labeled as the appropriate KKC or UUC, depending on whether the subclass it is assigned associates with the corresponding KKC or not.

Notably, unlike the previous OSR methods, CD-OSR does not need to define the thresholds using to determine the decision boundary between KKCs and UUCs. In contrast, it introduced some threshold using to control the number of subclasses in the corresponding class, and the selection of such a threshold has been experimentally indicated more generality (details c.f. [91]). Furthermore, CDOSR can provide explicit modeling for the UUCs appearing in testing, naturally resulting in a new class discovery function. Please note that such a new discovery is just at subclass level. Moreover, adopting the collective/batch decision strategy makes CD-OSR consider the correlations among the testing samples obviously ignored by other existing methods. Besides, as reported in [91], CD-OSR is just as a conceptual proof for open set recognition towards collective decision at present, and there are still many limitations. For example, the recognition process of CD-OSR seems to have the flavor of lazy learning to some extent, where the coclustering process will be repeated when other batch testing data arrives, resulting in higher computational overhead.

**Remark:** The key to Instance Generation-based OSR models is generating effective UUCs’ samples. Though these existing methods have achieved some results, generating more effective UUCs’ samples still need further study. Furthermore, the data adaptive property makes (hierarchical) Dirichlet process naturally suitable for dealing with the OSR task. Since only [91] currently gave a preliminary exploration using HDP, thus this study line is also worth further exploring. Besides, the collective decision strategy for OSR is a promising direction as well, since it not only takes the correlations among the testing samples into account but also provides a possibility for new class discovery, whereas single-sample decision strategy<sup>4</sup> adopted by other existing OSR methods cannot do such a work since it cannot directly tell whether the single rejected sample is an outlier or from new class.

## 4 BEYOND OPEN SET RECOGNITION

Please note that the existing open set recognition is indeed in an open scenario but not incremental and does not scale gracefully with the number of classes. On the other hand, though new classes (UUCs) are assumed to appear incremental in class incremental learning (C-IL) [125], [126], [127], [128], [129], [130], these studies mainly focused on how to enable the system to incorporate later coming training samples from new classes instead of handling the problem of recognizing UUCs. To jointly consider the OSR and CIL tasks, Bendale and Boult [68] expanded the existing open set recognition (Definition 2) to the open world recognition (OWR), where a recognition system should perform four tasks: detecting UUCs, choosing which samples to label for addition to the model, labelling those samples, and updating the classifier. Specifically, the authors give the following definition:

**Definition 3.** (Open World Recognition [68]) Let $$\mathcal{K}_T \in \mathbb{N}^+$$ be the set of labels of KKCs at time $$T$$, and let the zero label (0) be reserved for (temporarily) labeling data as unknown. Thus N includes the labels of KKCs and UUCs. Based on the Definition 2, a solution to open world recognition is a tuple $$[F,\varphi ,\nu , \mathcal{L}, I]$$ with:

1) A multi-class open set recognition function $$F(x) : \mathbb{R}^d \mapsto \mathbb{N}$$ using a vector function $$\varphi(x)$$ of $$i$$ per-class measurable recognition functions $$f_i(x)$$, also using a novelty detector $$\nu (\varphi) : \mathbb{R}_i \mapsto [0, 1]$$.  
We require the per-class recognition functions $$f_i(x) \in \mathcal{H} : \mathbb{R}^d \mapsto \mathbb{R}$$ for $$i \in \mathcal{K}_T$$ to be open set functions that manage open space risk as Eq. (1).  
The novelty detector $$\nu(\varphi) : \mathbb{R}_i \mapsto [0, 1]$$ determines if results from vector of recognition functions is from an UUC.

2) A labeling process $$\mathcal{L}(x) : \mathbb{R}^d \mapsto \mathbb{N}^+$$ applied to novel unknown data $$U_T$$ from time $$T$$, yielding labeled data $$D_T = \{(y_j, x_j)\}$$ where $$y_j = \mathcal{L}(x_j), \lor x_j \in U_T$$. Assume the labeling finds m new classes, then the set of KKCs becomes $$\mathcal{K}_{T+1} = \mathcal{K}_T \cup \{i + 1, ..., i + m\}$$.

3) An incremental learning function $$I_T (\varphi;D_T ) : \mathcal{H}^i \mapsto \mathcal{H}^{i+m}$$ to scalably learn and add new measurable functions $$f_{i+1}(x)...f_{i+m}(x)$$, each of which manages open space risk, to the vector $$\varphi$$ of measurable recognition functions.

For more details, we refer the reader to [68]. Ideally, all of these steps should be automated. However, [68] only presumed supervised learning with labels obtained by human labeling at present, and proposed the NNO algorithm which has been discussed in subsection 3.1.1.

Afterward, some researchers continued to follow up this research route. Rosa et al. [131] argued that to properly capture the intrinsic dynamic of OWR, it is necessary to append the following aspects: (a) the incremental learning of the underlying metric, (b) the incremental estimate of confidence thresholds for UUCs, and (c) the use of local learning to precisely describe the space of classes. Towards these goals, they extended three existing metric learning methods using online metric learning. Doan and Kalita [132] presented the Nearest Centroid Class (NCC) model, which is similar to the online NNO [131] but differs with two main aspects. First, they adopted a specific solution to address the initial issue of incrementally adding new classes. Second, they optimized the nearest neighbor search for determining the nearest local balls. Lonij et al. [133] tackled the OWR problem from the complementary direction of assigning semantic meaning to open-world images. To handle the openset action recognition task, Shu et al. [134] proposed the Open Deep Network (ODN) which first detects new classes by applying a multiclass triplet thresholding method, and then dynamically reconstructs the classification layer by adding predictors for new classes continually. Besides, EVM discussed in subsection 3.1.1 also adapts to the OWR scenario due to the nature of incremental learning [70]. Recently, Xu et al. [135] proposed a meta-learning method to learn to accept new classes without training under the open world recognition framework.

**Remark:** As a natural extension of OSR, OWR faces more serious challenges which require it to have not only the ability to handle the OSR task, but also minimal downtime, even to continuously learn, which seems to have the flavor of lifelong learning to some extent. Besides, although some progress regarding OWR has been made, there is still a long way to go.

## 5 DATASETS, EVALUATION CRITERIA AND EXPERIMENTS

### 5.1 Datasets

In open set recognition, most existing experiments are usually carried out on a variety of recast multi-class benchmark datasets at present, where some distinct labels in the corresponding dataset are randomly chosen as KKCs while the remaining ones as UUCs. Here we list some commonly used benchmark datasets and their combinations:

LETTER [136]: has a total of 20000 samples from 26 classes, where each class has around 769 samples with 16 features. To recast it for open set recognition, 10 distinct classes are randomly chosen as KKCs for training, while the remaining ones as UUCs.

PENDIGITS [137]: has a total of 10992 samples from 10 classes, where each class has around 1099 samples with 16 features. Similarly, 5 distinct classes are randomly chosen as KKCs and the remaining ones as UUCs.  

COIL20 [138]: has a total of 1440 gray images from 20 objects (72 images each object). Each image is down-sampled to 16 x 16, i.e., the feature dimension is 256. Following [91], we further reduce the dimension to 55 by principal component analysis (PCA) technique, remaining 95% of the samples’ information. 10 distinct objects are randomly chosen as KKCs, while the remaining ones as UUCs.  

YALEB [139]: The Extended Yale B (YALEB) dataset has a total of 2414 frontal-face images from 38 individuals. Each individuals has around 64 images. The images are cropped and normalized to 32 x 32. Following [91], we also reduce their feature dimension to 69 using PCA. Similar to COIL20, 10 distinct classes are randomly chosen as KKCs, while the remaining ones as UUCs.  

MNIST [140]: consists of 10 digit classes, where each class contains between 6313 and 7877 monochrome images with 28 x 28 feature dimension. Following [87], 6 distinct classes are randomly chosen as KKCs, while the remaining 4 classes as UUCs.

SVHN [141]: has ten digit classes, each containing between 9981 and 11379 color images with 32  32 feature dimension. Following [87], 6 distinct classes are randomly chosen as KKCs, while the remaining 4 classes as UUCs.  

CIFAR10 [142]: has a total of 6000 color images from 10 natural image classes. Each image has 32  32 feature dimension. Following [87], 6 distinct classes are randomly chosen as KKCs, while the remaining 4 classes as UUCs. To extend this dataset to larger openness, [87] further proposed the **CIFAR+10, CIFAR+50** datasets, which use 4 non-animal classes in CIFAR10 as KKCs, while 10 and 50 animal classes are respectively chosen from CIFAR1005 as UUCs.  

Tiny-Imagenet [143]: has a total of 200 classes with 500 images each class for training and 50 for testing, which is drawn from the Imagenet ILSVRC 2012 dataset [144] and down-sampled to 3232. Following [87], 20 distinct classes are randomly chosen as KKCs, while the remaining 180 classes as UUCs.

### 5.2 Evaluation Criteria

### 5.3 Experiments

### 6 FUTURE RESEARCH DIRECTIONS
