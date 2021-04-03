---
layout: post
bigtitle:  "Latent Space Autoregression for Novelty Detection"
subtitle:   "anommaly detection"
categories:
    - blog
    - papers
    - paper-etc
tags:
    - anommalydetection
comments: true
published: true
related_posts:
---
# Latent Space Autoregression for Novelty Detection

CVPR 2019 [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Abati_Latent_Space_Autoregression_for_Novelty_Detection_CVPR_2019_paper.pdf)

Davide Abati  
Angelo Porrello  
Simone Calderara  
Rita Cucchiara

University of Modena and Reggio Emilia

## ABSTRACT

Novelty detection is commonly referred to as the discrimination of observations that do not conform to a learned model of regularity.  
Despite its importance in different application settings, designing a novelty detector is utterly complex due to the unpredictable nature of novelties and its inaccessibility during the training procedure, factors which expose the unsupervised nature of the problem.  
In our proposal, we design a general framework where we equip a deep autoencoder with a parametric density estimator that learns the probability distribution underlying its latent representations through an autoregressive procedure.

> Novelty detection은 일반적으로 학습된 regularity 모델에 부합하지 않는 observations의 discrimination이라고 한다.  
다른 application 설정에서 그것의 중요성에도 불구하고, novelty detector 설계는 novelties의 unpredictable 특성과 training 절차 동안 inaccessibility으로 인해 완전히 complex하며, 이는 문제의 unsupervised 특성을 드러내는 요인이다.  
우리의 제안에서, 우리는 autoregressive procedure를 통해 latent representations에 기초하는 probability distribution를 학습하는 parametric density estimator를 deep autoencoder에 장착하는 general framework를 설계한다.  

We show that a maximum likelihood objective, optimized in conjunction with the reconstruction of normal samples, effectively acts as a regularizer for the task at hand, by minimizing the differential entropy of the distribution spanned by latent vectors.  
In addition to providing a very general formulation, extensive experiments of our model on publicly available datasets deliver on-par or superior performances if compared to state-of-the-art methods in one-class and video anomaly detection settings.  
Differently from prior works, our proposal does not make any assumption about the nature of the novelties, making our work readily applicable to diverse contexts.

> 우리는 정상 샘플의 reconstruction과 함께 최적화된 maximum likelihood 목표가 latent vectors에 의해 확장된 distribution의 differential entropy를 최소화함으로써 당면한 task에 대한 정규화 역할을 효과적으로 수행함을 보여준다.   
매우 일반적인 공식을 제공할 뿐만 아니라, 공개적으로 사용 가능한 datasets에 대한 모델의 광범위한 실험은 단일 클래스 및 비디오 이상 탐지 설정의 state-of-the-art methods과 비교할 때 동등하거나 우수한 성능을 제공한다.   
이전 연구와 달리, 우리의 제안은 novelties의 nature에 대한 어떠한 가정도 하지 않아 우리의 작업이 다양한 맥락에 쉽게 적용될 수 있게 한다.

## 1. Introduction

Novelty detection is defined as the identification of samples which exhibit significantly different traits with respect to an underlying model of regularity, built from a collection of normal samples.  
The awareness of an autonomous system to recognize unknown events enables applications in several domains, ranging from video surveillance [7, 11], to defect detection [19] to medical imaging [35].  
Moreover, the surprise inducted by unseen events is emerging as a crucial aspect in reinforcement learning settings, as an enabling factor in curiosity-driven exploration [31].

However, in this setting, the definition and labeling of novel examples are not possible. Accordingly, the literature agrees on approximating the ideal shape of the boundary separating normal and novel samples by modeling the intrinsic characteristics of the former. Therefore, prior works tackle such problem by following principles derived from the unsupervised learning paradigm [9, 34, 11, 23, 27]. Due to the lack of a supervision signal, the process of feature extraction and the rule for their normality assessment can only be guided by a proxy objective, assuming the latter will define an appropriate boundary for the application at hand.

According to cognitive psychology [4], novelty can be expressed either in terms of capabilities to remember an event or as a degree of surprisal [39] aroused by its observation. The latter is mathematically modeled in terms of low probability to occur under an expected model, or by lowering a variational free energy [15]. In this framework, prior models take advantage of either parametric [46] or nonparametric [13] density estimators. Differently, remembering an event implies the adoption of a memory represented either by a dictionary of normal prototypes - as in sparse coding approaches [9] - or by a low dimensional representation of the input space, as in the self-organizing maps [18] or, more recently, in deep autoencoders. Thus, in novelty detection, the remembering capability for a given sample is evaluated either by measuring reconstruction errors [11, 23] or by performing discriminative in-distribution tests [34].

Our proposal contributes to the field by merging remembering and surprisal aspects into a unique framework: we design a generative unsupervised model (i.e., an autoencoder, represented in Fig. 1i) that exploits end-to-end training in order to maximize remembering effectiveness for normal samples whilst minimizing the surprisal of their latent representation. This latter point is enabled by the maximization of the likelihood of latent representations through an autoregressive density estimator, which is performed in conjunction with the reconstruction error minimization. We show that, by optimizing both terms jointly, the model implicitly seeks for minimum entropy representations maintaining its remembering/reconstructive power. While entropy minimization approaches have been adopted in deep neural compression [3], to our knowledge this is the first proposal tailored for novelty detection. In memory terms, our procedure resembles the concept of prototyping the normality using as few templates as possible. Moreover, evaluating the output of the estimator enables the assessment of the surprisal aroused by a given sample.

## 2. Related work

**Reconstruction-based methods.** On the one hand, many works lean toward learning a parametric projection and reconstruction of normal data, assuming outliers will yield higher residuals. Traditional sparse-coding algorithms [45, 9, 24] adhere to such framework, and represent normal patterns as a linear combination of a few basis components, under the hypotheses that novel examples would exhibit a non-sparse representation in the learned subspace. In recent works, the projection step is typically drawn from deep autoencoders [11]. In [27] the authors recover sparse coding principles by imposing a sparsity regularization over the learned representations, while a recurrent neural network enforces their smoothness along the time dimension. In [34], instead, the authors take advantage of an adversarial framework in which a discriminator network is employed as the actual novelty detector, spotting anomalies by performing a discrete in-distribution test. Oppositely, future frame prediction [23] maximizes the expectation of the next frame exploiting its knowledge of the past ones; at test time, observed deviations against the predicted content advise for abnormality. Differently from the above-mentioned works, our proposal relies on modeling the prior distribution of latent representations. This choice is coherent with recent works from the density estimation community [38, 6]. However, to the best of our knowledge, our work is the first advocating for the importance of such a design choice for novelty detection.

**Probabilistic methods.** A complementary line of research investigates different strategies to approximate the density function of normal appearance and motion features. The primary issue raising in this field concerns how to estimate such densities in a high-dimensional and complex feature space. In this respect, prior works involve handcrafted features such as optical flow or trajectory analysis and, on top of that, employ both non-parametric [1] and parametric [5, 28, 22] estimators, as well as graphical modeling [16, 20]. Modern approaches rely on deep representations (e.g., captured by autoencoders), as in Gaussian classifiers [33] and Gaussian Mixtures [46]. In [13] the authors involve a Kernel Density Estimator (KDE) modeling activations from an auxiliary object detection network. A recent research trend considers training Generative Adversarial Networks (GANs) on normal samples. However, as such models approximate an implicit density function, they can be queried for new samples

but not for likelihood values. Therefore, GAN-based models employ different heuristics for the evaluation of novelty. For instance, in [35] a guided latent space search is exploited to infer it, whereas [32] directly queries the discriminator for a normality score.

## 3. Proposed model

Maximizing the probability of latent representations is analogous to lowering the surprisal of the model for a normal configuration, defined as the negative log-density of a latent variable instance [39]. Conversely, remembering capabilities can be evaluated by the reconstruction accuracy of a given sample under its latent representation.

We model the aforementioned aspects in a latent variable model setting, where the density function of training samples $$p(\mathbf{x})$$ is modeled through an auxiliary random variable z, describing the set of causal factors underlying all observations. By factorizing

$$p(\mathbf{x}) = \int p(\mathbf{x}|\mathbf{z})p(\mathbf{z})d\mathbf{z} \qquad \qquad (1) $$

,where $$p(\mathbf{x}|\mathbf{z})$$ is the conditional likelihood of the observation given a latent representation $$\mathbf{z}$$ with prior distribution $$p(\mathbf{z})$$, we can explicit both the memory and surprisal contribution to novelty.

We approximate the marginalization by means of an inference model responsible for the identification of latent space vector for which the contribution of $$p(\mathbf{x}|\mathbf{z})$$ is maximal. Formally, we employ a deep autoencoder, in which the reconstruction error plays the role of the negative logarithm of $$p(\mathbf{x}|\mathbf{z})$$, under the hypothesis that$$p(\mathbf{x}|\mathbf{z}) = \mathcal{N}(x|\tilde{x}, I)$$ where $$\tilde{x}$$ denotes the output reconstruction. Additionally, surprisal is injected in the process by equipping the autoencoder with an auxiliary deep parametric estimator learning the prior distribution $$p(\mathbf{z})$$ of latent vectors, and training it by means of Maximum Likelihood Estimation (MLE). Our architecture is therefore composed of three building blocks (Fig. 1i): an encoder $$f(x; \theta_f )$$, a decoder $$g(z; \theta_g)$$ and a probabilistic model $$h(z; \theta_h)$$:

$$f(x; \theta_f) : \mathbb{R^m} → \mathbb{R^d}, g(z; \theta_{g}) : \mathbb{d} → \mathbb{m}, h(z; \theta_h): \mathbb{R^d} → [0,1].  $$

The encoder processes input $$x$$ and maps it into a compressed representation $$z = f(x; \theta_{f} )$$, whereas the decoder provides a reconstructed version of the input $$\tilde{x} = g(z; \theta_{g})$$. The probabilistic model $$h(z; \theta_h)$$ estimates the density in $$\mathbf{z}$$ via an autoregressive process, allowing to avoid the adoption of a specific family of distributions (i.e., Gaussian), potentially unrewarding for the task at hand. On this latter point, please refer to supplementary materials for comparison w.r.t. variational autoencoders [17].

With such modules, at test time, we can assess the two sources of novelty: elements whose observation is poorly explained by the causal factors inducted by normal samples (i.e., high reconstruction error); elements exhibiting good reconstructions whilst showing surprising underlying
representations under the learned prior.

![Fig1](/assets/img/Blog/papers/LatentSpace/Fig1.JPG)

**Autoregressive density estimation.** Autoregressive models provide a general formulation for tasks involving sequential predictions, in which each output depends on previous observations [25, 29]. We adopt such a technique to factorize a joint distribution, thus avoiding to define its landscape a priori [21, 40]. Formally, $$p(\mathbf{z})$$ is factorized as

$$p(\mathbf{z}) = \prod_{i=1}^d p(z_i|\mathbf{z}_{<i}) \qquad \qquad (3) $$

so that estimating $$p(\mathbf{z})$$ reduces to the estimation of each single Conditional Probability Density (CPD) expressed as $$p(z_i|\mathbf{z}_{<i})$$, where the symbol $$<$$ implies an order over random variables. Some prior models obey handcrafted orderings [43, 42], whereas others rely on order agnostic training [41, 10]. Nevertheless, it is still not clear how to estimate the proper order for a given set of variables. In our model, this issue is directly tackled by the optimization. Indeed, since we perform auto-regression on learned latent representations, the MLE objective encourages the autoencoder to impose over them a pre-defined causal structure. Empirical evidence of this phenomenon is given in the supplementary material.

From a technical perspective, the estimator $$h(z; theta_{h})$$ outputs parameters for d distributions $$p(z_i|\mathbf{z}_{<i})$$. In our implementation, each CPD is modeled as a multinomial over B=100 quantization bins. To ensure a conditional estimate of each

underlying density, we design proper layers guaranteeing that the CPD of each symbol $$z_i$$ is computed from inputs $$\{z_1, . . . , z_{i−1}\}$$ only.

**Objective and connection with differential entropy.**
The three components f, g and h are jointly trained to minimize $$\mathcal{L} ≡ \mathcal{L}(\theta_f , \theta_g, \theta_h)$$ as follows:

![4](/assets/img/Blog/papers/LatentSpace/4.JPG)
,where $$\lambda$$ is a hyper-parameter controlling the weight of the $$\mathcal{L}_{LLK}$$ term. It is worth noting that it is possible to express the log-likelihood term as

![5](/assets/img/Blog/papers/LatentSpace/5.JPG)

where $$p^*(\mathbf{z}; \theta_f )$$ denotes the true distribution of the codes produced by the encoder, and is therefore parametrized by $$\theta_{f}$$. This reformulation of the MLE objective yields meaningful insights about the entities involved in the optimization. On the one hand, the Kullback-Leibler divergence ensures that the information gap between our parametric model h and the true distribution $$p^*$$ is small. On the other hand, this framework leads to the minimization of the differential entropy of the distribution underlying the codes produced by the encoder $$f$$.  
Such constraint constitutes a crucial point when learning normality.  
Intuitively, if we think about the encoder as a source emitting symbols (namely, the latent representations), its desired behavior, when modeling normal aspects in the data, should converge to a ‘boring’ process characterized by an intrinsic low entropy, since surprising and novel events are unlikely to arise during the training phase.  
Accordingly, among all the possible settings of the hidden representations, the objective begs the
encoder to exhibit a low differential entropy, leading to the extraction of features that are easily predictable, therefore common and recurrent within the training set.  
This kind of features is indeed the most useful to distinguish novel samples from the normal ones, making our proposal a suitable regularizer in the anomaly detection setting.

We report empirical evidence of the decreasing differential entropy in Fig. 2, that compares the behavior of the same model under different regularization strategies.

### 3.1. Architectural Components

**Autoencoder blocks.** Encoder and decoder are respectively composed by downsampling and upsampling residual blocks depicted in Fig. 1ii. The encoder ends with fully connected (FC) layers. When dealing with video inputs, we employ causal 3D convolutions [2] within the encoder (i.e., only accessing information from previous timesteps). Moreover, at the end of the encoder, we employ a temporally-shared full connection (TFC, namely a linear projection sharing parameters across the time axis on the input feature maps) resulting in a temporal series of feature vectors. This way, the encoding procedure does not shuffle information across time-steps, ensuring temporal ordering.

**Autoregressive layers.** To guarantee the autoregressive nature of each output CPD, we need to ensure proper connectivity patterns in each layer of the estimator h.
Moreover, since latent representations exhibit different shapes depending on the input nature (image or video), we propose two different solutions.
When dealing with images, the encoder provides feature vectors with dimensionality $$d$$. The autoregressive estimator is composed by stacking multiple Masked Fully Connections (MFC, Fig. 3-(a)). Formally, it computes output feature map $$\mathrm{o} \in \mathbb{R}^{d\times co}$$ (where _co_ is the number of output channels) given the input $$\mathrm{h} \in \mathbb{R}^{d\times ci}$$ (assuming $$ci = 1$$ at the input layer). The connection between the input element $$h^k_i$$ in position $$i$$, channel $$k$$ and the output element $$o^l_j$$ is parametrized by

![6](/assets/img/Blog/papers/LatentSpace/6.JPG)

Type A forces a strict dependence on previous elements (and is employed only as the first estimator layer), whereas type B masks only succeeding elements. Assuming each CPD modeled as a multinomial, the output of the last autoregressive layer (in $$\mathbb{R}^{d×B}$$) provides probability estimates for the B bins that compose the space quantization.

On the other hand, the compressed representation of video clips has dimensionality $$t \times d$$, being t the number of temporal time-steps and d the length of the code. Accordingly, the estimation network is designed to capture two-dimensional patterns within observed elements of the code. However, naively plugging 2D convolutional layers would assume translation invariance on both axes of the input map, whereas, due to the way the compressed representation is built, this assumption is only correct along the temporal axis. To cope with this, we apply d different convolutional kernels along the code axis, allowing the observation of the whole feature vector in the previous time-step as well as a portion of the current one. Every convolution is free to stride along the time axis and captures temporal patterns. In such operation, named Masked Stacked Convolution (MSC, Fig. 3-(b)), the $$i$$-th convolution is equipped with a kernel $$w^{(i)} \in \mathbb{R}^{3 \times d}$$ kernel, that gets multiplied by the binary mask $$M^{(i)}$$, defined as

![7](/assets/img/Blog/papers/LatentSpace/7.JPG)

where j indexes the temporal axis and k the code axis.
Every single convolution yields a column vector, as a result of its stride along time. The set of column vectors resulting from the application of the d convolutions to the input tensor $$h \in \mathbb{R}^{t \times d \times ci}$$ are horizontally stacked to build the output tensor $$\mathrm{o} ∈ \mathbb{R}^{t \times d \times co}$$, as follows:

![8](/assets/img/Blog/papers/LatentSpace/8.JPG)

where $$||$$ represents the horizontal concatenation operation.

![Fig3](/assets/img/Blog/papers/LatentSpace/Fig3.JPG)

## 4. Experiments

We test our solution in three different settings: images, videos, and cognitive data. In all experiments the novelty assessment on the $$i$$-th example is carried out by summing the reconstruction term ($$REC_i$$) and the log-likelihood term ($$LLK_i$$) in Eq. 4 in a single novelty score $$NS_i$$:  
 $$NS_i=norm_S (REC_i) + norm_S (LLK_i.) \qquad \qquad (9)$$

Individual scores are normalized using a reference set of examples $$S$$ (different for every experiment),

$$norm_S(L_i) = \frac{L_i − max_{j \in S} L_j}{\max_{j \in S} L_j − \min{j \in S} L_j}. \; (10)$$

Further implementation details and architectural hyperparameters are in the supplementary material.

> Code to reproduce results in this section is released at [https://github.com/aimagelab/novelty-detection](https://github.com/aimagelab/novelty-detection).

### 4.1. One-class novelty detection on images

To assess the model’s performances in one class settings, we train it on each class of either MNIST or CIFAR-10 separately. In the test phase, we present the corresponding test set, which is composed of 10000 examples of all classes, and expect our model to assign a lower novelty score to images sharing the label with training samples. We use standard train/test splits, and isolate 10% of training samples for validation purposes, and employ it as the normalization set (S in Eq. 9) for the computation of the novelty score.
As for the baselines, we consider the following:

- standard methods such as OC-SVM [36] and Kernel Density Estimator (KDE), employed out of features
extracted by PCA-whitening;
- a denoising autoencoder (DAE) sharing the same architecture as our proposal, but defective of the density estimation module. The reconstruction error is employed as a measure of normality vs. novelty;
- a variational autoencoder (VAE) [17], also sharing the same architecture as our model, in which the Evidence Lower Bound (ELBO) is employed as the score;
- Pix-CNN [42], modeling the density by applying auto-regression directly in the image space;
- the GAN-based approach illustrated in [35].

We report the comparison in Tab. 1 in which performances are measured by the Area Under Receiver Operating Characteristic (AUROC), which is the standard metric for the task.  
As the table shows, our proposal outperforms all baselines in both settings.

![Table1](/assets/img/Blog/papers/LatentSpace/Table1.JPG)

Considering MNIST, most methods perform favorably. Notably, Pix-CNN fails in modeling distributions for all digits but one, possibly due to the complexity of modeling densities directly on pixel space and following a fixed autoregression order. Such poor test performances are registered despite good quality samples that we observed during training: indeed, the weak correlation between sample quality and test log-likelihood of the model has been motivated in [37]. Surprisingly, OC-SVM outperforms most deep learning based models in this setting.

On the contrary, CIFAR10 represents a much more significant challenge, as testified by the low performances of most models, possibly due to the poor image resolution and visual clutter between classes. Specifically, we observe that our proposal is the only model outperforming a simple KDE baseline; however, this finding should be put into perspective by considering the nature of non-parametric estimators. Indeed, non-parametric models are allowed to access the whole training set for the evaluation of each sample. Consequently, despite they benefit large sample sets in terms of density modeling, they lead into an unfeasible inference as the dataset grows in size.

The possible reasons behind the difference in performance w.r.t. DAE are twofold. Firstly, DAE can recognize novel samples solely based on the reconstruction error, hence relying on its memorization capabilities, whereas our proposal also considers the likelihood of their representations under the learned prior, thus exploiting surprisal as well. Secondly, by minimizing the differential entropy of the latent distribution, our proposal increases the discriminative capability of the reconstruction. Intuitively, this last statement can be motivated observing that novelty samples are forced to reside in a high probability region of the latent space, the latter bounded to solely capture unsurprising factors of variation arising from the training set. On the other hand, the gap w.r.t. VAE suggests that, for the task at hand, a more flexible autoregressive prior should be pre- ferred over the isotropic multivariate Gaussian. On this last point, VAE seeks representations whose average surprisal converges to a fixed and expected value (i.e., the differential entropy of its prior), whereas our solution minimizes such quantity within its MLE objective. This flexibility allows modulating the richness of the latent representation vs. the reconstructing capability of the model. On the contrary, in VAEs, the fixed prior acts as a blind regularizer, potentially leading to over-smooth representations; this aspect is also appreciable when sampling from the model as shown in the supplementary material.

![Fig4](/assets/img/Blog/papers/LatentSpace/Fig4.JPG)

Fig. 4 reports an ablation study questioning the loss functions aggregation presented in Eq. 9. The figure illustrates ROC curves under three different novelty scores: i) the log-likelihood term, ii) the reconstruction term, and iii) the proposed scheme that accounts for both. As highlighted in the picture, accounting for both memorization and surprisal aspects is advantageous in each dataset. Please refer to the supplementary material for additional evidence.

### 4.2. Video anomaly detection

In video surveillance contexts, novelty is often considered in terms of abnormal human behavior. Thus, we evaluate our proposal against state-of-the-art anomaly detection models. For this purpose, we considered two standard benchmarks in literature, namely UCSD Ped2 [8] and ShanghaiTech [27]. Despite the differences in the number of videos and their resolution, they both contain anomalies that typically arise in surveillance scenarios (e.g., vehicles in pedestrian walkways, pick-pocketing, brawling). For UCSD Ped, we preprocessed input clips of 16 frames to extract smaller patches (we refer to supplementary materials for details) and perturbed such inputs with random Gaussian noise with $$ \sigma = 0.025$$. We compute the novelty score of each input clip as the mean novelty score among all patches. Concerning ShanghaiTech, we removed the dependency on

the scenario by estimating the foreground for each frame of a clip with a standard MOG-based approach and removing the background. We fed the model with 16-frames clips, but ground-truth anomalies are labeled at frame level. In order to recover the novelty score of each frame, we compute the mean score of all clips in which it appears. We then merge the two terms of the loss function following the same strategy illustrated in Eq. 9, computing however normalization coefficients in a per-sequence basis, following the standard approach in the anomaly detection literature. The scores for each sequence are then concatenated to compute the overall AUROC of the model. Additionally, we envision localization strategies for both datasets. To this aim, for UCSD, we denote a patch exhibiting the highest novelty score in a frame as anomalous. Differently, in ShanghaiTech, we adopt a sliding-window approach [44]: as expected, when occluding the source of the anomaly with a rectangular patch, the novelty score drops significantly. Fig. 5 reports results in comparison with prior works, along with qualitative assessments regarding the novelty score and localization capabilities. Despite a more general formulation, our proposal scores on-par with the current state-ofthe- art solutions specifically designed for video applications and taking advantage of optical flow estimation and motion constraints. Indeed, in the absence of such hypotheses (FFP entry in Fig. 5), our method outperforms future frame prediction on UCSD Ped2.

![Fig5](/assets/img/Blog/papers/LatentSpace/Fig5.JPG)

### 4.3. Model Analysis

**CIFAR-10 with semantic features.**  
We investigate the behavior of our model in the presence of different assumptions regarding the expected nature of novel samples. We expect that, as the correctness of such assumptions increases, novelty detection performances will scale accordingly. Such a trait is particularly desirable for applications in which prior beliefs about novel examples can be envisioned. To this end, we leverage the CIFAR-10 benchmark described in Sec. 4.1 and change the type of information provided as input. Specifically, instead of raw images, we feed our model with semantic representations extracted by ResNet-50 [12], either pre-trained on Imagenet (i.e., assume semantic novelty) or CIFAR-10 itself (i.e., assume data-specific novelty). The two models achieved respectively 79.26 and 95.4 top-1 classification accuracies on the respective test sets. Even though this procedure is to be considered unfair in novelty detection, it serves as a sanity check delivering the upper-bound performances our model can achieve when applied to even better features. To deal with dense inputs, we employ a fully connected autoencoder and MFC layers within the estimation network.

Fig. 6-(a) illustrates the resulting ROC curves, where semantic descriptors improve AUROC w.r.t. raw image inputs (entry “Unsupervised”). Such results suggest that our model profitably takes advantage of the separation between normal and abnormal input representations and scales accordingly, even up to optimal performances for the task under consideration. Nevertheless, it is interesting to note how different degrees of supervision deliver significantly different performances. As expected, dataset-specific supervision increases the AUROC from 0.64 up to 0.99 (a perfect score). Surprisingly, semantic feature vectors trained on Imagenet (which contains all CIFAR classes) provide a much lower boost, yielding an AUROC of 0.72. Such result suggests that, even in the rare cases where the semantic of novelty can be known in advance, its contribution has a limited impact in modeling the normality, mostly because novelty can depend on other cues (e.g., low-level statistics).

**Autoregression via recurrent layers.**  
To measure the contribution of the proposed MFC and MSC layers described in Sec. 3, we test on CIFAR-10 and UCSD Ped2, alternative solutions for the autoregressive density estimator. Specifically, we investigate recurrent networks, as they represent the most natural alternative featuring autoregressive properties. We benchmark the proposed building blocks against an estimator composed of LSTM layers, which is designed to sequentially observe latent symbols z<i and output the CPD of zi as the hidden state of the last layer. We test MFC, MSC and LSTM in single-layer and multi-layer settings, and report all outcomes in Fig. 6-(b).

It emerges that, even though our solutions perform similarly to the recurrent baseline when employed in a shallow setting, they significantly take advantage of their depth when stacked in consecutive layers. MFC and MSC, indeed, employ disentangled parametrizations for each output CPD. This property is equivalent to the adoption of a specialized estimator network for each zi, thus increasing the proficiency in modeling the density of its designated CPD. On the contrary, LSTM networks embed all the history (i.e., the observed symbols) in their memory cells, but manipulate each input of the sequence through the same weight matrices. In such a regime, the recurrent module needs to learn parameters shared among symbols, losing specialization and eroding its modeling capabilities.

![Fig6](/assets/img/Blog/papers/LatentSpace/Fig6.JPG)

### 4.4. Novelty in cognitive temporal processes

As a potential application of our proposal, we investigate its capability in modeling human attentional behavior. To this end, we employ the DR(eye)VE dataset [30], introduced for the prediction of focus of attention in driving contexts. It features 74 driving videos where frame-wise fixation maps are provided, highlighting the region of the scene attended by the driver. In order to capture the dynamics of attentional patterns, we purposely discard the visual content of the scene and optimize our model on clips of fixation maps, randomly extracted from the training set. After training, we rely on the novelty score of each clip as a proxy for the uncommonness of an attentional pattern. Moreover, since the dataset features annotations of peculiar and unfrequent patterns (such as distractions, recording errors), we can measure the correlation of the captured novelty w.r.t. those. In terms of AUROC, our model scores 0.926, highlighting that novelty can arise from unexpected behaviors of the driver, such as distractions or other shifts in attention. Fig. 7 reports the different distribution of novelty scores for ordinary and peculiar events.

![Fig7](/assets/img/Blog/papers/LatentSpace/Fig7.JPG)

## 5. Conclusions

We propose a comprehensive framework for novelty detection. We formalize our model to capture the twofold nature of novelties, which concerns the incapability to remember unseen data and the surprisal aroused by the observation of their latent representations. From a technical perspective, both terms are modeled by a deep generative autoencoder, paired with an additional autoregressive density estimator learning the distribution of latent vectors by maximum likelihood principles. To this aim, we introduce two different masked layers suitable for image and video data. We show that the introduction of such an auxiliary module, operating in latent space, leads to the minimization of the encoder’s differential entropy, which proves to be a suitable regularizer for the task at hand. Experimental results show state-of-the-art performances in one-class and anomaly detection settings, fostering the flexibility of our framework for different tasks without making any data-related assumption.

### Acknowledgements.
We gratefully acknowledge Facebook Artificial Intelligence Research and Panasonic Silicon Valley Lab for the donation of GPUs used for this research.
