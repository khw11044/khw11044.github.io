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

Novelty detection is commonly referred to as the discrimination of observations that do not conform to a learned model of regularity. Despite its importance in different application settings, designing a novelty detector is utterly complex due to the unpredictable nature of novelties and its inaccessibility during the training procedure, factors which expose the unsupervised nature of the problem. In our proposal, we design a general framework where we equip a deep autoencoder with a parametric density estimator that learns the probability distribution underlying its latent rep- resentations through an autoregressive procedure. We show that a maximum likelihood objective, optimized in conjunction with the reconstruction of normal samples, effectively acts as a regularizer for the task at hand, by minimizing the differential entropy of the distribution spanned by latent vectors. In addition to providing a very general formulation, extensive experiments of our model on publicly avail- able datasets deliver on-par or superior performances if compared to state-of-the-art methods in one-class and video anomaly detection settings. Differently from prior works, our proposal does not make any assumption about the nature of the novelties, making our work readily applicable to diverse contexts.

## 1. Introduction

Novelty detection is defined as the identification of samples which exhibit significantly different traits with respect to an underlying model of regularity, built from a collection of normal samples. The awareness of an autonomous system to recognize unknown events enables applications in several domains, ranging from video surveillance [7, 11], to defect detection [19] to medical imaging [35]. Moreover, the surprise inducted by unseen events is emerging as a crucial aspect in reinforcement learning settings, as an enabling factor in curiosity-driven exploration [31].

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
