---
layout: post
bigtitle:  "CLASSIFICATION-BASED ANOMALY DETECTION FOR GENERAL DATA"
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
# CLASSIFICATION-BASED ANOMALY DETECTION FOR GENERAL DATA

ICLR 2020 [paper](https://openreview.net/forum?id=H1lK_lBtvS)

Liron Bergman  
Yedid Hoshen

School of Computer Science and Engineering
The Hebrew University of Jerusalem, Israel

## ABSTRACT

Anomaly detection, finding patterns that substantially deviate from those seen previously, is one of the fundamental problems of artificial intelligence. Recently, classification-based methods were shown to achieve superior results on this task. In this work, we present a unifying view and propose an open-set method, GOAD, to relax current generalization assumptions. Furthermore, we extend the applicability of transformation-based methods to non-image data using random affine transformations. Our method is shown to obtain state-of-the-art accuracy and is applicable to broad data types. The strong performance of our method is extensively validated on multiple datasets from different domains.

## 1 INTRODUCTION

Detecting anomalies in perceived data is a key ability for humans and for artificial intelligence. Humans often detect anomalies to give early indications of danger or to discover unique opportunities. Anomaly detection systems are being used by artificial intelligence to discover credit card fraud, for detecting cyber intrusion, alert predictive maintenance of industrial equipment and for discovering attractive stock market opportunities. The typical anomaly detection setting is a one class classification task, where the objective is to classify data as normal or anomalous. The importance of the task stems from being able to raise an alarm when detecting a different pattern from those seen in the past, therefore triggering further inspection. This is fundamentally different from supervised learning tasks, in which examples of all data classes are observed.

There are different possible scenarios for anomaly detection methods. In supervised anomaly detection, we are given training examples of normal and anomalous patterns. This scenario can be quite well specified, however obtaining such supervision may not be possible. For example in cyber security settings, we will not have supervised examples of new, unknown computer viruses making supervised training difficult. On the other extreme, fully unsupervised anomaly detection, obtains a stream of data containing normal and anomalous patterns and attempts to detect the anomalous data. In this work we deal with the semi-supervised scenario. In this setting, we have a training set of normal examples (which contains no anomalies). After training the anomaly detector, we detect anomalies in the test data, containing both normal and anomalous examples. This supervision is easy to obtain in many practical settings and is less difficult than the fully-unsupervised case.

Many anomaly detection methods have been proposed over the last few decades. They can be broadly classified into reconstruction and statistically based methods.Recently, deep learning methods based on classification have achieved superior results. Most semi-supervised classificationbased methods attempt to solve anomaly detection directly, despite only having normal training data. One example is: Deep-SVDD (Ruff et al., 2018) - one-class classification using a learned deep space. Another type of classification-based methods is self-supervised i.e. methods that solve one or more classification-based auxiliary tasks on the normal training data, and this is shown to be useful for solving anomaly detection, the task of interest e.g. (Golan & El-Yaniv, 2018). Self-supervised classification-based methods have been proposed with the object of image anomaly detection, but we show that by generalizing the class of transformations they can apply to all data types.

In this paper, we introduce a novel technique, GOAD, for anomaly detection which unifies current state-of-the-art methods that use normal training data only and are based on classification. Our method first transforms the data into M subspaces, and learns a feature space such that inter-class separation is larger than intra-class separation. For the learned features, the distance from the cluster center is correlated with the likelihood of anomaly. We use this criterion to determine if a new data point is normal or anomalous. We also generalize the class of transformation functions to include affine transformation which allows our method to generalize to non-image data. This is significant as tabular data is probably the most important for applications of anomaly detection. Our method is evaluated on anomaly detection on image and tabular datasets (cyber security and medical) and is shown to significantly improve over the state-of-the-art.

### 1.1 PREVIOUS WORKS

Anomaly detection methods can be generally divided into the following categories:

Reconstruction Methods: Some of the most common anomaly detection methods are reconstructionbased. The general idea behind such methods is that every normal sample should be reconstructed accurately using a limited set of basis functions, whereas anomalous data should suffer from larger reconstruction costs. The choice of features, basis and loss functions differentiates between the different methods. Some of the earliest methods use: nearest neighbors (Eskin et al., 2002), low-rank PCA (Jolliffe, 2011; Candes et al., 2011) or K-means (Hartigan &Wong, 1979) as the reconstruction basis. Most recently, neural networks were used (Sakurada & Yairi, 2014; Xia et al., 2015) for learning deep basis functions for reconstruction. Another set of recent methods (Schlegl et al., 2017; Deecke et al., 2018) use GANs to learn a reconstruction basis function. GANs suffer from mode-collapse and are difficult to invert, which limits the performance of such methods.

Distributional Methods: Another set of commonly used methods are distribution-based. The main theme in such methods is to model the distribution of normal data. The expectation is that anomalous test data will have low likelihood under the probabilistic model while normal data will have higher likelihoods. Methods differ in the features used to describe the data and the probabilistic model used to estimate the normal distribution. Some early methods used Gaussian or Gaussian mixture models. Such models will only work if the data under the selected feature space satisfies the probabilistic assumptions implicied by the model. Another set of methods used non-parametric density estimate methods such as kernel density estimate (Parzen, 1962). Recently, deep learning methods (autoencoders or variational autoencoders) were used to learn deep features which are sometimes easier to model than raw features (Yang et al., 2017). DAGMM introduced by Zong et al. (2018) learn the probabilistic model jointly with the deep features therefore shaping the features space to better conform with the probabilistic assumption.

Classification-Based Methods: Another paradigm for anomaly detection is separation between space regions containing normal data from all other regions. An example of such approach is One-Class SVM (Scholkopf et al., 2000), which trains a classifier to perform this separation. Learning a good feature space for performing such separation is performed both by the classic kernel methods as well as by the recent deep learning approach (Ruff et al., 2018). One of the main challenges in unsupervised (or semi-supervised) learning is providing an objective for learning features that are relevant to the task of interest. One method for learning good representations in a self-supervised way is by training a neural network to solve an auxiliary task for which obtaining data is free or at least very inexpensive. Auxiliary tasks for learning high-quality image features include: video frame prediction (Mathieu et al., 2016), image colorization (Zhang et al., 2016; Larsson et al., 2016), puzzle solving (Noroozi & Favaro, 2016) - predicting the correct order of random permuted image patches. Recently, Gidaris et al. (2018) used a set of image processing transformations (rotation by 0; 90; 180; 270 degrees around the image axis, and predicted the true image orientation has been used to learn high-quality image features. Golan & El-Yaniv (2018), have used similar image-processing task prediction for detecting anomalies in images. This method has shown good performance on detecting images from anomalous classes. In this work, we overcome some of the limitations of previous classification-based methods and extend their applicability of self-supervised methods to general data types. We also show that our method is more robust to adversarial attacks.

## 2 CLASSIFICATION-BASED ANOMALY DETECTION

Classification-based methods have dominated supervised anomaly detection.  
In this section we will analyse semi-supervised classification-based methods:

Let us assume all data lies in space $$R^L$$ (where $$L$$ is the data dimension).  
Normal data lie in subspace $$X \sub R^L$$.  
We assume that all anomalies lie outside X.  
To detect anomalies, we would therefore like to build a classifier $$C$$, such that $$C(x) = 1$$ if $$x \in X$$ and $$C(x) = 0$$ if $$x \in R^L / X$$.

One-class classification methods attempt to learn $$C$$ directly as $$P(x \in X)$$.  
Classical approaches have learned a classifier either in input space or in a kernel space.  
Recently, Deep-SVDD (Ruff et al., 2018) learned end-to-end to i) transform the data to an isotropic feature space $$f(x)$$ ii) fit the minimal hypersphere of radius $$R$$ and center $$c_0$$ around the features of the normal training data.  
Test data is classified as anomalous if the following normality score is positive: $$||f(x) - c_0||^2 - R^2$$.
Learning an effective feature space is not a simple task, as the trivial solution of $$f(x) = 0 \; \forall x$$ results in the smallest hypersphere, various tricks are used to avoid this possibility.

Geometric-transformation classification (GEOM), proposed by Golan & El-Yaniv (2018) first transforms the normal data subspace $$X$$ into $$M$$ subspaces $$X_1 .. X_M$$.  
This is done by transforming each image $$x \in X$$ using $$M$$ different geometric transformations (rotation, reflection, translation) into $$T(x,1)..T(x,M)$$.  
Although these transformations are image specific, we will later extend the class of transformations to all affine transformations making this applicable to non-image data. They set an auxiliary task of learning a classifier able to predict the transformation label m given transformed data point $$T(x,m)$$.  
As the training set consists of normal data only, each sample is $$x \in X$$ and the transformed sample is in $$\union _mX_m$$.  
The method attempts to estimate the following conditional probability:
