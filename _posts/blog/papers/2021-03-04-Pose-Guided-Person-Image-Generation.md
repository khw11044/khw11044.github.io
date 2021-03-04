---
layout: post
bigtitle:  "Pose Guided Person Image Genration"
subtitle:   "Pose Guided Person Image Genration"
categories:
    - blog
    - papers
tags:
    - FashionAI
    - Pose-Guided-Person-Image
comments: true
published: true
---

# Pose Guided Person Image Genration

Liqian Ma, Xu Jia2, Qianru Sun, Bernt Schiele, Tinne Tuytelaars, Luc Van Gool

28 Jan 2018

## Abstract

본 논문은 그 사람의 이미지와 새로운 포즈를 기반으로 임의의 포즈로 사람 이미지를 합성할 수 있는 새로운 PG<sup>2</sup> (Pose Guided Person Generation Network)를 제안한다.

우리의 generation framework PG<sup>2</sup>는 포즈 정보를 명시적으로 활용하고 pose integration과 image refinement의 두 가지 주요 단계로 구성된다.

첫 번째 단계에서 condition image와 target pose는 U-Net과 유사한 네트워크로 공급되어 target pose를 가진 사람의 coarse(조잡한) image이지만 초기 이미지를 생성한다.

그런 다음 두 번째 단계는 U-Net 같은 generator를 적대적 방법(adversarial way)으로 훈련하여 초기 결과와 흐릿한 결과를 개선한다.

128x64 재식별 이미지와 256x256 패션 사진에 대한 광범위한 실험 결과는 우리 모델이 설득력 있는 디테일로 고품질 사람 이미지를 생성한다는 것을 보여준다.

## 1 Introduction

Generating realistic-looking images is of great value for many applications such as face editing, movie making and image retrieval based on synthesized images. Consequently, a wide range of methods have been proposed including Variational Autoencoders (VAE) [14], Generative Adversarial Networks (GANs) [6] and Autoregressive models (e.g., PixelRNN [30]).

Recently, GAN models have been particularly popular due to their principle ability to generate sharp images through adversarial training.

For example in [21, 5, 1], GANs are leveraged to generate faces and natural scene images and several methods are proposed to stabilize the training process and to improve the quality of generation.

From an application perspective, users typically have a particular intention in mind such as changing the background, an object’s category, its color or viewpoint.

The key idea of our approach is to guide the generation process explicitly by an appropriate representation of that intention to enable direct control over the generation process.

More specifically, we propose to generate an image by conditioning it on both a reference image and a specified pose.

With a reference image as condition, the model has sufficient information about the appearance of the desired object in advance.

The guidance given by the intended pose is both explicit and flexible. So in principle this approach can manipulate any object to an arbitrary pose.

In this work, we focus on transferring a person from a given pose to an intended pose.

There are many interesting applications derived from this task.

For example, in movie making, we can directly manipulate a character’s human body to a desired pose or, for human pose estimation, we can generate training data for rare but important poses.


Transferring a person from one pose to another is a challenging task. A few examples can be seen in Figure 1.

It is difficult for a complete end-to-end framework to do this because it has to generate both correct poses and detailed appearance simultaneously.

Therefore, we adopt a divide-and conquer strategy, dividing the problem into two stages which focus on learning global human body structure and appearance details respectively similar to [35, 9, 3, 19].


At stage-I, we explore different ways to model pose information. A variant of U-Net is employed to integrate the target pose with the person image.

It outputs a coarse generation result that captures the global structure of the human body in the target image.
A masked L1 loss is proposed to suppress the influence of background change between condition image and target image.
However, it would generate blurry result due to the use of L1.

At stage-II, a variant of Deep Convolutional GAN (DCGAN) model is used to further refine the initial generation result.

The model learns to fill in more appearance details via adversarial training
and generates sharper images. Different from the common use of GANs which directly learns to
generate an image from scratch, in this work we train a GAN to generate a difference map between
the initial generation result and the target person image. The training converges faster since it is an
easier task.

Besides, we add a masked L1 loss to regularize the training of the generator such that it
will not generate an image with many artifacts. Experiments on two dataset, a low-resolution person
re-identification dataset and a high-resolution fashion photo dataset, demonstrate the effectiveness of
the proposed method.

![Figure_1](/assets/img/Blog/papers/Pose-Guided-Person-Image-Generation/1.JPG)

Our contribution is three-fold.  

i) We propose a novel task of conditioning image generation on a reference image and an intended pose, whose purpose is to manipulate a person in an image to an arbitrary pose.  

ii) Several ways are explored to integrate pose information with a person image. A novel mask loss is proposed to encourage the model to focus on transferring the human body appearance instead of background information.

iii) To address the challenging task of pose transfer, we divide the problem into two stages, with the stage-I focusing on global structure of the human body and the stage-II on filling in appearance details based on the first stage result.

## 2 Related works

Recently there have been a lot of works on generative image modeling with deep learning techniques.

These works fall into two categories.
The first line of works follow an unsupervised setting.

One popular method under this setting is variational autoencoders proposed by Kingma and Welling [14]
and Rezende et al. [25], which apply a re-parameterization trick to maximize the lower bound of the
data likelihood.

Another branch of methods are autogressive models [28, 30, 29] which compute the product of conditional distributions of pixels in a pixel-by-pixel manner as the joint distribution of pixels in an image.

The most popular methods are generative adversarial networks (GAN) [6], which simultaneously learn a generator to generate samples and a discriminator to discriminate generated samples from real ones.

Many works show that GANs can generate sharp images because of using the adversarial loss instead of L1 loss.

In this work, we also use the adversarial loss in our framework in order to generate high-frequency details in images.

![Figure_2](/assets/img/Blog/papers/Pose-Guided-Person-Image-Generation/2.JPG)

The second group of works generate images conditioned on either category or attribute labels, texts or
images.

Yan et al. [32] proposed a Conditional Variational Autoencoder (CVAE) to achieve attribute conditioned image generation.

Mirza and Osindero [18] proposed to condition both generator and discriminator of GAN on side information to perform category conditioned image generation.

Lassner et al. [15] generated full-body people in clothing, by conditioning on the fine-grained body part segments.

Reed et al. proposed to generate bird image conditioned on text descriptions by adding textual information to both generator and discriminator [24] and further explored the use of additional location, keypoints or segmentation information to generate images [22, 23].

With only these visual cues as condition and in contrast to our explicit condition on the intended pose, the control exerted over the image generation process is still abstract.

Several works further conditioned image generation not only on labels and texts but also on images.

Researchers [34, 33, 11, 8] addressed the task of face image generation conditioned on a reference image and a specific face viewpoint.

Chen et al. [4] tackled the unseen view inference as a tensor completion problem, and use latent factors to impute the pose in unseen views.

Zhao et al. [36] explored generating multi-view cloth images from only a single view input, which is most similar to our task.

However, a wide range of poses is consistent with any given viewpoint making the conditioning less expressive than in our work.

In this work, we make use of pose information in a more explicit and flexible way, that is, using poses in the format of keypoints to model diverse human body appearance.

It should be noted that instead of doing expensive pose annotation, we use a state-of-the-art pose estimation approach to obtain the desired human body keypoints.

## 3 Method

Our task is to simultaneously transfer the appearance of a person from a given pose to a desired pose and keep important appearance details of the identity.

As it is challenging to implement this as an end to-end model, we propose a two-stage approach to address this task, with each stage focusing on one aspect.

For the first stage we propose and analyze several model variants and for the second stage we use a variant of a conditional DCGAN to fill in more appearance details.

The overall framework of the proposed Pose Guided Person Generation Network (PG<sup>2</sup>) is shown in Figure 2.

### 3.1 Stage-I: Pose integration

At stage-I, we integrate a conditioning person image _I<sub>A</sub>_ with a target pose _P<sub>B</sub>_ to generate a coarse result $\hat{I}_B$ that captures the global structure of the human body in the target image $I_B$.

**Pose embedding**. To avoid expensive annotation of poses, we apply a state-of-the-art pose estimator [2] to obtain approximate human body poses.
The pose estimator generates the coordinates of 18 keypoints.

Using those directly as input to our model would require the model to learn to map each keypoint to a position on the human body.

Therefore, we encode pose $P_B$ as 18 heatmaps. Each heatmap is filled with 1 in a radius of 4 pixels around the corresponding keypoints and 0 elsewhere (see Figure 3, target pose).

We concatenate $I_A$ and $P_B$ as input to our model. In this way, we can directly use convolutional layers to integrate the two kinds of information.

**Generator G1**. As generator at stage I, we adopt a U-Net-like architecture [20], i.e., convolutional autoencoder with skip connections as is shown in Figure 2.

Specifically, we first use several stacked convolutional layers to integrate $I_A$ and $P_B$ from small local neighborhoods to larger ones so that appearance information can be integrated and transferred to neighboring body parts.

Then, a fully connected layer is used such that information between distant body parts can also exchange
information.

After that, the decoder is composed of a set of stacked convolutional layers which are symmetric to the encoder to generate an image.

The result of the first stage is denoted as $\hat{I}$<sub>B1</sub>. In the U-Net, skip connections between encoder and decoder help propagate image information directly from input to output.

In addition, we find that using residual blocks as basic component improves the generation performance.

In particular we propose to simplify the original residual block [7] and have only two consecutive conv-relu inside a residual block.

**Pose mask loss**. To compare the generation $\hat{I}$<sub>B1</sub> with the target image $I_B$, we adopt L1 distance as the generation loss of stage-I.

However, since we only have a condition image and a target pose as input, it is difficult for the model to generate what the background would look like if the target image has a different background from the condition image.

Thus, in order to alleviate the influence of background changes, we add another term that adds a pose mask $M_B$ to the L1 loss such that the human body is given more weight than the background.
The formulation of pose mask loss is given in Eq. 1 with $\odot$ denoting the pixels-wise multiplication:

$$L_{G1}=||G1(I_A,P_B) - I_B) \odot (1+M_B)||_1, \qquad \qquad(1)$$

![Figure_3](/assets/img/Blog/papers/Pose-Guided-Person-Image-Generation/3.JPG)

The pose mask $M_B$ is set to 1 for foreground and 0 for background and is computed by connecting human body parts and applying a set of morphological operations such that it is able to approximately cover the whole human body in the target image, see the example in Figure 3.

The output of $G_1$ is blurry because the L1 loss encourages the result to be an average of all possible cases [10].

However, $G_1$ does capture the global structural information specified by the target pose, as shown in Figure 2, as well as other low-frequency information such as the color of clothes.

Details of body appearance, i.e. the high-frequency information, will be refined at the second stage through adversarial training.

### 3.2 Stage-II: Image refinement

Since the model at the first stage has already synthesized an image which is coarse but close to the target image in pose and basic color, at the second stage, we would like the model to focus on generating more details by correcting what is wrong or missing in the initial result.

We use a variant of conditional DCGAN [21] as our base model and condition it on the stage-I generation result.

**Generator G2**. Considering that the initial result and the target image are already structurally similar, we propose that the generator G2 at the second stage aims to generate an appearance difference map that brings the initial result closer to the target image.

The difference map is computed using a U-Net similar to the first stage but with the initial result $\hat{I}_{B1}$ and condition image $I_A$ as input instead.

The difference lies in that the fully-connected layer is removed from the U-Net.
This helps to preserve more details from the input because a fully-connected layer compresses a lot of information contained in the input.

The use of difference maps speeds up the convergence of model training since the model focuses on learning the missing appearance details instead of synthesizing the target image from scratch.

In particular, the training already starts from a reasonable result. The overall architecture of G2 can be seen in Figure 2.

**Discriminator D**. In traditional GANs, the discriminator distinguishes between real groundtruth images and fake generated images (which is generated from random noise).

However, in our conditional network, G2 takes the condition image $I_A$ instead of a random noise as input.
Therefore, real images are the ones which not only are natural but also satisfy a specific requirement.
Otherwise, G2 will be mislead to directly output $I_A$ which is natural by itself instead of refining the coarse result of the first stage $\hat{I}_{B1}$.

To address this issue, we pair the G2 output with the condition image to make the discriminator D to recognize the pairs’ fakery, i.e., ($\hat{I}_{B2}$, $I_A$) vs ($I_B$, $I_A$). This is diagrammed in
Figure 2. The pairwise input encourages D to learn the distinction between $I_{B2}$ and $I_B$ instead of only the distinction between synthesized and natural images.

Another difference from traditional GANs is that noise is not necessary anymore since the generator is conditioned on an image $I_A$, which is similar to [17]. Therefore, we have the following loss function for the discriminator D and the generator G2 respectively,

$$L^D_{adv} = L_{bce}(D(I_A,I_B),1) + L_{bce}(D(I_A,G2(I_A, \hat{I}_{B1})),0),\qquad \qquad (2)$$
$$L^G_{adv} = L_{bce}(D(I_A,G2(I_A,\hat{I}_{B1})),1), \qquad \qquad (3)$$

where $\lambda$ is the weight of L1 loss. It controls how close the generation looks like the target image at low frequencies. When $\lambda$ is small, the adversarial loss dominates the training and it is more likely to generate artifacts; when $\lambda$ is big, the the generator with a basic L1 loss dominates the training, making the whole model generate blurry results<sup>2</sup>.

In the training process of our DCGAN, we alternatively optimize discriminator D and generator G2.

As shown in the left part of Figure 2, generator G2 takes the first stage result and the condition image as input and aims to refine the image to confuse the discriminator. The discriminator learns to classify the pair of condition image and the generated image as fake while classifying the pair including the target image as real.

### 3.3 Network architecture

We summarize the network architecture of the proposed model PG2.  
At stage-I, the encoder of G1 consists of $N$ residual blocks and one fully-connected layer, where $N$ depends on the size of input.  
Each residual block consists of two convolution layers with stride=1 followed by one sub-sampling convolution layer with stride=2 except the last block.   
At stage-II, the encoder of G2 has a fully convolutional architecture including N-2 convolution blocks.  
Each block consists of two convolution layers with stride=1 and one sub-sampling convolution layer with stride=2.  
Decoders in both G1 and G2 are symmetric to corresponding encoders. Besides, there are shortcut connections between decoders and encoders, which can be seen in Figure 2.  
In G1 and G2, no batch normalization or dropout are applied.  
All convolution layers consist of 3x3 filters and the number of filters are increased linearly with each block.  
We apply rectified linear unit (ReLU) to each layer except the fully connected layer and the output convolution layer.  
For the discriminator, we adopt the same network architecture as DCGAN [21] except the size of the input convolution layer due to different image resolutions.

## 4 Experiments
