---
layout: post
bigtitle:  "A BASELINE FOR DETECTING MISCLASSIFIED AND OUT-OF-DISTRIBUTION EXAMPLES IN NEURAL NETWORKS"
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
    - _posts/blog/githubpage/2021-02-01-tacotron1_expain.md
    - _posts/blog/githubpage/2021-02-01-tacotron1_summary.md
    - _posts/blog/githubpage/2020-02-02-seq2seq-with-attention.md
---
# A BASELINE FOR DETECTING MISCLASSIFIED AND OUT-OF-DISTRIBUTION EXAMPLES IN NEURAL NETWORKS

ICLR 2017 [paper](https://arxiv.org/pdf/1610.02136.pdf)


## ABSTRACT

We consider the two related problems of detecting if an example is misclassified or out-of-distribution.  
We present a simple baseline that utilizes probabilities from softmax distributions.  
Correctly classified examples tend to have greater maximum softmax probabilities than erroneously classified and out-of-distribution examples, allowing for their detection.  
We assess performance by defining several tasks in computer vision, natural language processing, and automatic speech recognition, showing the effectiveness of this baseline across all.  
We then show the baseline can sometimes be surpassed, demonstrating the room for future research on these underexplored detection tasks.

## INTRODUCTION

When machine learning classifiers are employed in real-world tasks, they tend to fail when the training and test distributions differ.  
Worse, these classifiers often fail silently by providing high-confidence predictions while being woefully incorrect (Goodfellow et al., 2015; Amodei et al., 2016).  
Classifiers failing to indicate when they are likely mistaken can limit their adoption or cause serious accidents.  
For example, a medical diagnosis model may consistently classify with high confidence, even while it should flag difficult examples for human intervention.  
The resulting unflagged, erroneous diagnoses could blockade future machine learning technologies in medicine.  
More generally and importantly, estimating when a model is in error is of great concern to AI Safety (Amodei et al., 2016).

These high-confidence predictions are frequently produced by softmaxes because softmax probabilities are computed with the fast-growing exponential function.  
Thus minor additions to the softmax inputs, i.e. the logits, can lead to substantial changes in the output distribution.   Since the softmax function is a smooth approximation of an indicator function, it is uncommon to see a uniform distribution outputted for out-of-distribution examples.  
Indeed, random Gaussian noise fed into an MNIST image classifier gives a “prediction confidence” or predicted class probability of 91%, as we show later.  
Throughout our experiments we establish that the prediction probability from a softmax distribution has a poor direct correspondence to confidence.  
This is consistent with a great deal of anecdotal evidence from researchers (Nguyen & O’Connor, 2015; Yu et al., 2010; Provost et al., 1998; Nguyen et al., 2015).

However, in this work we also show the prediction probability of incorrect and out-of-distribution examples tends to be lower than the prediction probability for correct examples.  
Therefore, capturing prediction probability statistics about correct or in-sample examples is often sufficient for detecting whether an example is in error or abnormal, even though the prediction probability viewed in isolation can be misleading.

These prediction probabilities form our detection baseline, and we demonstrate its efficacy through various computer vision, natural language processing, and automatic speech recognition tasks.  
While these prediction probabilities create a consistently useful baseline, at times they are less effective, revealing room for improvement.  
To give ideas for future detection research, we contribute one method which outperforms the baseline on some (but not all) tasks.
This new method evaluates the quality of a neural network’s input reconstruction to determine if an example is abnormal.

In addition to the baseline methods, another contribution of this work is the designation of standard tasks and evaluation metrics for assessing the automatic detection of errors and out-of-distribution examples.  
We use a large number of well-studied tasks across three research areas, using standard neural network architectures that perform well on them.  
For out-of-distribution detection, we provide ways to supply the out-of-distribution examples at test time like using images from different datasets and realistically distorting inputs.  
We hope that other researchers will pursue these tasks in future work and surpass the performance of our baselines.

In summary, while softmax classifier probabilities are not directly useful as confidence estimates, estimating model confidence is not as bleak as previously believed.  
Simple statistics derived from softmax distributions provide a surprisingly effective way to determine whether an example is misclassified or from a different distribution from the training data, as demonstrated by our experimental results spanning computer vision, natural language processing, and speech recognition tasks.  
This creates a strong baseline for detecting errors and out-of-distribution examples which we hope future research surpasses.

## 2 PROBLEM FORMULATION AND EVALUATION
