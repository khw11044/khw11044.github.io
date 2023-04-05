---
layout: post
bigtitle:  "Geometry-Driven Self-Supervised Method for 3D Human Pose Estimation"
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



# Geometry-Driven Self-Supervised Method for 3D Human Pose Estimation

Li, Yang, et al. "Geometry-driven self-supervised method for 3d human pose estimation." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 07. 2020.[paper](https://ojs.aaai.org/index.php/AAAI/article/view/6808)

* toc
{:toc}

## Abstract

The neural network based approach for 3D human pose estimation from monocular images has attracted growing interest.  
However, annotating 3D poses is a labor-intensive and expensive process.  
In this paper, we propose a novel self-supervised approach to avoid the need of manual annotations.  
Different from existing weakly/self-supervised methods that require extra unpaired 3D ground-truth data to alleviate the depth ambiguity problem, our method trains the network only relying on geometric knowledge without any additional 3D pose annotations.  
The proposed method follows the two-stage pipeline: 2D pose estimation and 2D-to-3D pose lifting.  
We design the transform re-projection loss that is an effective way to explore multi-view consistency for training the 2D to 3D lifting network.  
Besides, we adopt the confidences of 2D joints to integrate losses from different views to alleviate the influence of noises caused by the self-occlusion problem.  
Finally, we design a two-branch training architecture, which helps to preserve the scale information of re-projected 2D poses during training, resulting in accurate 3D pose predictions.  
We demonstrate the effectiveness of our method on two popular 3D human pose datasets, Human3.6M and MPIINF-3DHP.  
The results show that our method significantly outperforms recent weakly/self-supervised approaches.

> monocular images에서 3D human pose estimation을 위한 neural network based approach은 점점 더 많은 관심을 끌었다.  
그러나 annotating 3D poses은 노동 집약적이고 비용이 많이 드는 process이다.  
본 논문에서는 수동 annotations의 필요성을 피하기 위한 novel self-supervised approach를 제안한다.  
depth ambiguity problem를 완화하기 위해 extra unpaired 3D ground-truth data가 필요한 기존의 weakly/self-supervised methods와 달리, 우리의 방법은 추가적인 3D pose annotations 없이 geometric 지식에만 의존하여 network를 훈련시킨다.  
제안된 방법은 다음 two-stage pipeline를 따른다 : 2D pose estimation 그리고 2D-to-3D pose lifting.  
우리는 2D to 3D lifting network를 훈련하기 위한 multi-view consistency을 탐색하는 효과적인 방법인 'transform re-projection loss'를 설계한다.  
또한, 우리는 self-occlusion problem로 인한 noises의 영향을 완화하기 위해 different views에서 losses을 integrate하기 위해 2D joints의 confidences를 채택한다.  
마지막으로, 우리는 훈련 중에 re-projected 2D poses의 scale information을 보존하여 정확한 3D pose predictions을 가능하게 하는 two-branch training architecture를 설계한다.  
우리는 인기 있는 두 개의 3D human pose datasets인 Human3.6M 그리고 MPIINF-3DHP에서 우리의 방법의 효과를 보여준다.  
결과는 우리의 방법이 최근 weakly/self-supervised approaches을 크게 능가한다는 것을 보여준다.
