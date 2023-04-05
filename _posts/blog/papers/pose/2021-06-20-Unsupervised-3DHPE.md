---
layout: post
bigtitle:  "Unsupervised 3D Pose Estimation with Geometric Self-Supervision"
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



# Unsupervised 3D Pose Estimation with Geometric Self-Supervision

Amazon Lab126, 2Georgia Institute of Technology Chen, Ching-Hang, et al. "Unsupervised 3d pose estimation with geometric self-supervision." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.[paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Unsupervised_3D_Pose_Estimation_With_Geometric_Self-Supervision_CVPR_2019_paper.html)

* toc
{:toc}

## Abstract

We present an unsupervised learning approach to recover 3D human pose from 2D skeletal joints extracted from a single image.  
Our method does not require any multi-view image data, 3D skeletons, correspondences between 2D-3D points, or use previously learned 3D priors during training.  
A lifting network accepts 2D landmarks as inputs and generates a corresponding 3D skeleton estimate.  
During training, the recovered 3D skeleton is reprojected on random camera viewpoints to generate new ‘synthetic’ 2D poses.  
By lifting the synthetic 2D poses back to 3D and re-projecting them in the original camera view, we can define self-consistency loss both in 3D and in 2D.  
The training can thus be self supervised by exploiting the geometric self-consistency of the lift-reproject-lift process.  
We show that self-consistency alone is not sufficient to generate realistic skeletons, however adding a 2D pose discriminator enables the lifter to output valid 3D poses.   
Additionally, to learn from 2D poses ‘in the wild’, we train an unsupervised 2D domain adapter network to allow for an expansion of 2D data.  
This improves results and demonstrates the usefulness of 2D pose data for unsupervised 3D lifting.  
Results on Human3.6M dataset for 3D human pose estimation demonstrate that our approach improves upon the previous unsupervised methods by 30% and outperforms many weakly supervised approaches that explicitly use 3D data.

> 우리는 단일 이미지에서 추출한 2D 골격 관절에서 3D 인간 자세를 회복하기 위한 비지도 학습 접근법을 제시한다. 우리의 방법은 다중 뷰 이미지 데이터, 3D 골격, 2D-3D 지점 간의 대응 또는 훈련 중에 이전에 학습한 3D 이전 버전을 사용할 필요가 없다. 리프팅 네트워크는 2D 랜드마크를 입력으로 받아들이고 그에 상응하는 3D 스켈레톤 추정치를 생성합니다. 훈련 중에, 복구된 3D 골격은 무작위 카메라 관점에 재투영되어 새로운 '합성' 2D 포즈를 생성한다. 합성 2D 포즈를 다시 3D로 들어올려 원래 카메라 뷰에서 다시 투사함으로써 3D와 2D 모두에서 자기 일관성 손실을 정의할 수 있습니다. 따라서 훈련은 리프트-재프로젝트-리프트 프로세스의 기하학적 자기 일관성을 활용하여 자체 감독될 수 있다. 우리는 자기 일관성만으로는 현실적인 골격을 생성하기에 충분하지 않다는 것을 보여주지만, 2D 포즈 판별기를 추가하면 리프트가 유효한 3D 포즈를 출력할 수 있다. 또한, '야생'의 2D 포즈로부터 배우기 위해, 우리는 2D 데이터의 확장을 허용하도록 감독되지 않은 2D 도메인 어댑터 네트워크를 훈련시킨다. 이것은 결과를 개선하고 감독되지 않은 3D 리프팅에 대한 2D 포즈 데이터의 유용성을 보여준다. 인간3에 대한 결과.3D 인간 포즈 추정을 위한 6M 데이터 세트는 우리의 접근 방식이 이전의 감독되지 않은 방법보다 30% 향상되고 3D 데이터를 명시적으로 사용하는 많은 약하게 감독되는 접근 방식을 능가한다는 것을 보여준다.
