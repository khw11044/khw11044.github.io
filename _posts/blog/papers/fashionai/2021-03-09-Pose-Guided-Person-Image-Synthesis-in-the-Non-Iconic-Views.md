---
layout: post
bigtitle:  "Pose Guided Person Image Synthesis in the Non-Iconic-Views"
subtitle:   "Pose Guided Person Image Synthesis in the Non-Iconic-Views"
categories:
    - blog
    - papers
    - fashionai
tags:
    - FashionAI
    - Pose-Guided-Person-Image-Synthesis
comments: true
published: true
---

# Pose Guided Person Image Synthesis in the Non-Iconic-Views

Chengming Xu , Yanwei Fu , Chao Wen , Ye Pan, Yu-Gang Jiang , Member, IEEE,
and Xiangyang Xue , Member, IEEE

## Abstract

Generating realistic images with the guidance of reference images and human poses is challenging.  
Despite the success of previous works on synthesizing person images in the iconic views, no efforts are made towards the task of poseguided image synthesis in the non-iconic views.  
Particularly, we find that previous models cannot handle such a complex task, where the person images are captured in the non-iconic views by commercially-available digital cameras.  

To this end, we propose a new framework – Multi-branch Refinement Network (MR-Net), which utilizes several visual cues, including target person poses, foreground person body and scene images parsed.  
Furthermore, a novel Region of Interest (RoI) perceptual loss is proposed to optimize the MR-Net.  
Extensive experiments on two non-iconic datasets, Penn Action and BBC-Pose, as well as an iconic dataset
– Market-1501, show the efficacy of the proposed model that can tackle the problem of pose-guided person image generation from the non-iconic views.  
The data, models, and codes are downloadable from https://github.com/loadder/MR-Net.

Index Terms—Image processing, image generation.

## 1. INTRODUCTION

SYNTHESIZING images with bespoke human poses has recently attracted pervasive research attention in computer
vision, multimedia, and graphics communities [5], [23], [24], [31], [33].  
Broadly speaking, pose-guided person image synthesis can be applied in many scenarios, including virtual environment rendering, photography editing, character animation, physics-based simulation, and motion control, etc.  
Furthermore, the forged person images can also be utilized in the applications of video generation [10] and video completion [5].  

Recently, extensive works have been conducted in synthesizing iconic person images.  
Here we inherit the definition of “iconic” in [19].  
Particularly, as shown in Fig. 1 (a), the person images of Market-1501 dataset are in the iconic views: high-quality person instances in the center of images, but lacking important contextual information and non-canonical viewpoints.  
Previous works [5], [23], [24], [31], [33] perform fairly well on such iconic person image datasets, e.g., Market-1501 and Deep Fashion [21].  

These datasets, in general, are in very simple scenes, mostly street views which are not required to predict, or backgroundswith single color.  
Moreover, persons are either standing or walking and not occluded by objects.  
However, extending previous models to non-iconic person images would lead to unstable training and generated results Wwith low quality, as shown in Fig. 2.
