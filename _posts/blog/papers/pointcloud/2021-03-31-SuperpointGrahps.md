---
layout: post
bigtitle:  "Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs"
subtitle:   "."
categories:
    - blog
    - papers
    - pointcloud
tags:
    - point-cloud
    - detection
comments: true
published: true
---



# Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs

CVPR 2018 [paper](https://ieeexplore.ieee.org/document/8578577)


* toc
{:toc}

## Abstract
We propose a novel deep learning-based framework to tackle the challenge of semantic segmentation of largescale point clouds of millions of points.  
We argue that the organization of 3D point clouds can be efficiently captured by a structure called superpoint graph (SPG), derived from a partition of the scanned scene into geometrically homogeneous elements.  
SPGs offer a compact yet rich representation of contextual relationships between object parts, which is then exploited by a graph convolutional network.  
Our framework sets a new state of the art for segmenting outdoor LiDAR scans (+11.9 and +8.8 mIoU points for both Semantic3D test sets), as well as indoor scans (+12.4 mIoU points for the S3DIS dataset).

## 1. Introduction

Semantic segmentation of large 3D point clouds presents numerous challenges, the most obvious one being the scale of the data. Another hurdle is the lack of clear structure akin to the regular grid arrangement in images. These obstacles have likely prevented Convolutional Neural Networks (CNNs) from achieving on irregular data the impressive performances attained for speech processing or images.

Previous attempts at using deep learning for large 3D data were trying to replicate successful CNN architectures used for image segmentation. For example, SnapNet [5] converts a 3D point cloud into a set of virtual 2D RGBD snapshots, the semantic segmentation of which can then be projected on the original data. SegCloud [44] uses 3D convolutions on a regular voxel grid. However, we argue that such methods do not capture the inherent structure of 3D point clouds, which results in limited discrimination performance. Indeed, converting point clouds to 2D format comes with loss of information and requires to perform surface reconstruction, a problem arguably as hard as semantic segmentation. Volumetric representation of point clouds is inefficient and tends to discard small details.

Deep learning architectures specifically designed for 3D point clouds [37, 43, 40, 38, 10] display good results, but are limited by the size of inputs they can handle at once.

We propose a representation of large 3D point clouds as a collection of interconnected simple shapes coined superpoints, in spirit similar to superpixel methods for image segmentation [1]. As illustrated in Figure 1, this structure can be captured by an attributed directed graph called the superpoint graph (SPG). Its nodes represent simple shapes while edges describe their adjacency relationship characterized by rich edge features.

The SPG representation has several compelling advantages. First, instead of classifying individual points or voxels, it considers entire object parts as whole, which are easier to identify. Second, it is able to describe in detail the relationship between adjacent objects, which is crucial for contextual classification: cars are generally above roads, ceilings are surrounded by walls, etc. Third, the size of the SPG is defined by the number of simple structures in a scene rather than the total number of points, which is typically several order of magnitude smaller. This allows us to model long-range interaction which would be intractable otherwise without strong assumptions on the nature of the pairwise connections. Our contributions are as follows:

- We introduce superpoint graphs, a novel point cloud representation with rich edge features encoding the contextual relationship between object parts in 3D point clouds.

- Based on this representation, we are able to apply deep learning on large-scale point clouds without major sacrifice in fine details.  
Our architecture consists of Point-Nets [37] for superpoint embedding and graph convolutions
for contextual segmentation. For the latter, we introduce a novel, more efficient version of Edge-Conditioned Convolutions [43] as well as a new form of input gating in Gated Recurrent Units [8].  

- We set a new state of the art on two publicly available datasets: Semantic3D [14] and S3DIS [3]. In particular, we improve mean per-class intersection over union (mIoU) by 11.9 points for the Semantic3D reduced test set, by 8.8 points for the Semantic3D full test set, and by up to 12.4 points for the S3DIS dataset.

## 2. Related Work

![Fig1](/assets/img/Blog/papers/LargescalePointCloudSuperpointGraphs/Fig1.JPG)

The classic approach to large-scale point cloud segmentation is to classify each point or voxel independently using handcrafted features derived from their local neighborhood [46]. The solution is then spatially regularized using graphical models [35, 22, 32, 42, 20, 2, 36, 33, 47] or structured optimization [25]. Clustering as preprocessing [16, 13] or postprocessing [45] have been used by several frameworks to improve the accuracy of the classification.

**Deep Learning on Point Clouds**.  
Several different approaches going beyond naive volumetric processing of point clouds have been proposed recently, notably setbased [37, 38], tree-based [40, 21], and graph-based [43]. However, very few methods with deep learning components have been demonstrated to be able to segment large-scale point clouds. PointNet [37] can segment large clouds with a sliding window approach, therefore constraining contextual information within a small area only. Engelmann et al. [10] improves on this by increasing the context scope with multi-scale windows or by considering directly neighboring window positions on a voxel grid. SEGCloud [44] handles large clouds by voxelizing followed by interpolation back to the original resolution and post-processing with a conditional random field (CRF). None of these approaches is able to consider fine details and long-range contextual information simultaneously. In contrast, our pipeline partitions point clouds in an adaptive way according to their geometric complexity and allows deep learning architecture to use both fine detail and interactions over long distance.

**Graph Convolutions**.  
A key step of our approach is using graph convolutions to spread contextual information. Formulations that are able to deal with graphs of variable sizes can be seen as a form of message passing over graph edges [12]. Of particular interest are models supporting continuous edge attributes [43, 34], which we use to represent interactions. In image segmentation, convolutions on graphs built over superpixels have been used for postprocessing: Liang et al. [30, 29] traverses such graphs in a sequential node order based on unary confidences to improve the final labels. We update graph nodes in parallel and exploit edge attributes for informative context modeling. Xu et al. [48] convolves information over graphs of object detections to infer their contextual relationships. Our work infers relationships implicitly to improve segmentation results. Qi et al. [39] also relies on graph convolutions on 3D point clouds. However, we process large point clouds instead of small RGBD images with nodes embedded in 3D instead of 2D in a novel, rich-attributed graph. Finally, we note that graph convolutions also bear functional similarity to deep learning formulations of CRFs [49], which we discuss more in Section 3.4.

## 3. Method

The main obstacle that our framework tries to overcome is the size of LiDAR scans. Indeed, they can reach hundreds of millions of points, making direct deep learning approaches intractable. The proposed SPG representation allows us to split the semantic segmentation problem into three distinct problems of different scales, shown in Figure 2, which can in turn be solved by methods of corresponding complexity:

1. **Geometrically homogeneous partition:**  
The first step of our algorithm is to partition the point cloud into geometrically simple yet meaningful shapes, called superpoints. This unsupervised step takes the whole point cloud as input, and therefore must be computationally very efficient. The SPG can be easily computed from this partition.

2. **Superpoint embedding:**  
Each node of the SPG corresponds to a small part of the point cloud corresponding to a geometrically simple primitive, which we assume to be semantically homogeneous. Such primitives can be reliably represented by downsampling small point clouds to at most hundreds of points. This small size allows us to utilize recent point cloud embedding methods such as PointNet [37].

3. **Contextual segmentation:**  
The graph of superpoints is by orders of magnitude smaller than any graph built on the original point cloud. Deep learning algorithms based on graph convolutions can then be used to classify its nodes using rich edge features facilitating longrange interactions.

The SPG representation allows us to perform end-to-end learning of the trainable two last steps. We will describe each step of our pipeline in the following subsections.

### 3.1. Geometric Partition with a Global Energy

In this subsection, we describe our method for partitioning the input point cloud into parts of simple shape. Our objective is not to retrieve individual objects such as cars or chairs, but rather to break down the objects into simple parts, as seen in Figure 3. However, the clusters being geometrically simple, one can expect them to be semantically homogeneous as well, i.e. not to cover objects of different classes. Note that this step of the pipeline is purely unsupervised and makes no use of class labels beyond validation.

We follow the global energy model described by [13] for its computational efficiency. Another advantage is that the segmentation is adaptive to the local geometric complexity. In other words, the segments obtained can be large simple shapes such as roads or walls, as well as much smaller components such as parts of a car or a chair.

Let us consider the input point cloud $$C$$ as a set of $$n$$ 3D points.  
Each point $$i \in C$$ is defined by its 3D position $$p_i$$, and, if available, other observations $$o_i$$ such as color or intensity. For each point, we compute a set of $$d_g$$ geometric features $$f_i \in R^{d_g}$$ characterizing the shape of its local neighborhood. In this paper, we use three dimensionality values proposed by [9]: linearity, planarity and scattering, as well as the verticality feature introduced by [13]. We also compute the elevation of each point, defined as the z coordinate of $$p_i$$ normalized over the whole input cloud.

The global energy proposed by [13] is defined with respect to the 10-nearest neighbor adjacency graph $$G_{nn} = (C,E_{nn})$$ of the point cloud (note that this is not the SPG). The geometrically homogeneous partition is defined as the constant connected components of the solution of the following optimization problem:

$$\arg \min_{g \in \mathbb{R^{d_g}}} \sum_{i \in C}||g_i - f_i||^2 + \mu \sum_{(i,j) \in E_nn} w_{i,j}[g_i - g_j \ne 0] \qquad (1)$$

,where [·] is the Iverson bracket. The edge weight $$w \in \mathbb{R}^{|E|}_{+}$$ is chosen to be linearly decreasing with respect to the edge length.  
The factor $$μ$$ is the regularization strength and determines the coarseness of the resulting partition.

The problem defined in _Equation 1_ is known as _generalized minimal partition problem_, and can be seen as a continuous-space version of the Potts energy model, or an $$ℓ_0$$ variant of the graph total variation. The minimized functional being nonconvex and noncontinuous implies that the problem cannot realistically be solved exactly for large point clouds. However, the $$ℓ_0$$-cut pursuit algorithm introduced by [24] is able to quickly find an approximate solution with a few graph-cut iterations. In contrast to other optimization methods such as α-expansion [6], the $$ℓ_0$$-cut pursuit algorithm does not require selecting the size of the partition in advance. The constant connected components $$S = \{S_1, · · · , S_k\}$$ of the solution of Equation 1 define our geometrically simple elements, and are referred as _super-points_ (i.e. set of points) in the rest of this paper.

![Table1](/assets/img/Blog/papers/LargescalePointCloudSuperpointGraphs/Table1.JPG)
