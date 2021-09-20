---
layout: post
bigtitle: PyTorch KeypointRCNN
subtitle: 'KeypointRCNN'
categories:
    - blog
    - library
tags: PyTorch
comments: true
related_posts:

published: true
---

# KeypointRCNN
## KeypointRCNN
---

~~~python
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torchvision import models
import numpy as np


model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model.eval()

if __name__ == '__main__':
    # inputs = np.zeros((3,3,160,160))
    # image = Image.open('f_00715_0000.png')
    t = np.zeros([3,3,160,160])
    ft = torch.FloatTensor(t)
    outputs = model(ft)
    print(outputs)
~~~

결과
~~~
outputs
[{'boxes': tensor([], size=(0, ...Backward>), 'labels': tensor([], dtype=torch.int64), 'scores': tensor([], grad_fn=<...Backward>), 'keypoints': tensor([], size=(0, 17, 3)), 'keypoints_scores': tensor([], size=(0, 17))}, {'boxes': tensor([], size=(0, ...Backward>), 'labels': tensor([], dtype=torch.int64), 'scores': tensor([], grad_fn=<...Backward>), 'keypoints': tensor([], size=(0, 17, 3)), 'keypoints_scores': tensor([], size=(0, 17))}, {'boxes': tensor([], size=(0, ...Backward>), 'labels': tensor([], dtype=torch.int64), 'scores': tensor([], grad_fn=<...Backward>), 'keypoints': tensor([], size=(0, 17, 3)), 'keypoints_scores': tensor([], size=(0, 17))}]
~~~
