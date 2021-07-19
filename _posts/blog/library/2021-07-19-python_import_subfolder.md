---
layout: post
bigtitle: [Ubuntu 20.04] 우분투 python subfolder from, import 오류
subtitle: '.'
categories:
    - blog
    - library
tags:
    - 오류
    - import
comments: true
date: '2021-07-18 01:45:51 +0900'
related_posts:
  - category/_posts/blog/blog-etc/2020-07-18-ubuntu_googledrive.md
  - category/_posts/blog/blog-etc/2020-07-18-Macbuntu.md
published: true
---

# python subfolder from, import 오류, ModuleNotFoundError


![그림1](/assets/img/Blog/Etc/macbuntu/15.PNG)
이런 폴더 구조인데

![그림1](/assets/img/Blog/Etc/macbuntu/16.PNG)
lib폴더에 있는 코드를 import하고 싶은데 이상하게 우분투에서는 아래와 같은 오류가 생긴다.
![그림1](/assets/img/Blog/Etc/macbuntu/17.PNG)

이럴때는
>import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

해주면 된다.
