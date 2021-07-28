---
layout: post
bigtitle:  "[Ubuntu 20.04] 우분투 python subfolder from, impor"
subtitle:   "."
categories:
    - blog
    - blog-etc
tags:
    - pose
comments: true
published: true
---

* toc
{:toc}


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
