---
layout: post
bigtitle:  "윈도우 11 우클릭 스타일을 윈도우 10 우클릭 스타일로 바꾸기"
subtitle:   "윈도우 11 우클릭"
categories:
    - blog
    - blog-etc
tags:
    - pose
comments: true
published: true
---

# 윈도우 11 우클릭 스타일을 윈도우 10 우클릭 스타일로 바꾸기

윈도우 11 우클릭 스타일은 너무 불편하다. 
그래서 윈도우 10 우클릭 스타일로 바꾸는 방법을 기록한다. 

## 1. 명령 프롬프트 (cmd) 열기

Win + r 단축키에 cmd 

![그림1](/assets/img/Blog/Etc/RightClick/2.png) {: width="400" height="400"}

cmd에 차례대로 작성 

<pre>
<code>
reg.exe add "HKCU\Software\Classes\CLSID\{86ca1aa0-34aa-4e8b-a509-50c905bae2a2}\InprocServer32" /f /ve


taskkill /f /im explorer.exe


explorer.exe
</code>
</pre>

## 2. 성공 확인 

![그림2](/assets/img/Blog/Etc/RightClick/1.png) {: width="400" height="400"}