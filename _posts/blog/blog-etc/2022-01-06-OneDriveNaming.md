---
layout: post
bigtitle:  "원드라이브(OneDrive) 폴더이름 바꾸기"
subtitle:   "OneDrive"
categories:
    - blog
    - blog-etc
tags:
    - pose
comments: true
published: true
---

# 원드라이브(OneDrive) 동기화 폴더이름 바꾸기

깃헙 블로그를 어디에서도 작성, 관리할 수 있도록 원드라이브에 관리하려고 하나 아래 그림과 같이 한국어가 있어 깃헙 블로그 에러가 발생하였다.

![그림1](/assets/img/Blog/Etc/OneDrive/1.png){: width="200" height="400"}

깃헙블로그의 대부분 에러는 폴더, 파일, 위치 이름이 한국어일때 이다. 혹시 ```bundle exec jekyll serve``` 명령어에 에러가 생기면 파일이나, 폴더, 전체 폴더 위치중 한국어가 있는지 (예: 바탕화면>GitBlog) 확인해보자

### 그럼 본격적으로 바꿔보자            


### 1. 먼저 원드라이브를 종료한다. 

![그림2](/assets/img/Blog/Etc/OneDrive/2.png) {: width="200" height="400"}

![그림3](/assets/img/Blog/Etc/OneDrive/3.png) {: width="400" height="400"}

### 2. C:\Users\사용자이름 에서 OneDrive 이름을 바꾼다.

![그림4](/assets/img/Blog/Etc/OneDrive/4.png) {: width="400" height="400"}

![그림5](/assets/img/Blog/Etc/OneDrive/5.png) {: width="400" height="400"}

### 3. OneDrive 폴더 경로 변경 


<pre>
<code>
%LOCALAPPDATA%\Microsoft\OneDrive\settings\
</code>
</pre>

에서 원드라이브 계정이 여러개라면 Business의 숫자가 증가한다.
나의 경우 하나이기 때문에 Business1로 들어간다. 

![그림6](/assets/img/Blog/Etc/OneDrive/6.png) {: width="400" height="400"}


원드라이브 설정이 저장된 ini 파일을 메모장으로 연 다음 폴더 경로를 수정해서 원드라이브가 정상적으로 작동할 수 있게 설정한다.

![그림7](/assets/img/Blog/Etc/OneDrive/7.png) {: width="400" height="400"}

![그림8](/assets/img/Blog/Etc/OneDrive/8.png) {: width="400" height="400"}

오른쪽으로 조금 이동하면 원드라이브 폴더명을 바꾸기전 폴더명이 작성되어있다.

### 4. OneDrive 켜고 다시 동기화

해서 확인하기 

![그림9](/assets/img/Blog/Etc/OneDrive/9.png) {: width="400" height="400"}

![그림10](/assets/img/Blog/Etc/OneDrive/10.png) {: width="400" height="400"}