---
layout: post
bigtitle: Atom에 Anaconda 연동하기
subtitle: 'Atom 터미널에 Anaconda (base) 뜨게 하기'
categories:
    - blog
    - blog-etc
tags:
    - atom
comments: true
date: '2020-12-21 14:45:51 +0900'
related_posts:
  - category/_posts/etc/2020-12-21-markdown-tutorial.md
  - category/_posts/etc/2020-12-21-markdown-tutorial2.md
published: true
---

# Atom에 Anaconda 연동하기

## 개요
> Atom에 VS Code처럼 터미널을 이용할수 있게 세팅하고 아나콘다와 연동하여 터미널에 (base)가 뜨게 한다.


## atom platformio-ide-terminal 패키지 설치
---

[File] - [Settings] (Ctrl + ,) - [Install]

platformio-ide-terminal 검색해서 설치



## atom platformio-ide-terminal에 아나콘다 연동
---
atom platformio-ide-terminal에 anaconda를 연동을 해보자.

열심히 찾아봤는데 일단 한국 블로그는 설명이 없다. 대박~!!

내가 최초 게시물이다.

다른분들은 퍼갈때 제 블로그 출처 남겨주세요~ ㅎㅎㅎ

![그림1](/assets/img/Blog/Etc/setting/atom-conda1.png)

보는것과 같이 platformio-ide-terminal을 설치하고 <code>Ctrl+\`</code> 하면 위 그림과 같이 나온다.

당연히 <code>conda</code> 명령어를 치면 오류메시지가 나온다.

<code>Ctrl+,</code>로 Settings에 들어간다.

[Settings] - [Packages]에  platformio-ide-terminal 패키지를 찾고 Settings 버튼을 누른다.

**Core**에 **Auto run Command**에 default를 자신의 아나콘다 activate 위치를 넣어준다.

C:/Anaconda3/Scripts/activate

![그림2](/assets/img/Blog/Etc/setting/atom-conda2.png)

\+ 추가적으로 나는 Shell Override도 바꿔주었다.

C:\WINDOWS\System32\cmd.exe

![그림3](/assets/img/Blog/Etc/setting/atom-conda3.png)

성공~~

### Reference
> > 참고 : [https://stackoverflow.com/questions/43207427/using-anaconda-environment-in-atom](https://stackoverflow.com/questions/43207427/using-anaconda-environment-in-atom)
