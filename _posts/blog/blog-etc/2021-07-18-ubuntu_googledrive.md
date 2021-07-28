---
layout: post
bigtitle:  "[Ubuntu 20.04] 우분투에서 구글 드라이브 사용하기"
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


# How to mount google drive on ubuntu 20.04

ubuntu에 google drive 마운트하기

구글 드라이브를 fuse형태로 마운트한다.  
마운트를 하면 웹에서 하나하나 다운로드할 필요가 없기 때문에 나는 우분투를 깔면 무조건 구글 드라이브부터 마운트를 한다.  

우분투에 카카오톡을 깔기 힘들기 때문에 자료전송할때 일일이 이메일로 보내기 귀찮은데 클라우드 드라이브 하나쯤있으면 굉장히 편하다.

## 구글 드라이브 프로그램 설치

<pre>
<code>
sudo add-apt-repository ppa:alessandro-strada/ppa

sudo apt update

sudo apt install google-drive-ocamlfuse
</code>
</pre>

## 구글 드라이브로 사용할 임의 폴더 만들기

<pre>
<code>
mkdir ~/GoogleDrive
</code>
</pre>

![그림1](/assets/img/Blog/Etc/ubuntu_googledrive/1.png)

## 구글 드라이브 이용하기

<pre>
<code>
google-drive-ocamlfuse ~/GoogleDrive
</code>
</pre>

![그림2](/assets/img/Blog/Etc/ubuntu_googledrive/2.png)

위 명령어를 치면 아래와 같이 **구글 계정으로 로그인** 창이 뜬다.

![그림3](/assets/img/Blog/Etc/ubuntu_googledrive/3.jpg)

원하는 구글 계정으로 로그인 해준다.  

gdfuse에서 Google 계정에 엑세르하려고 합니다가 뜬다.  
우측 하단에 허용을 누른다.

![그림4](/assets/img/Blog/Etc/ubuntu_googledrive/4.jpg)
![그림5](/assets/img/Blog/Etc/ubuntu_googledrive/5.png)

위 화면이 뜨면 성공이다.

## 결과 확인

![그림6](/assets/img/Blog/Etc/ubuntu_googledrive/6.png)

폴더에서 GoogleDrive로 마운트된것을 볼수 있다.
GoogleDrive 폴더에 들어가도 정상적으로 모든 폴더 파일들을 마운트 한것을 볼수 있다.

## 추가 필수

하지만 컴퓨터를 끄고 키면 GoogleDrive가 마운트되어있지 않다. 원할때 마다 위 명령어들을 쳐주는 것이 굉장히 귀찮으므로 단축키를 설정해 준다.

<pre>
<code>
google-drive-ocamlfuse ~/GoogleDrive
</code>
</pre>


**환경 변수 편집창 열기**
<pre>
<code>
gedit ~/.bashrc
</code>
</pre>

![그림7](/assets/img/Blog/Etc/ubuntu_googledrive/7.png)

아래 명령어 입력 후 저장
> alias gd='google-drive-ocamlfuse ~/GoogleDrive'

![그림8](/assets/img/Blog/Etc/ubuntu_googledrive/8.png)

**변경 내용 저장**

<pre>
<code>
source ~/.bashrc
</code>
</pre>

![그림9](/assets/img/Blog/Etc/ubuntu_googledrive/9.png)

터미널을 열고 gd만 입력하면 구글드라이브를 이용할수 있다.

<pre>
<code>
gd
</code>
</pre>
