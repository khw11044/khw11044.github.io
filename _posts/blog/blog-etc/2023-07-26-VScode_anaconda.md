---
layout: post
bigtitle:  "VScode에서 Anaconda 연동 안될 때"
subtitle:   "OneDrive"
categories:
    - blog
    - blog-etc
tags:
    - vscode
    - anaconda
    - conda 
comments: true
published: true
---

# VScode에서 Anaconda 연동 안될 때

## 개요 

보통 Anaconda를 먼저 설치하든, VScode를 먼저 설치하든 보통은 VScode에서 터미널과 Interpreter에 conda가 잘 연동된다. 

그러나 특히, 윈도우에서 잘 연동이 안될때가 있는데 

설치 순서를 바꿔보기도하고 anaconda3 다운받은 폴더 위치를 바꿔보기도 하였는데 

Interpreter에는 연동이 되나 터미널에서는 아래와 같이 안될 때가 있다. 

![0-0](https://github.com/khw11044/blog-comments-repo/assets/51473705/b7a95577-982d-40e5-b325-a0d255cf5040)

근데 또 Interpreter와 명령프롬프트에서는 연동이 잘된다. 

![001](https://github.com/khw11044/blog-comments-repo/assets/51473705/83c41437-ec08-4318-acb5-7d55f5926010)

![002](https://github.com/khw11044/blog-comments-repo/assets/51473705/018fbe9e-8e76-41ac-87cb-64fbc5aaccd0)

이럴때 아래와 같이 해주자.
          

## 1. Terminal에서 Conda 명령어 사용 가능하게 하기위한 Extension 설치 

만약 필자처럼 터미널에 conda 명령어를 쳤을때 경고 메세지가 나오지 않는다면 1. 섹션은 넘어가도 된다.

![003](https://github.com/khw11044/blog-comments-repo/assets/51473705/d6f49c25-ce6b-4700-803f-4ed530e0861e)

위 그림처럼 Extension창에 terminal을 검색하여 설치한다.


### 2. (중요) conda 경로를 환경변수에 등록하기 

윈도우라 확실히 이런것들을 지정해줘야 하는것 같다.

![0](https://github.com/khw11044/blog-comments-repo/assets/51473705/a2c361f5-dacc-49c2-9383-a281ab8dd5b3)

내 PC 우클릭 -> 속성 

![1](https://github.com/khw11044/blog-comments-repo/assets/51473705/90d5c813-5347-4870-a1c4-fe451b851a9b)

'고급 시스템 설정' 클릭 

![2](https://github.com/khw11044/blog-comments-repo/assets/51473705/25848eb7-c118-472e-a1ba-c099b029b578)

'환경 변수' 클릭

![3](https://github.com/khw11044/blog-comments-repo/assets/51473705/ba3f00d7-5525-48f7-a67b-1a129b0da3dd)

'시스템 변수'에서 Path 클릭하고 '편집'버튼 클릭 

![4](https://github.com/khw11044/blog-comments-repo/assets/51473705/c25a260b-8182-41a6-9e70-501d3f3790b1)

설치한 anaconda3 또는 Anaconda3 폴더 위치를 '새로 만들기'버튼을 통해 추가해준다. 

<code>anaconda3</code>, <code>anaconda3\Library</code>, <code>anaconda3\Scripts</code> 3개의 경로를 추가해준다.

## 3. VScode에서 터미널 지정해주기 

이제 다시 VScode로 돌아와서 <code>Ctrl + Shift + P</code>를 누른 뒤 팝업되는 검색 창에 'Terminal: Select Default Profile'을 입력한다. 

![5](https://github.com/khw11044/blog-comments-repo/assets/51473705/63225f9e-5ef2-46f9-828d-f60970ade841)

이후, <code>Command Prompt</code>를 클릭해준다. 

![6](https://github.com/khw11044/blog-comments-repo/assets/51473705/23b6ca29-7f68-4f0b-8547-12991f937e4a)

그리고 터미널 창 오른쪽 위쪽 <code>+</code> 버튼을 통해 새로운 터미널을 여는데 이때 Powershell이 아니라 cmd로 생성되면서 

conda 가상환경을 열어주는것을 확인할 수 있다. 

![7](https://github.com/khw11044/blog-comments-repo/assets/51473705/dc7cd87b-fb9e-42fd-9152-e9aaffacdf81)

-------

## 마치며 

정말 안될때 마다 까먹고 구글링하고 찾는거 그만하고 나를 위해 작성한 포스팅... 

이지만 많은 사람들에게 도움이 되었으면 좋겠다. 