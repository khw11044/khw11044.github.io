---
layout: post
bigtitle:  "[Github] Push 오류"
subtitle:   "Unable to push remote"
categories:
    - blog
    - blog-etc
tags:
    - pose
comments: true
published: true
---


# [Github] Push 오류, Unable to push remote

아침에 일어나 github repository에 블로그 게시물을 업로드하려니 이런 문구가 뜨면서 Push가 되지 않았다.


**remote: Support for password authentication was removed on August 13, 2021. Please use a personal access token instead.
remote: Please see https://github.blog/2020-12-15-token-authentication-requirements-for-git-operations/ for more information.
fatal: unable to access 'https://github.com/khw11044/khw11044.github.io.git/': The requested URL returned error: 403**


**Unable to push remote: Support for password authentication was removed on August 13, 2021. Please use a personal access token instead. remote: Please see https://github.blog/2020-12-15-token-authentication-requirements-for-git-operations/ for more information. fatal: unable to access 'https://github.com/khw11044/khw11044.github.io.git/': The requested URL returned error: 403**

뭔일인지 바로 구글링하니 8월 13일부터 내 레파지토리에 올릴때
git config --global user.name 이름
git config --global user.email 이메일

하고 비밀번호 물어볼때 계정 비밀번호 하던것을 깃헙 개인 개발자 토큰? 이나 ssh로 해야한다고 한다...

https://github.blog/2020-12-15-token-authentication-requirements-for-git-operations/

즉, 2021년 8월 13일부터 '비밀번호를 통한 인증'을 지원하지 않는다. Git CLI나 GitHub에 접근하는 기타 서비스 등에서 더는 패스워드로 인증을 진행할 수 없다.

패스워드를 통한 인증이 만료되면 Personal Access Token 또는 SSH Key를 통해서 인증을 진행해야한다.

이전에 ssh key발급받고 여러 깃허브 계정을 한 컴퓨터에서 관리할려고 했으나 계속되는 실패에 포기했어서... 일단 ssh는 일단 일단 패스하고 Personal Access Token부터 해보도록 했다.

Settings > Developer settings > Personal access tokens > Generate new token

![그림1](/assets/img/Blog/Etc/githubtoken/1.png)

![그림2](/assets/img/Blog/Etc/githubtoken/2.jpg)

![그림3](/assets/img/Blog/Etc/githubtoken/3.jpg)

![그림4](/assets/img/Blog/Etc/githubtoken/4.jpg)

![그림5](/assets/img/Blog/Etc/githubtoken/5.jpg)

이렇게 얻은 토큰은 바로 복사해서 어디 메모장이나 스티커 메모에 적어 저장한다.


제어판 > 사용자 계정 > Windows 자격 증명 관리 > github.com 찾기 > 편집
그리고 암호에 토큰을 넣어주면 된다.

![그림6](/assets/img/Blog/Etc/githubtoken/6.jpg)

![그림7](/assets/img/Blog/Etc/githubtoken/7.jpg)

![그림8](/assets/img/Blog/Etc/githubtoken/8.png)

![그림9](/assets/img/Blog/Etc/githubtoken/9.jpg)

![그림9](/assets/img/Blog/Etc/githubtoken/10.png)

이제 push가 아무 오류없이 잘된다.
