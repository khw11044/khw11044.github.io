---
layout: post
bigtitle:  "Windows11 설치하기"
subtitle:   "UEFI BIOS Utility"
categories:
    - blog
    - blog-etc
tags:
    - pose
comments: true
published: true
---


# Windows 11 Insider Preview 설치하기

## 윈도우 11 베터 인사이더 프리뷰 설치 방법

UEFI BIOS Utility 에서 TPM 또는 PTT Enable하기  
최신 컴인데 TPM이 없다고 나오는경우!!

### 윈도우11 인사이더 프리뷰 설치 조건  
1. 윈도우 11이 요구하는 하드웨어 소유자
  - 인텔 6세대 이상, 라이젠 1세대 이상, TPM 2.0또는 PTT 2.0, 그래픽 인텔 7.5세대 지포스 GTX 200, 라데온 HD 6000이상
2. 윈도우10 정품 사용자
3. 선택형 진단 데이터 설정
4. Windows 참가자 프로그램 계정 등록
5. 참가자 설정(Dev 채널(개발자모드)/ 베타 채널(얼리어답터모드))

컴퓨터를 새로사서 윈도우11을 설치해보기로 했다.

새컴퓨터니까 당연히 하드웨어적 조건은 다 통과인줄 알았다.

### 윈도우11 설치 가능 여부 TPM.msc 확인

- 'Windows키 + r' > '실행' > 'tpm.msc' 입력
- TPM 2.0 지원 확인

 ![그림1](/assets/img/Blog/Etc/window11/1.PNG)

하면 TPM이 없다고 한다. 멘붕... 뭐지 하면서 포기할려고 했으나 할것도 없고 해서 좀더 알아보기로 했다. 알고보니 BIOS 설정에서 TPM 또는 PTT를 Enable이 안되어서 인식이 안되는 거라는 것을 알게되었다.

리부팅을 하고 BIOS환경에 진입한다.
 ![그림1-1](/assets/img/Blog/Etc/window11/2.jpg)

Advanced Mode > PCH-FW Configuration > TPM Device selection
에서 Enable Firmware TPM을 골라준다.

 ![그림1-2](/assets/img/Blog/Etc/window11/1-1.jpg)
 ![그림1-3](/assets/img/Blog/Etc/window11/1-2.jpg)

그럼 저장하고 다시 부팅하면 아래와같이 TPM을 인식하는것을 알수 있다.

 ![그림2](/assets/img/Blog/Etc/window11/2.PNG)

### 윈도우11 설치 가능여부 장치 보안 확인
1. 윈도우 설정 ('Windows키 + i')
2. '업데이트 및 보안' > '장치보안' > '보안프로세서' > '보안프로세서 정보'
3. TPM 1.2 - 2.0 확인

![그림3](/assets/img/Blog/Etc/window11/3.jpg)

![그림4](/assets/img/Blog/Etc/window11/4.jpg)

![그림5](/assets/img/Blog/Etc/window11/5.jpg)

![그림6](/assets/img/Blog/Etc/window11/6.jpg)

## 본격 Windows 11 Insider Preview 설치하기

1. Windows 참가자 프로그램 설정
2. 시작할 계정 선택
3. 참가자 설정
4. 재부팅
5. 업데이트 확인

1. Windows 참가자 프로그램 설정
 - '업데이트 및 보안' > 'Windows 보안' > 'Windows 보안 열기' > '피드백 및 진단'
![그림7](/assets/img/Blog/Etc/window11/7.jpg)  
![그림8](/assets/img/Blog/Etc/window11/8.jpg)

- '업데이트 및 보안' > 'Windows 참가자 프로그램' > '진단 & 피드백 설정으로 이동하여-선택형 진단 데이터를 설정합니다.' > '시작'
![그림9](/assets/img/Blog/Etc/window11/9.jpg)
![그림10](/assets/img/Blog/Etc/window11/10.jpg)

- 등록 > 등록 > 로그인 > 참가자 설정 선 > 확인

![그림11](/assets/img/Blog/Etc/window11/11.png)
![그림12](/assets/img/Blog/Etc/window11/12.png)
![그림13](/assets/img/Blog/Etc/window11/13.png)
![그림14](/assets/img/Blog/Etc/window11/14.png)
![그림15](/assets/img/Blog/Etc/window11/15.png)

자 이러고 잠시 기다리고 다시부팅이 되고 보니!  
![그림16](/assets/img/Blog/Etc/window11/16.png)  
아무것도 변한게 없어서 당황....
위 그림의 '업데이트 확인 버튼'을 만든다.

![그림17](/assets/img/Blog/Etc/window11/17.png)
위와 같이 다운이 시작된다.

![그림18](/assets/img/Blog/Etc/window11/18.png)
다운이 완료되면 다음과 같은 알람이 뜬다.

![그림19](/assets/img/Blog/Etc/window11/19.jpg)
![그림20](/assets/img/Blog/Etc/window11/20.jpg)
![그림21](/assets/img/Blog/Etc/window11/21.jpg)
![그림22](/assets/img/Blog/Etc/window11/22.png)

완성~
