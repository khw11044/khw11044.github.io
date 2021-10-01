---
layout: post
bigtitle:  "Git 명령어 모음"
subtitle:   "Git"
categories:
    - blog
    - blog-etc
tags:
    - pose
comments: true
published: true
---

# Git 명령어 모음

## Git 설치 (우분투에서)

1) 터미널 (ctrl + Alt + T)  
2) <code>sudo apt install git</code>  
3) <code>git --version</code>  

## 1.1 Git 명령어

|분류|명령어|내용설명|
|---|---|---|
|새로운 저장소 생성 | $ git init | .git 하위 디렉토리 생성|
|저장소 복사/다운로드(clone) | $ git clone \<https://...URL\> | 기존 소스 코드 다운로드/복제|
|추가 및 확정 (add&commit) | $ git add <파일명> \ | 커밋에 단일 파일의 변경 사항을 포함 (인덱스에 추가된 상태) |
| | $ git add -A | 커밋에 파일의 변경 사항 모두 |
| | $ git commit -m "메세지" | 커밋 생성 (실제 변경사항 확정)|
| | $ git status | 파일 상태 확인
|가지치기(brantch) | $ git branch | 브랜치 목록 |
| | $ git branch <브랜치이름> | 새 브랜치 생성 (local로 만듦) |
| | $ git checkout -b <브렌치이름> | 브랜치 생성 & 이동 |
| | $ git checkout master | master branch로 되돌아 옴 |
| | $ git branch -d <브렌치이름> | 브렌치 삭제 |
| | $ git push origin <브렌치이름> | 만든 브렌치를 원격 서버에 전송 |
| | $ git push -u <remote> <브렌치이름> | 새 브렌치를 원격 저장소에 push |
| | $ git pull <remote> <브렌치이름> | 운격에 저장된 git 프로젝트의 현재 상태를 다운받고 + 현재 위치한 브렌치로 병합 |
| <변경 사항 발행(push)> | $ git push origin master | 변경사항 원격 서버 업로드 |
| | $ git push <remote> <브렌치이름> | 커밋을 원격 서버에 업로드 |
| | $ git push -u <remote> <브렌치이름> | 커밋을 user로 원격 서버에 업로드 |
| | $ git push -f <remote> <브렌치이름> | 커밋을 강제(force)로 원격 서버에 업로드 |
| | $ git remote add origin <등록된원격서버주소> | 클라우드 주소 등록 및 발행 |
| | $ git remote remove <등록된 클라우드 조소> | 클라우드 주소 삭제 |
| <갱신 및 병합(merge)> | $ git pull | 원격 저장소의 변경 내용이 현재 디렉토리에 가져(fetch)와 병합(merge) |
| | $ git merge <다른브렌치이름> | 현재 브렌치에 다른 브렌치의 수정사항 병합 |
| | $ git add <파일명> | 각 파일을 병합할 수 있음 |
| | $ git diff <브렌치이름> <다른 브렌치이름> | 변경 내용 merge 전에 바뀐 내용을 비교할 수 있음 |
| <태그tag> | $ git log | 현재 위치한 브렌치 커밋 내용 확인 및 식별자 부여 |
| <로컬 변경사항 return> | $ git checkout -- <파일명> | 로컬의 변경 사항을 변경 전으로 되돌림 |
| | $ git fetch origin | 원격에 저장된 git 프로젝트의 현 상태를 다운로드 |


