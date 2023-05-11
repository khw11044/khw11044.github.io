---
layout: post
bigtitle:  "[Softeer] [파이썬] 주행거리 비교하기"
subtitle:  "[4일차] [코딩테스트] [Level 1] [Python]"
categories:
    - study
    - codingtest
tags:
  - codingtest
comments: true
published: true
date: '2023-04-09 02:45:51 +0900'
---

# [4일차] [Softeer] [파이썬] [Level 1] 주행거리 비교하기

[https://softeer.ai/practice/info.do?idx=1&eid=1016](https://softeer.ai/practice/info.do?idx=1&eid=1016)

### 입력예제1
3500 2000

### 출력예제1
A

### 입력예제2
1500 1800

### 출력예제2
B


---

### 🚀 나의 풀이 ⭕

```python
import sys 
# sys.stdin = open('input.txt', 'r')
if __name__=="__main__":
    a,b = map(int, input().split())

    if a>b:
        print('A')
    elif a<b:
        print('B')
    elif a==b:
        print('same')
        

```


***
    🌜 개인 공부 기록용 블로그입니다. 오류나 틀린 부분이 있을 경우 
    언제든지 댓글 혹은 메일로 지적해주시면 감사하겠습니다! 😄