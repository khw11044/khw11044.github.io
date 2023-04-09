---
layout: post
bigtitle:  "[Softeer] [파이썬] A+B"
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

# A+B

[https://softeer.ai/practice/info.do?idx=1&eid=362&sw_prbl_sbms_sn=173126](https://softeer.ai/practice/info.do?idx=1&eid=362&sw_prbl_sbms_sn=173126)


### 입력형식
첫째 줄에 테스트 케이스의 개수 T가 주어진다.
각 테스트 케이스는 한 줄로 이루어져 있으며, 각 줄에 A와 B가 주어진다.

### 출력형식
각 테스트 케이스마다 "Case #(테스트 케이스 번호): "를 출력한 다음, A+B를 출력한다.
테스트 케이스 번호는 1부터 시작한다.

### 입력예제1
5      <br>
1 1     <br>
2 3     <br>
3 4     <br>
9 8     <br>
5 2     <br>

### 출력예제1
Case #1: 2      <br>
Case #2: 5      <br>
Case #3: 7      <br>
Case #4: 17     <br>
Case #5: 7      <br>


---

### 🚀 나의 풀이 ⭕

```python
import sys 

if __name__=="__main__":
    n = int(input())

    for i in range(1,n+1):
        a,b = map(int, input().split())
        print('Case #{}: {}'.format(i,str(a+b)))

```


***
    🌜 개인 공부 기록용 블로그입니다. 오류나 틀린 부분이 있을 경우 
    언제든지 댓글 혹은 메일로 지적해주시면 감사하겠습니다! 😄