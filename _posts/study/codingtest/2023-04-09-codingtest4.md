---
layout: post
bigtitle:  "[프로그래머스] [파이썬] 둘만의 암호"
subtitle:  "[4일차] [코딩테스트] [연습문제] [LV.1] [Python]"
categories:
    - study
    - codingtest
tags:
  - codingtest
comments: true
published: true
date: '2023-04-09 02:45:51 +0900'
---

# [4일차] [프로그래머스] [파이썬] 둘만의 암호

[https://school.programmers.co.kr/learn/courses/30/lessons/155652](https://school.programmers.co.kr/learn/courses/30/lessons/155652)


---

### 🚀 나의 풀이 ⭕

```python
def solution(s, skip, index):
    answer = ''
    alpha= [chr(i) for i in range(ord('a'),ord('z')+1) if chr(i) not in skip]
    
    for x in s:
        change=alpha[(alpha.index(x)+index) % len(alpha)]
        answer += change
    return answer
```

a부터 z까지중에 skip에 있는 문자들은 뺀 alpha 라는 변수의 리스트를 만들고 
입력받은 s의 문자의 alpha위치에 입력받은 index를 더해서 문자를 바꾸는데 리스트 인덱스를 넘으면 다시 0번째부터 세기 위해 %를 쓴다. 


***
    🌜 개인 공부 기록용 블로그입니다. 오류나 틀린 부분이 있을 경우 
    언제든지 댓글 혹은 메일로 지적해주시면 감사하겠습니다! 😄