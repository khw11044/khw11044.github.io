---
layout: post
bigtitle:  "[프로그래머스] [파이썬] 약수의 합"
subtitle:  "[1일차] [코딩테스트] [연습문제] [LV.1] [Python]"
categories:
    - study
    - codingtest
tags:
  - codingtest
comments: true
published: true
date: '2023-04-05 02:45:51 +0900'
---

# 약수의 합

__문제 설명__ 

정수 <mark>n</mark>을 입력받아 n의 약수를 모두 더한 값을 리턴하는 함수, solution을 완성해주세요.

__제한 사항__

+ n은 0이상 3000이하인 정수입니다. 

__입출력 예__

| n | return |
|---|---|
| 12 | 28 |
| 5 | 6 |

입출력 예 설명
입출력 예 #1
12의 약수는 1, 2, 3, 4, 6, 12입니다. 이를 모두 더하면 28입니다.

입출력 예 #2
5의 약수는 1, 5입니다. 이를 모두 더하면 6입니다.

---

### 🚀 나의 풀이 ⭕

```python
def solution(n):
    answer = 0
    for i in range(1,n+1):
        if n%i == 0:
            answer+=i
    return answer
```

정수 n을 입력받으면 1부터 n까지 수를 나열하고 그중 약수인 것만 골라 더하자 

처음 1은 12를 나누어 나머지가 없으므로 약수고 

다음 2도 12를 나누어 나머지가 없으므로 약수이다. 

이런식으로 1부터 n까지 반복한다