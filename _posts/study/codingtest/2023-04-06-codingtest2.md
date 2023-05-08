---
layout: post
bigtitle:  "[프로그래머스] [파이썬] 달리기 경주"
subtitle:  "[2일차] [코딩테스트] [연습문제] [LV.1] [Python]"
categories:
    - study
    - codingtest
tags:
  - codingtest
comments: true
published: true
date: '2023-04-06 02:45:51 +0900'
---


# [2일차] [프로그래머스] [파이썬] 달리기 경주

__문제 설명__ 

얀에서는 매년 달리기 경주가 열립니다. 해설진들은 선수들이 자기 바로 앞의 선수를 추월할 때 추월한 선수의 이름을 부릅니다. 예를 들어 1등부터 3등까지 "mumu", "soe", "poe" 선수들이 순서대로 달리고 있을 때, 해설진이 "soe"선수를 불렀다면 2등인 "soe" 선수가 1등인 "mumu" 선수를 추월했다는 것입니다. 즉 "soe" 선수가 1등, "mumu" 선수가 2등으로 바뀝니다.

선수들의 이름이 1등부터 현재 등수 순서대로 담긴 문자열 배열 players와 해설진이 부른 이름을 담은 문자열 배열 callings가 매개변수로 주어질 때, 경주가 끝났을 때 선수들의 이름을 1등부터 등수 순서대로 배열에 담아 return 하는 solution 함수를 완성해주세요.

__제한 사항__

+ 5 ≤ players의 길이 ≤ 50,000
  - players[i]는 i번째 선수의 이름을 의미합니다.
  - players의 원소들은 알파벳 소문자로만 이루어져 있습니다.
  - players에는 중복된 값이 들어가 있지 않습니다.
  - 3 ≤ players[i]의 길이 ≤ 10

+ 2 ≤ callings의 길이 ≤ 1,000,000
  - callings는 players의 원소들로만 이루어져 있습니다.
  - 경주 진행중 1등인 선수의 이름은 불리지 않습니다.

__입출력 예__

| players | callings | result |
|---|---| ---| 
| ["mumu", "soe", "poe", "kai", "mine"] | ["kai", "kai", "mine", "mine"] | ["mumu", "kai", "mine", "soe", "poe"]


입출력 예 설명
입출력 예 #1

4등인 "kai" 선수가 2번 추월하여 2등이 되고 앞서 3등, 2등인 "poe", "soe" 선수는 4등, 3등이 됩니다. 5등인 "mine" 선수가 2번 추월하여 4등, 3등인 "poe", "soe" 선수가 5등, 4등이 되고 경주가 끝납니다. 1등부터 배열에 담으면 ["mumu", "kai", "mine", "soe", "poe"]이 됩니다.

---

### 🚀 나의 풀이 ⭕

```python
# 포인트는 insert()와 pop() 또는 remove()등을 쓰지 않는 것이다. 
def solution(players, callings):
    # 선수: 위치
    p_idx_dict = {player: i for i, player in enumerate(players)}
    # 위치: 선수
    idx_p_dict = {i: player for i, player in enumerate(players)}
    
    for call in callings:
        cur_idx = p_idx_dict[call]  # 현재 선수의 위치
        pre_idx = cur_idx-1         # 현재 선수보다 앞에 있는 선수 등수
        cur_player = call
        pre_player = idx_p_dict[pre_idx]    # *

        p_idx_dict[cur_player] = pre_idx    # 현재 선수는 앞에 선수 등수가 되고
        p_idx_dict[pre_player] = cur_idx    # 앞에 선수는 뒷 등수가 되고 
        
        idx_p_dict[pre_idx] = cur_player    # 등수별 선수 표 업데이트 * 때문에
        idx_p_dict[cur_idx] = pre_player

    return list(idx_p_dict.values())
```

포인트는 insert()와 pop() 또는 remove()등을 쓰지 않는 것이었다. 

그럼 딕셔너리를 써야하는데 

현재 불러진 player의 등수를 업데이트하고 <br>
앞에 등수 player도 업데이트를 해야하는데 <br>
앞에 등수 player를 딕셔너리의 등수(value)로 이름을 찾자니 또 for문을 써야하기 때문에 <br>

{이름: 등수} 인 딕셔너리랑 <br>
{등수: 이름} 인 딕셔너리를 둘다 만들어 이름이 불러질 때마다 (for문이 돌때마다) 두 딕셔너리를 업데이트 해준다.


***
    🌜 개인 공부 기록용 블로그입니다. 오류나 틀린 부분이 있을 경우 
    언제든지 댓글 혹은 메일로 지적해주시면 감사하겠습니다! 😄