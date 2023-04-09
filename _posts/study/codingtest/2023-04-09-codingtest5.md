---
layout: post
bigtitle:  "[Softeer] [파이썬] 로봇이 지나간 경로"
subtitle:  "[4일차] [코딩테스트] [인증평가(1차) 기출] [Python]"
categories:
    - study
    - codingtest
tags:
  - codingtest
comments: true
published: true
date: '2023-04-09 02:45:51 +0900'
---

# 로봇이 지나간 경로

[https://softeer.ai/practice/info.do?idx=1&eid=577](https://softeer.ai/practice/info.do?idx=1&eid=577)


### 입력형식
첫 번째 줄에 격자판의 세로 크기 H와 가로 크기 W가 공백 하나를 사이로 두고 주어진다. 다음에는 사수가 넘겨준 지도가 주어진다. H개의 줄에 W개의 문자가 주어지는데, 이 중 i(1 ≤ i ≤ H)번째 줄의 j(1 ≤ j ≤ W)번째 문자는, 사수가 조작한 로봇이 i행 j열을 방문했다면 '#'이고, 방문하지 않았다면 '.'이다.

### 출력형식
첫 번째 줄에 두 개의 정수 a(1 ≤ a ≤ H)와 b(1 ≤ b ≤ W)를 공백 하나씩을 사이로 두고 출력한다. 이는 처음 로봇을 격자판의 a행 b열에 두어야 함을 의미한다.


두 번째 줄에 '>', '<', 'v', '^' (따옴표 제외) 중 하나를 출력한다. 이 문자는 처음 로봇이 바라보는 방향을 의미하며, >는 동쪽, <는 서쪽, v는 남쪽, ^는 북쪽이다.<br> 세 번째 줄에 당신이 로봇에 내려야 하는 명령어들을 공백 없이 명령을 내릴 순서대로 출력한다. 이 문자열의 길이가 곧 당신이 내리는 명령어의 개수이며, 명령어의 개수를 최소화해야 정답 처리된다.

명령어의 개수를 최소화하면서 목표를 달성할 수 있는 방법이 여러 가지라면, 그 중 한 가지를 아무거나 출력하면 된다.

![이미지1](../../../assets\img\Study\CodingTest\2023-04-09-cote5_1.png){: .align-center}

![이미지1](../../../assets\img\Study\CodingTest\2023-04-09-cote5_2.png){: .align-center}

처음에 문제를 볼때 이해하기 힘든데 #을 이어주면 그게 로봇이 지나간 루트이고 어디서 시작했는지를 찾으며 컨트롤 명령어순서를 찾는 문제로 __지도,미래__ 문제와 비슷하다는 느낌이 들면서 DFS나 BFS문제로 느껴진다.

1. 시작위치를 찾기 위해 격자판을 행,열로 탐색한다. 
2. '#'이 시작하는 시작위치를 찾으면 BFS나 DFS로 상하좌우를 살피며 '#'인곳으로만 길을 간다.
3. 간 길에 방향표시와 방문표시를 한다
4. 2칸에 A 명령어, 방향 바뀌면 L 또는 R

---

### 🚀 나의 풀이 ⭕

```python
import sys 
from collections import deque   # BFS면 deque

# 먼저 '#'을 다 잇는다
# 다 이은 최적 루트는 2가지 경우 밖에 없다.
# 그 중 한 가지를 아무거나 출력하면 된다.

# 상하좌우 조합
dx = [-1,0,1,0]                 
dy = [0,1,0,-1]
directions = ['^','>','v','<']

def check(x,y):
    cnt = 0 
    for i in range(4):  # 상하좌우 4개 
        xx=x+dx[i]
        yy=y+dy[i]      # 상하좌우 좌표 업데이트
        # 업데이트 된 좌표가 격자판을 넘지 않으며 4방향중 다음 위치가 '#'이 있는 위치일때만 
        if 0<=xx<H and 0<=yy<W and maps[xx][yy]=='#':
            start = directions[i]       # 시작할때 어디를 보고 시작하는 지 방향 체크 
            cnt+=1
    if cnt>1:               # 꺽이는 부분이면
        return False
    return start 

def BFS(x,y):
    path = []
    Q = deque()
    Q.append((x,y))
    visited[x][y] = True         # 일단 방문 체크

    while Q:                    # 한쪽 방향 끝날때 까지
        tmp=Q.popleft()         # 시작 위치 좌표 꺼내
        for i in range(4):      # 4방향 확인
            xx=tmp[0]+dx[i]             # 거기에 좌표 업데이트 값
            yy=tmp[1]+dy[i]
            direction=directions[i]     # 그때 방향, 4방향 확인
            # 격자판 안에서 '#'이면서 한번도 방문한적없으면
            if 0<=xx<H and 0<=yy<W and maps[xx][yy]=='#' and visited[xx][yy]==False:
                visited[xx][yy] = True          # 방문 체크 
                path.append(direction)          # 방향 
                Q.append((xx,yy))               # 좌표 
                # -> 4방향을 다 살피며 
    return deque(path) # '#'을 다 이어서 

if __name__=="__main__":
    H, W = map(int, sys.stdin.readline().split())     # 격자판 만들기 
    maps = [list(sys.stdin.readline().rstrip()) for _ in range(H)]
    # sys.stdin = open('input.txt', 'r')
    # H,W=map(int, input().split())
    # maps = [list(sys.stdin.readline().rstrip()) for _ in range(H)]   

    visited = [[False] * W for _ in range(H)]                           # 방문 표 만들기 
    ans = []

    for row in range(H):
        for col in range(W):        # 행, 열 탐색 
            if maps[row][col]=='#' and check(row,col):      # '#'인 경우이자 직선방향이면
                trace=BFS(row,col)                          # 갈 수 있는 최적의 루트가 나온다.
                print(row+1, col+1)                         # 시작 위치 
                # print(trace)                              # 그때 방향들
                print(trace[0])                             # 첫 방향

                # 이때 시작 위치의 로봇 방향
                current = trace.popleft()
                cnt = 1
                # 명령어 출력을 위해 
                for next in trace:
                    if current == next:
                        cnt += 1
                        current = next
                        if cnt % 2 == 0:        # 방향이 2개씩 되어있으면
                            ans.append("A")     # 이동 명령어 'A'
                            cnt = 0   

                    else:   # 그다음 방향이 현재 방향과 일지 하지 않으면 회전인데
                        # 왼쪽이냐 오른쪽이냐
                        if directions[directions.index(current) - 1] == next:
                            ans.append("L")
                        else:
                            ans.append("R")

                        current = next  # 회전수행 후 방향이 바뀐다. 
                        cnt = 1

                for i in ans:
                    print(i, end="") 
                
                # 찾으면 더 수행 할 필요없이 프로그램 끝내기
                sys.exit(0)
```

***
    🌜 개인 공부 기록용 블로그입니다. 오류나 틀린 부분이 있을 경우 
    언제든지 댓글 혹은 메일로 지적해주시면 감사하겠습니다! 😄