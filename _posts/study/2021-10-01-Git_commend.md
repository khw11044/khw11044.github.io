---
layout: post
bigtitle:  "파이썬으로 기본 수학수식 구현하기"
subtitle:   "python"
categories:
    - study
tags:
  - etc
comments: true
published: true
---

# 파이썬으로 기본 수학수식 구현하기

Machine Learning 프로젝트를 작업할 때 코드로 구현해야 하는 다양한 방정식을 접하게 됩니다.  
수학 수식은 개념은 확실히 이해되지만 코드로 표현하기에는 익숙치 않아 구현하기 망설여질때가 있습니다.  

이번 포스트에서는 가장 일반적인 수학 수식을 파이썬의 개념과 연결하여 설명하겠습니다.  
일단 배우면 방정식의 의도를 직관적으로 파악하고 코드를 구현할 수 있을겁니다.  

## Indexing

$$x_i$$

이 수식은 vector의 $$i^{th}$$번째 value(값)을 나타냅니다.  

~~~python
x = [10, 20, 30]
i = 0
print(x[i]) # = 10

for i in range(len(x)):
  print(x[i])
~~~
이것은 2D vectors 등으로 확장될수 있습니다.  
$$x_{ij}$$

~~~python
x = [ [10, 20, 30], [40, 50, 60] ]
i = 0
j = 0
print(x[i][j])  # = 10

for i in range(len(x)):
  for j in range(len(x[i])):
    print(x[i][j])
~~~

## Sigma  

$$\sum^N_{i=1}x_i$$  

다음 수식은 주어진 범위에 대한 vector의 모든 요소의 합을 나타냅니다.  
하한 $$i=1$$부터 상한 $$N$$ 까지를 모두 포함합니다.   
파이썬에서는 index 0부터 index N-1까지 벡터를 반복하는 것과 같습니다.  

~~~python
x = [1, 2, 3, 4, 5]
result = 0
N = len(x)
for i in range(N):
  result += x[i]  # result = result + x[i]
print(result) # = 15
~~~

위 코드는 파이썬내에 built-in functions를 사용하여 다음과 같이 간단하게 나타낼 수 있습니다.  

~~~python
x = [1, 2, 3, 4, 5]
result = sum(x)
~~~

## Average

$$\frac{1}{N}\sum^N_{i=1}x_i$$

여기서 Sigma 표기법을 다시 사용하고 벡터의 원소 개수로 나눠어 평균을 구합니다.  

~~~python
x = [1, 2, 3, 4, 5]
result = 0
N = len(x)
for i in range(N):
  result = result + x[i]
average = result / N
print(average)
~~~

위 코드를 마찬가지로 간단히 나타내면 다음과 같습니다.

~~~python
x = [1, 2, 3, 4, 5]
result = sum(x) / len(x)
~~~

## PI

$$\Pi^N_{i=1}x_i$$

다음 수식은 주어진 범위에 대한 vector의 모든 원소의 곱을 나타냅니다.

~~~python
x = [1, 2, 3, 4, 5]
result = 1
N = len(x)
for i in range(N):
  result *= x[i]  # result = result * x[i]
print(result)
~~~

## Absolute Value

$$\Vert x\Vert$$  
$$\Vert y\Vert$$

다음 수식은 수의 절대값을 나타냅니다.  

~~~python
x = 10
y = -20
abs(x)  # 10
abs(y)  # 20
~~~

## Norm of vector

$$\Vert x\Vert$$  

Norm은 vector의 크기를 계산하는데 사용됩니다.  
파이썬에서는 array의 각 원소를 제곱하여 더한 다음 제곱근을 취하는것을 의미합니다.  

~~~python
import math

x = [1, 2, 3]
math.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
# or
math.sqrt(sum([v**2 for v in x]))
~~~

## Belongs to

$$3 \in X$$

다음 수식은 만일 원소가 집합의 부분집합인지를 확인하는 수식입니다.  
파이썬에서는 다음과 같습니다.  

~~~python
X = {1, 2, 3}
print(3 in X) # True
~~~

## Function

$$f: X \rightarrow Y$$  

다음은 도메인 X를 가져와서 범위 Y에 매핑하는 함수를 나타냅니다.  
파이썬에서는 X 값들을 가져와 계산하여 Y 값들 구하는 작업과 같습니다.  

~~~python
def f(X):
  Y = ...
  return Y
~~~

$$f:R \rightarrow R$$  

R은 input과 output이 실수임을 나타내고 실수 어느값도 될 수 있습니다. (integer, float, irrational, rational).  
파이썬에서 이것은 complex numbers(복소수)를 제외한 어떤 값입니다.  

~~~python
import math
x = 1
y = 2.5
z = math.pi

def f(k):
  k = ...
  return k
~~~

$$f:R^d \rightarrow R$$

$R^d$는 실수의 d-차원 vector를 의미합니다.  
d = 2라고 가정하면 파이썬에서 다음 예시와 같이 2-D list를 입력으로 받고 그것의 합을 return하는 function일 수 있습니다.  
이것은 $R^d$ 에서 $R$로 mapping하는것을 의미합니다.

~~~python
X = [1,2]
f = sum
Y = f(X)
~~~

## Tensor

### Transpose

$$X^T$$

다음 수식은 기본적으로 행과 열을 바꾸는것입니다.  
파이썬에서는 다음과 같습니다.  
~~~python
import numpy as np
X = [[1, 2, 3],
    [4, 5, 6]]
np.transpose(X)
~~~
Output은 행과 열이 바뀐 list로 다음과 같습니다.
~~~
[[1,4],
 [2,5],
 [3,6]]
~~~

### Element wise multiplication  

$$z = x \odot y$$

이 수식은 두개의 tensor에서 해당 원소들끼리의 곱을 의미합니다.  
파이썬에서는 다음과 같이 두개의 list의 해당 원소들을 곱하는것과 같습니다.  

~~~python  
import numpy as np
x = [[1,2],
    [3,4]]
y = [[2,2],
    [2,2]]
z = np.multiply(x,y)
~~~
Output:
~~~
[[2 4]
 [6 8]]
~~~

### Dot Product

$$xy$$  
$$x \cdot y$$

다음 수식은 내적(inner product)라고 하며 dot product라고도 합니다.  
두 벡터를 내적하려면 두 벡터의 차원(길이)가 같아야합니다.

$$x = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}, y = \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix}$$  

$$x^Ty = [1 \; 2 \; 3] \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix} = 1 \cdot 4 + 2 \cdot 5 + 3 \cdot 6 = 32$$

~~~python
x = [1, 2, 3]
y = [4, 5, 6]
dot = sum([i*j for i, j in zip(x, y)])
# 1*4 + 2*5 + 3*6
# 32
~~~

### Hat

$$\hat{x}$$

hat은 단위벡터를 뜻합니다. 이것은 vector의 각 구성원소를 길이(norm)으로 나누는 것을 의미합니다.  

~~~python
import math
x = [1, 2, 3]
length = math.sqrt(sum([e**2 for e in x]))
x_hat = [e/length for e in x]
~~~
이것은 벡터의 방향은 유지하면서 벡터의 크기는 1로 만듭니다.
~~~python
import math
math.sqrt(sum(x**2 for x in x_hat))
# 1.0
~~~

## Exclamation

$$x!$$

이것은 factorial을 나타냅니다. 1부터 그 숫자까지의 곱을 의미합니다.  
파이썬에서는 다음과같이 표현할 수 있습니다.

~~~python
x = 5
fact = 1
for i in range(x, 0, -1):
    fact *= i # fact = fact * i
print(fact)
~~~
이것은 다음 built-in function으로 계산할 수 있습니다.  

~~~python
import math
x = 5
math.factorial(x)
~~~
Output:  
~~~
# 5*4*3*2*1
120
~~~
