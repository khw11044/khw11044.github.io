---
layout: post
bigtitle:  "마크다운 문법정리2_수식"
subtitle:   "나를 위한 Markdown 작성법2(수학수식)"
categories:
    - blog
    - blog-etc
tags:
    - markdown
date: '2020-12-21 13:45:51 +0900'
comments: true
related_posts:
    - _posts/blog/blog-etc/2020-12-21-jekyll-Latex.md
    - _posts/blog/blog-etc/2020-12-21-setting-start.md
    - _posts/blog/blog-etc/2020-12-21-setting-start2.md
    - _posts/blog/blog-etc/2020-12-21-setting-atom-anaconda.md

---

# 마크다운 문법정리2_수학수식, 특수문자

## 개요
> 내가 보기위한 마크다운 `Markdown` 작성법2 수식  

### katex 기호 모음
> [katex 기호 모음](https://jjycjnmath.tistory.com/117)

> [여기 다 있음](https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:TeX_%EB%AC%B8%EB%B2%95)

* toc
{:toc}

# Math
---

이 장에 관심이 있다면 논문, 학술자료, 공학, 교사에 관련된 직업일 확률이 높습니다.
글 또는 그림과 다르게 수학 기호와 수식은 그래프의 표현은 책을 집필할 때 어려움이 됩니다. 이 상황에서 Latex를 사용하면 유용합니다.
이 책은 LaTex, WebTex, MathML을 소개합니다. LaTex를 심도있게 다루고 나머지 솔루션은 소개정도만 다룹니다.
그럼 문서를 생성할 때 어떻게 수학의 기호나 그래프를 그리는지 알아보겠습니다.

## LaTex
---
LaTex는 오픈소스 조판시스템(Typesetting System)입니다.
조판작업이란 최종 결과물이 출력이 되기전에 출력될 결과물에 맞게 도형,수식,글을 배치하는 작업입니다.
LaTex를 이용해서 모든 형태를 그리고 배치할 수 있지만 대부분 수식, 그래프 작업이 필요할 때 일반적으로 많이 사용합니다.
또한 물리학자들을 위한 학술 커뮤니케이션 언어로 많이 사용됩니다. 어렵고 이상해 보이지만 35년이상 사용되면서 필요한 표기를 잘 처리할 수 있었습니다.
수식을 표현하는 솔루션은 LaTex, WebTex, MathML .. 등등 굉장히 많습니다.
수식을 모든 문서에 표기하기 위해서는 비효율적이지만 LaTex문법을 이용해서 이미지로 렌더링 후 문서에 첨부하는 형태를 많이 사용합니다.
불편하지만 다른 문서포멧으로 컨버팅 되더라도 문제없이 표시되기 때문입니다.
만약 공유와 협업이 필요한 상황에서는 [ShareLatex](https://www.sharelatex.com)를 사용하면 인터넷에서 친구들과 LaTex문서를 작성할 수 있습니다.

## Jekyll&Github Blog에서 LaTex 적용
---

### MathJax

<code>_includes</code>폴더에 <a>Mathjax.html</a>를 만들고 아래 코드를 넣는다.



## LaTex문법
---

이 챕터에서는 컴퓨터로 수학기호를 표현할 수 있는 LaTex의 수식표현 문법을 배워보겠습니다.
이 책은 Pandoc책이지만 Pandoc 만큼이나 LaTex를 조금 자세히 다루어 보는 책 입니다.
수학기호를 책에 넣기위해서 노력하다 보면 LaTex를 잘 다루고 있는 자료를 찾기위해서 매번 인터넷을 해메이기 때문입니다.
LaTex의 많은 문법들은 \\문자로 시작합니다.

### 1사칙연산
---

덧셈에 대한 표현입니다. 우리가 사용하는 표현과 크게 다르지 않습니다.

$1 + 1 = 2$

$$1 + 1 = 2$$

```
$$1 + 1 = 2$$
```

뺄셈의 표현

$$2 - 1 = 1$$

```
$$2 - 1 = 1$$
```

곱셈의 표현

$$2 \times 2 = 4$$

```
$$2 \times 2 = 4$$
```

나눗셈의 표현

$$4 \div 2 = 2$$

```
$4 \div 2 = 2$
```

### 2분수 / fraction
분수 영어로는 fraction. LaTex에서는 약자로 frac이라는 문법을 사용합니다.

$$\frac{1}{2}$$

```
$\frac{1}{2}$
```

#### 요리에 자주 사용하는 분수
우리가 사용하는 기호중에 간장 1큰술반 같은 표현식도 분수 표기법중에 하나입니다. 요리책에 많이나오는 형태의 분수도 입력해보겠습니다.

$$^1/_2$$

```
$^1/_2$
```

### 수학 공식, 수식 번호

뒤에 \tag{1}, \tag{2} 이런식으로 붙이면 된다.

$$X_{1,j} \mathbf{F}X_{2,j}  = 0, \tag{1}$$

~~~
$$X_{1,j} \mathbf{F}X_{2,j}  = 0, \tag{1}$$
~~~

### 3괄호, 중괄호, 대괄호
LaTex에서 괄호를 사용하는 방법을 배워보겠습니다.

#### 소괄호

$$(1+2)$$

```
$$(1+2)$$
```

#### 중괄호

$$\{1+2\}$$

```
$$\{1+2\}$$
```

#### 대괄호

$$[1+2]$$

```
$$[1+2]$$
```

#### 자동 괄호 리사이즈
자동으로 괄호의 리사이즈가 되기 위해서는  "left", "right" 문자를 좌우로 넣어줍니다.

$$\left(\frac{2}{3}\right)$$

```
$$\left(\frac{2}{3}\right)$$
```

#### 수동 괄호 리사이즈
big, Big, bigg, Bigg 문자를 사용하면 수동으로 괄호 사이즈를 조절할 수 있습니다.

$$\Bigg( \bigg( \Big( \big( ( ) \big) \Big) \bigg) \Bigg)$$

```
$$\Bigg( \bigg( \Big( \big( ( ) \big) \Big) \bigg) \Bigg)$$
```

### 4위첨자 지수 / Power
승, 제곱에 해당하는 표기를 위해서 ^ 문자를 사용합니다.

$$2^2=4$$

```
$$2^2=4$$
```

### 5아래첨자 / Indices
각 아이템의 아래첨자는 _ 문자로 표기합니다. _문자는 이후에 극한, 시그마, 적분표기 처럼 아래에 표기해야 하는 수식에서도 활용됩니다.

$$a_1, a_2, a_3$$

```
$$a_1, a_2, a_3$$
```

### 6dots
점을 출력하는 방법을 알아보겠습니다.
수와 수 사이에 값이 존재하는 상황에서 생략적 표기로 점을 많이 사용합니다.

$$\dots$$

```
$$\dots$$
```

가운데를 기준으로 점을 표기합니다.

$$\cdots$$

```
$$\cdots$$
```

새로로 점을 표기하는 방법입니다. 세로형태의 행렬, 매트릭스 내부에서 활용합니다.

$$\vdots$$

```
$$\vdots$$
```

대각선을 점을 표시하는 방법입니다. 행렬, 매트릭스를 표기할 때 대각선방향에 활용합니다.

$$\ddots$$

```
$$\ddots$$
```

### 7루트(거듭제곱근) / Root
루트. square root의 약자로 sqrt 라고 사용합니다.

$$\sqrt{2}$$

```
$$\sqrt{2}$$
```

### 8펙토리얼 / Factorial
!문자는 수학에서 팩토리얼을 뜻합니다.
3!이라는 뜻은 3,2,1 숫자로 만들수 있는 경우의 수 가 몇개인지를 뜻합니다.
3!은 (3,2,1),(3,1,2)(2,1,3)(2,3,1),(1,2,3)(1,3,2) 총 6개의 경우의 수가 존재합니다.
3x2x1=6 으로 계산했을때 결과와 경우의 수는 같습니다.

$$n!$$

$$n! = 1 \times 2 \times 3 \times \ldots n$$

~~~
$$n!$$

$$n! = 1 \times 2 \times 3 \times \ldots n$$
~~~

펙토리얼을 Product 표기법을 사용해서 표현하면 아래와 같습니다.

$$n! = \prod_{k=1}^n k$$

~~~
$$n! = \prod_{k=1}^n k$$
~~~


### 9집합 / Set
LaTex에서 집합의 표기법을 다루어 보겠습니다.

#### 합집합

$$\{a,b,c\} \cup \{d,e\} = \{a,b,c,d,e\}$$

```
$$\{a,b,c\} \cup \{d,e\} = \{a,b,c,d,e\}$$
```

#### 교집합
위 표기의 역표기.

$$\{a,b,c\} \cap \{a,b,d\} = \{a,b\}$$

```
$$\{a,b,c\} \cap \{a,b,d\} = \{a,b\}$$
```

#### 차집합
B-A=파이 공집함

#### 집합의 역
문자 위에 C 달린것

#### 포함된다
$$x \in [-1,1]$$

~~~
$$x \in [-1,1]$$
~~~

### 10삼각함수, 싸인, 코싸인, 탄젠트, 세타
$$\cos (2\theta) = \cos^2 \theta - \sin^2 \theta$$

~~~
$$\cos (2\theta) = \cos^2 \theta - \sin^2 \theta$$
~~~

#### 파이 / Pi
$$\pi$$

$$\Pi$$

$$\phi$$

~~~
$$\pi$$

$$\Pi$$

$$\phi$$
~~~


#### 각도
90도

$$90^\circ$$

~~~
$$90^\circ$$
~~~

### 11극한, limit
$$\lim_{x \to \infty} \exp(-x) = 0$$

~~~
$$\lim_{x \to \infty} \exp(-x) = 0$$
~~~

### 12시그마,for
$$\sum_{i=1}^{10} t_i$$

$$\displaystyle\sum_{i=1}^{10} t_i$$

~~~
$$\sum_{i=1}^{10} t_i$$

$$\displaystyle\sum_{i=1}^{10} t_i$$
~~~

### 13로그 / log

$$\log_b a$$

```
$$\log_b a$$
```

### 14 미분 / differential

$$\dv{Q}{t} = \dv{s}{t}$$

```
$$\dv{Q}{t} = \dv{s}{t}$$
```


### 15 적분 / integral
$$\int_0^\infty \mathrm{e}^{-x}\,\mathrm{d}x$$

$$\int\limits_a^b$$

~~~
$$\int_0^\infty \mathrm{e}^{-x}\,\mathrm{d}x$$

$$\int\limits_a^b$$
~~~

### 16 행렬 / matrix

$$A_{m,n} =
 \begin{pmatrix}
  a_{1,1} & a_{1,2} & \cdots & a_{1,n} \\
  a_{2,1} & a_{2,2} & \cdots & a_{2,n} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  a_{m,1} & a_{m,2} & \cdots & a_{m,n}
 \end{pmatrix}$$

~~~
$$A_{m,n} =
 \begin{pmatrix}
  a_{1,1} & a_{1,2} & \cdots & a_{1,n} \\
  a_{2,1} & a_{2,2} & \cdots & a_{2,n} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  a_{m,1} & a_{m,2} & \cdots & a_{m,n}
 \end{pmatrix}$$
~~~

$ \begin{bmatrix}
a & b \\\\
c & d
\end{bmatrix}$

~~~
$ \begin{bmatrix}
a & b \\\\
c & d
\end{bmatrix}$
~~~

### 17 벡터, 스칼라 / Vector, Scalar
$$\overrightarrow{AB}$$

$$\overline{AB}$$

~~~
$$\overrightarrow{AB}$$

$$\overline{AB}$$
~~~

### 18선그리기

$$\setlength{\unitlength}{3cm}
\begin{picture}(1,1)
\put(0,0){\line(1,0){1}}
\put(0,0){\line(0,1){1}}
\end{picture}$$

~~~
$$\setlength{\unitlength}{3cm}
\begin{picture}(1,1)
\put(0,0){\line(1,0){1}}
\put(0,0){\line(0,1){1}}
\end{picture}$$
~~~

### 좌표그리기
선을 그리는 방법을 응용하면 좌표를 그릴 수 있습니다. 또한 각진 도형들도 그릴 수 있습니다.

#### 원그리기

$$\setlength{\unitlength}{1mm}
\begin{picture}(60, 40)
\put(30,20){\circle{10}}
\end{picture}$$

```
\setlength{\unitlength}{1mm}
\begin{picture}(60, 40)
\put(30,20){\circle{10}}
\end{picture}
```

### 벡터(화살표) 그리기
```
\setlength{\unitlength}{0.75mm}
\begin{picture}(60,40)
\put(30,20){\vector(1,0){30}}
\put(30,20){\vector(4,1){20}}
\put(30,20){\vector(3,1){25}}
\put(30,20){\vector(2,1){30}}
\put(30,20){\vector(1,2){10}}
\thicklines
\put(30,20){\vector(-4,1){30}}
\put(30,20){\vector(-1,4){5}}
\thinlines
\put(30,20){\vector(-1,-1){5}}
\put(30,20){\vector(-1,-4){5}}
\end{picture}
```

$$\setlength{\unitlength}{0.75mm}
\begin{picture}(60,40)
\put(30,20){\vector(1,0){30}}
\put(30,20){\vector(4,1){20}}
\put(30,20){\vector(3,1){25}}
\put(30,20){\vector(2,1){30}}
\put(30,20){\vector(1,2){10}}
\thicklines
\put(30,20){\vector(-4,1){30}}
\put(30,20){\vector(-1,4){5}}
\thinlines
\put(30,20){\vector(-1,-1){5}}
\put(30,20){\vector(-1,-4){5}}
\end{picture}$$

### 논문에 자주 나오는 기호

**특수문자**

| 이름 | 명령어 | 반환 | | 이름 | 명령어 | 반환|
| ---- | ------ | ---- | -- | --- | --- | --- |
| 알파 | \alpha | $$\alpha$$ | | 크사이  | \xi  | $$\xi$$ |
| 베타 | \beta | $$\beta$$ | |  오미크론 | o  | $$o$$ |
| 감마 | \gamma | $$\gamma$$ | | 파이 | \pi  | $$\pi$$ |
| 델타 | \delta | $$\delta$$ | | 로 | \rho | $$\rho$$ |
| 엡실론 | \epsilon | $$\epsilon$$ | | 시그마 | \sigma | $$\sigma$$ |
| 제타 | \zeta | $$\zeta$$ | | 타우  | \tau  | $$\tau$$ |
| 에타 | \eta | $$\eta$$ | | 입실론  | \upsilon  | $$\upsilon$$ |
| 세타 | \theta | $$\theta$$ | | 파이  | \phi | $$\phi$$ |
| 이오타 | \iota | $$\iota$$ | | 카이  | \chi | $$\chi$$ |
| 카파 | \kappa | $$\kappa$$ | | 오메가  | \omega | $$\omega$$ |
| 람다 | \lambda | $$\lambda$$ | | 뉴  | \nu | $$\nu$$ |
| 뮤 | \mu | $$\mu$$ | |   |   | |



**관계연산자**

| 이름 | 명령어 | 반환 | | 이름 | 명령어 | 반환|
| ---- | ------ | ---- | -- | --- | --- | --- |
| 합동 | \equiv | $$\equiv$$ | | 근사  | \approx | $$\approx$$ |
| 비례 | \propto | $$\propto$$ | | 같고 근사  | \simeq | $$\simeq$$ |
| 닮음 | \sim | $$\sim$$ | | 같지 않음  | \neq | $$\neq$$ |
| 작거나 같음 | \leq | $$\leq$$ | | 크거나 같음  | \geq  | $$\geq$$ |
| 매우작음 | \ll | $$\ll$$ | | 매우 큼 | \gg  | $$\gg$$ |


**논리기호**

| 이름 | 명령어 | 반환 | | 이름 | 명령어 | 반환|
| ---- | ------ | ---- | -- | --- | --- | --- |
| 불릿 | \bullet | $$\bullet$$ | | 부정 | \neq  | $$\neq$$ |
| wedge | \wedge | $$\wedge$$ | | vee | \vee | $$\vee$$ |
| 논리합 | \oplus | $$\oplus$$ | | 어떤 | \exists | $$\exists$$ |
| 오른쪽 </br>화살표 | \rightarrow | $$\rightarrow$$ | | 왼쪽 <\br>화살표 | \leftarrow  | $$\leftarrow$$ |
| 왼쪽 <\br>큰화살표 | \Leftarrow | $$\Leftarrow$$ | | 오른쪽 <\br>큰화살표 | \Rightarrow | $$\Rightarrow$$ |
| 양쪽 <\br>큰화살표 | \Leftrightarrow | $$\Leftrightarrow$$ | | 양쪽 <\br>화살표  | \leftarrow | $$\leftarrow$$ |
| 모든 | \forall | $$\forall$$ | |   |   |  |


**집합기호**

| 이름 | 명령어 | 반환 | | 이름 | 명령어 | 반환|
| ---- | ------ | ---- | -- | --- | --- | --- |
| 교집합 | \cap | $$\cap$$ | | 합집합 | \cup | $$\cup$$ |
| 상위집합 | \supset | $$\supset$$ | | 진상위집합 | \supseteq | $$\supseteq$$ |
| 하위집합 | \subset | $$\subset$$ | | 진하위집 | \subseteq | $$\subseteq$$ |
| 부분집합아님 | \not\subset | $$\not\subset$$ | | 공집합 | \emptyset, \varnothing | $$\emptyset$$ $$\varnothing$$ |
| 원소 | \in | $$\in$$ | | 원소아님  | \notin  | $$\notin$$ |


**기타**

| 이름 | 명령어 | 반환 | | 이름 | 명령어 | 반환|
| ---- | ------ | ---- | -- | --- | --- | --- |
| hat | \hat{x} | $$\hat{x}$$ | | widehat  | \widehat{x} | $$\widehat{x}$$ |
| 물결 | \tilde{x} | $$\tilde{x}$$ | | wide물결 | \widetilde{x} | $$\widetilde{x}$$ |
| bar | \bar{x} | $$\bar{x}$$ | | overline | \overline{x} | $$\overline{x}$$ |
| check | \check{x} | $$\check{x}$$ | | acute | \acute{x} | $$\acute{x}$$ |
| grave | \grave{x} | $$\grave{x}$$ | | dot | \dot{x} | $$\dot{x}$$ |
| ddot | \ddot{x} | $$\ddot{x}$$ | | breve | \breve{x} | $$\breve{x}$$ |
| vec | \vec{x} | $$\vec{x}$$ | | 델,나블라  | \nabla  | $$\nabla$$ |
| 수직 | \perp | $$\perp$$ | | 평행 | \parallel | $$\parallel$$ |
| 부분집합아님 | \not\subset | $$\not\subset$$ | | 공집합 | \emptyset | $$\emptyset$$ |
| 가운데 점 | \cdot | $$\cdot$$ | | ... | \dots | $$\dots$$ |
| 가운데 점들 | \cdots | $$\cdots$$ | | 세로점들 | \vdots | $$\vdots$$ |
| 나누기 | \div | $$\div$$ | | 물결표 | \sim | $$\sim$$ |
| 플마,마플 | \pm, \mp | $$\pm$$ $$\mp$$ | | 겹물결표 | \approx | $$\approx$$ |
| prime | \prime | $$\prime$$ | | 무한대 | \infty | $$\infty$$ |
| 적분 | \int | $$\int$$ | | 편미분 | \partial | $$\partial$$ |
| 한칸띄어 | x \, y | $$x\,y$$ | | 두칸 | x\;y  | $$x \; y$$ |
| 네칸띄어 | x \quad y | $$x \quad y$$ | | 여덟칸띄어 | x \qquad y  | $$x \qquad y$$ |


#### LaTex 참고자료
아래 링크를 참고하시면 더 많은 LaTex정보를 보실 수 있습니다.

- LaTex 문법을 쉽게 익힐 수 있는 곳은 아래 사이트입니다.
	https://en.wikibooks.org/wiki/LaTeX/Mathematics

- LaTex를 이용해서 많은것을 그릴 수 있습니다. 아래 URL에서 구경해보세요.
	https://en.wikibooks.org/wiki/LaTeX/Picture


## WebTex
LaTex는 종이 인쇄물 기반의 기술입니다. 현대 과학의 대부분의 정보는 Web으로 표기됩니다.
WebTex 프로젝트는 LaTex를 html문서로 문제없이 컴파일 하는 것을 목표로 두고 있습니다.
문법은 LaTex와 거의 같습니다.
만약 여러분이 마크다운에서 WebTex 문법을 사용했다면 epub 파일을 제작할 때 아래 옵션을 추가해주면 됩니다.

	--webtex

WebTex 문법은 "$$수식$$" 형태로 구성되어 있습니다.

```
f(x)=\sum_{n=0}^\infty\frac{f^{(n)}(a)}{n!}(x-a)^n
```

위 문장이 문제없이 잘 처리되었다면 epub문서에 아래와 같은 수식이 그려집니다.

$$f(x)=\sum_{n=0}^\infty\frac{f^{(n)}(a)}{n!}(x-a)^n$$


#### 참고자료
- 프로젝트 사이트 : [http://pkgw.github.io/webtex/](http://pkgw.github.io/webtex/)
- 프로젝트 코드 : [https://github.com/pkgw/webtex/](https://github.com/pkgw/webtex/)

## MathML
Mathematical Markup Language의 약자입니다.
epub3 에서 수학기호를 표현하는 방법중 하나입니다.
XML 용용기술이며, HTML5 기술의 하나입니다.
개인적으로 LaTex문법이 짧고 함축적이고 가독성이 더 좋다고 생각합니다.
MathML은 여러분이 사용하는 브라우저가 지원할 수도 있고 지원하지 않을수도 있습니다.
브라우저에 따라 결과가 다를 수 있습니다.
파이어폭스, 사파리에서는 정상적으로 값이 출력되지만, 크롬에서는 아직 지원이 되지 않습니다.


#### MathML의 작성

2차방정식에 대한 근의 공식에 해당하는 MathML의 문법입니다.

    <math>
    <mi>x</mi>
    <mo>=</mo>
    <mfrac>
      <mrow>
      <mrow>
        <mo>-</mo>
        <mi>b</mi>
      </mrow>
      <mo>&PlusMinus;</mo>
      <msqrt>
        <msup>
        <mi>b</mi>
        <mn>2</mn>
        </msup>
        <mo>-</mo>
        <mrow>
        <mn>4</mn>
        <mo>&InvisibleTimes;</mo>
        <mi>a</mi>
        <mo>&InvisibleTimes;</mo>
        <mi>c</mi>
        </mrow>
      </msqrt>
      </mrow>
      <mrow>
      <mn>2</mn>
      <mo>&InvisibleTimes;</mo>
      <mi>a</mi>
      </mrow>
    </mfrac>
    </math>

xml 문법을 사용하기 때문에 많은 태그가 작성되어야 합니다.
일반적으로 LaTex 문법이 더 간단하기 때문에
LaTex로 수식을 작성하고 MathML로 바꾸어주는 작업이 더 쉽습니다.

LaTex to MathML 컨버팅을 지원하는 사이트입니다.
- [https://www.mathtowebonline.com](https://www.mathtowebonline.com)
