---
layout: post
bigtitle:  "Jekyll,Github 블로그에 LaTex, MathJax 적용, 오류해결"
subtitle:   "기존에 mathjax가 적용안되는것을 해결했다."
categories:
    - blog
    - blog-etc
tags:
    - jekyll
date: '2020-12-21 15:45:51 +0900'
comments: true
related_posts:
    - _posts/blog/blog-etc/2020-12-21-setting-start.md
    - _posts/blog/blog-etc/2020-12-21-setting-start2.md
published: true
---

# "Jekyll,Github 블로그에 LaTex, MathJax 적용, 오류해결

# 개요
> 기존에 mathjax가 적용안되거나 되다 안되다하는것을 해결하였다.



## 기존 문제
---

### 안되는 유형 1

가장 대표적으로 이분꺼
![그림1](/assets/img/Blog/Etc/jekyll-Latex/1.JPG)

>[안되는 위 블로그](https://mkkim85.github.io/blog-apply-mathjax-to-jekyll-and-github-pages/)
>[안되는 블로그2](https://jamiekang.github.io/2017/04/28/blogging-on-github-with-jekyll/)
>[안되는 블로그3](https://seongkyun.github.io/others/2019/01/03/MathJax/)
>[안되는 블로그4](https://johngrib.github.io/wiki/mathjax-latex/)

안된다.

뭔가 **_includes**폴더에 새 html파일을 만들어 코드를 넣고 **_includes**폴더든 **_layouts**폴더든 **post.html**, **page.html**, **head.html**, **body.html**에 ==include 하는 구조==는 전부 되다 안되다 한다. (다해봄 다 되다 안되다 개빡침)

처음에는 안되는데 새로고침하면 된다던가 그런식이다.

### 안되는 유형 2

![그림3](/assets/img/Blog/Etc/jekyll-Latex/3.JPG)

![그림2](/assets/img/Blog/Etc/jekyll-Latex/2.JPG)

**_config.yml**에서

이거저것 다운받아서 **_plugin**폴더에 넣고 **plugins**에 선언하는것도 안된다.

**math_engine :** 을 mathjax도 해보고 nil도 해보고 null도 해보고 katex도 해보았다 .

다 안된다.

뭐 다운받고 넣고 파일 만들고 넣고 **Gemfile**에 뭐 추가하고 **bundle install**하고 이틀을 밤새며 알아봤는데 해결 못하다가 알아냈다.


## LaTex
---

이거저것 설명 추가안하고 바로 본론을 말하자면

이전 블로그들은 다 2019년쯤? 2020년초반대인데 뭐 이번부터인지 MathJax버전2가 안된다. 버전3이여야 한다고 하는데 그거마져 쉽지않다. 내 블로그 테마는 _js/src katex.js가 있어서 katex를 알아보았다.

> [KATEX설명](https://katex.org/docs/browser.html)

너무 쉽다... 위 사이트에 들어가서 **Starter template**에 <a>\<head\></a>부분(link1개 script2개)을 복사해서 <a>_includes/head.html</a>에 넣는다.

~~~html
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/katex.min.css" integrity="sha384-AfEj0r4/OFrOo5t7NnNe46zW/tFgW6x/bCJG8FqQCEo3+Aro6EYUG4+cU+KJWu/X" crossorigin="anonymous">
<!-- The loading of KaTeX is deferred to speed up page rendering -->
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/katex.min.js" integrity="sha384-g7c+Jr9ZivxKLnZTDUhnkOnsh30B4H0rpLUpJ4jAIKs4fnJI+sEnkvrMWph2EDg4" crossorigin="anonymous"></script>
<!-- To automatically render math in text elements, include the auto-render extension: -->
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/contrib/auto-render.min.js" integrity="sha384-mll67QQFJfxn0IYznZYonOWZ644AWYC+Pt2cHqMaRhXVrursRwvLnLaebdGIlYNa" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>
~~~

![그림4](/assets/img/Blog/Etc/jekyll-Latex/4.png)

그대로 넣고 확인하면 위 사진과 같이 inline이 안된다.

옵션을 바꿔줘야 한다. render가 현재 auto-render이며 onload="renderMathInElement"부분을 지우고 renderMathInElement()가 body에 들어가야한다.

includes/katex_render.html을 만들어준다. 아래코드를 넣어준다.

~~~html
<script>
      renderMathInElement(
          document.body,
          {
              delimiters: [
                  {left: "$$", right: "$$", display: true},
                  {left: "\\[", right: "\\]", display: true},
                  {left: "$", right: "$", display: false},
                  {left: "\\(", right: "\\)", display: false}
              ]
          }
      );
</script>
~~~

_includes/head.html에 넣어줬던 코드는 지우고 아래 코드로 바꿔준다.

~~~html
<!--mathjax-->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/contrib/auto-render.min.js"></script>
~~~

![그림5](/assets/img/Blog/Etc/jekyll-Latex/5.JPG)

**인라인**(\$)과 **아웃라인**(\$\$) 둘다 잘 적용된것을 확인할수 있다.


$$\lim_{x \to \infty} \exp(-x) = 0$$

$\lim_{x \to \infty} \exp(-x) = 0$

하지만 outline일때 배경이 회색인것은 어떻게 바꾸는지 모르겠다.(그냥 흰색배경이였으면 좋겠는데....)

아시는분 알려주세요.

<kbd>==다음은 LaTex 수식 문법을 배워보자.==</kbd>

[LaTex 수식 문법](https://khw11044.github.io/blog/2020/12/21/markdown-tutorial2.html)

## Reference
---
\>[https://stackoverflow.com/questions/27375252/how-can-i-render-all-inline-formulas-in-with-katex](https://stackoverflow.com/questions/27375252/how-can-i-render-all-inline-formulas-in-with-katex)

\>[https://katex.org/docs/browser.html](https://katex.org/docs/browser.html)

\>[https://chrisyeh96.github.io/2020/03/29/mathjax3.html](https://chrisyeh96.github.io/2020/03/29/mathjax3.html)
