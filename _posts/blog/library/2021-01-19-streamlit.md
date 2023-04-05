---
layout: post
bigtitle: "streamlit"
subtitle: 'streamlit 라이브러리'
categories:
    - blog
    - library
tags:
    - streamlit
    - python
date: '2021-01-19 02:45:51 +0900'
comments: true
published: true
---

# streamlit

## 개쩌는 파이썬 라이브러리 발견

---

* toc
{:toc}

## streamlit

데이언트 사이언스 웹앱을 단 몇분만에 만들수있는 파이썬 라이브러리를 알게 되었다.

단순히 import streamlit 을 하고

터미널 창에 streamlit run 파일이름.py만 해주면 자신의 로컬에서 웹으로 깔끔하게 디자인된 데이터 분석 이미지가 들어있는 웹을 볼수 있다.

그전에는 저장해둔 colab이나 matplotlib으로 보인 이미지를 저장하여 ppt등 첨부하였지만 이제는 아예 웹으로 보여줄수 있다!!


**streamlit 사이트** :   
> [streamlit 사이트](https://www.streamlit.io/)




## 웹 deploy하기

하지만 로컬에서만 볼수 있다면 또 보여주기 위해서 터미널을 키고 명령어를 쳐야한다면.... 굳이? 뭐하러? 라는 생각이 듣다.

나는 어디서든 현재 이 포스팅에서도 내가 만든것을 주소 링크를 첨부해서 보여주고 싶다.

무료로 내 웹을 Deploy해보자.

![그림1](/assets/img/Blog/library/streamlit01.JPG){: height="500px"}
![그림2](/assets/img/Blog/library/streamlit02.JPG){:  height="500px"}


<br>

### 내가 만든 페이지이다. 확인 ㄱ

[https://btcsamsungapp.herokuapp.com/](https://btcsamsungapp.herokuapp.com/)

단 한 파일로 몇줄의 코드로 구현할수 있다는게 정말 놀랍다.

~~~python
import streamlit as st
from cryptocmd import CmcScraper
import pandas_datareader as pdr
import plotly.express as px
from datetime import datetime, timedelta

st.write('# 비트코인 데이터 Web app')

st.sidebar.header('Menu')

name = st.sidebar.selectbox('Name', ['BTC', 'ETH', 'USDT'])

sevendayago = datetime.today() - timedelta(7)

start_date = st.sidebar.date_input('Start date', sevendayago)
end_date = st.sidebar.date_input('End date', datetime.today())

# https://coinmarketcap.com
scraper = CmcScraper(name, start_date.strftime('%d-%m-%Y'), end_date.strftime('%d-%m-%Y'))

df = scraper.get_dataframe()

fig_close = px.line(df, x='Date', y=['Open', 'High', 'Low', 'Close'], title='가격')
fig_volume = px.line(df, x='Date', y=['Volume'], title='Volume')

st.plotly_chart(fig_close)
st.plotly_chart(fig_volume)



st.write('''
# 삼성전자 주식 데이터
마감 가격과 거래량을 차트로 보여줍니다!
''')

# https://finance.yahoo.com/quote/005930.KS?p=005930.KS
dr = pdr.get_data_yahoo('005930.KS',start_date,end_date)

st.line_chart(dr.Close)
st.line_chart(dr.Volume)
~~~

위에는 코드이다. 정말 이것만 있으면 로컬에서는 볼수있다. 로컬에서 보는것도 놀랍긴하다.

하지만 말했듯이 나는 무료로 deploy하고 싶다

그렇다면 아래 첨부 영상을 보면 된다.


[Heroku](https://dashboard.heroku.com/)

<br>


<br>

필요한 파일들은 내 git repo에서 다운받으면 되겠다.

> [내 레파지토리](https://github.com/khw11044/second-streamlit-app)

## Reference
---
[빵형의 개발도상국](https://www.youtube.com/watch?v=JLVB8ZUPojw)
