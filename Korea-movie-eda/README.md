[![Python Badge](http://img.shields.io/badge/-Python%20-blue?style=flat-square&fontColor&logoColor=yellow&logo=python&link=https://www.python.org/)](https://www.python.org/) [![Numpy Badge](http://img.shields.io/badge/-Numpy%20-013243?style=flat-square&&logoColor=white&logo=numpy&link=https://numpy.org/)](https://numpy.org/) [![Pandas Badge](http://img.shields.io/badge/-Pandas%20-150458?style=flat-square&logoColor=white&logo=pandas&link=https://pandas.pydata.org/)](https://pandas.pydata.org/) [![Matplotlib Badge](http://img.shields.io/badge/-Matplotlib%20-2350A9?style=flat-square&logoColor=white&logo=matplotlib&link=https://matplotlib.org/)](https://matplotlib.org/) [![Seaborn Badge](http://img.shields.io/badge/-Seaborn%20-212E50?style=flat-square&logoColor=white&logo=seaborn&link=https://seaborn.pydata.org/)](https://seaborn.pydata.org/) [![Plotly Badge](http://img.shields.io/badge/-Plotly%20-3F4F75?style=flat-square&logoColor=white&logo=plotly&link=https://plotly.com/)](https://plotly.com/) [![Pyecharts Badge](http://img.shields.io/badge/-Pyecharts%20-34E0A1?style=flat-square&logoColor=black&logo=pyecharts&link=https://pyecharts.org/)](https://pyecharts.org/) [![Jupyter Badge](http://img.shields.io/badge/-Jupyter%20-orange?style=flat-square&&logoColor=white&logo=jupyter&link=https://jupyter.org/)](https://jupyter.org/)

 #### 역활 분담 : 
 > 정솔 : 데이터 전처리, 데이터 시각화
 >
 > 정현석 : 데이터 전처리, 리팩토링


<h1><b> KoreaMovie_EDA </b></h1>

- 극장 뿐만 아니라 OTT 서비스에서도 독립/예술 영화 서비스 경쟁 진행 중
- 이는 지난 십년 동안 한국에서도 독립 영화 시장이 커져갔고 이에 따라 **‘영화’** 라는 단어의 의미 확장이 진행되었다고 생각

- 전체적인 영화 산업에 대한 분석 진행 
- 추가적으로 일반영화와 독립/예술영화을 기준으로 분류하여 비교 분석


<br>
<h2><b> 데이터 소개 </b></h2>
<p align="center">
<img src = 'https://postfiles.pstatic.net/MjAyMTA0MjBfMjcg/MDAxNjE4OTI1MzM1NDQ5.1ycZuuuOgnXxm3xfIgNEObKuHgBdWb_JYLYlTBz_HKMg.hFEzo5R39BQ37r6C5w8VLVGcHn8kMJgPhjHjodPtrQEg.PNG.greatsol98/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7_2021-04-18_%EC%98%A4%ED%9B%84_8.01.22.png?type=w966' width='700'></p>

- 2011년 1월 부터 2021년 4월 9일까지 개봉한 전체 영화 일람 데이터 

> **출처 : 한국영화진흥위원회 박스오피스 https://www.kofic.or.kr/kofic/business/infm/introBoxOffice.do**

<br>
<h2><b> 결론 </b></h2>

<br>
<p align='center'><img src='https://user-images.githubusercontent.com/72845895/121806283-2e57ef00-cc8a-11eb-96b7-62217f4ab107.png'></p>
<br>

- 일반 영화가 4개의 작품이 개봉할 때 독립/예술 영화는 1개의 작품만 개봉

<br>
<p align='center'><img src='https://user-images.githubusercontent.com/72845895/121806270-28620e00-cc8a-11eb-99ba-ed598dbd9df3.png'></p>
<br>

- 2018년부터 독립/예술 영화의 지속적인 감소세가 보이고 있었으며 2019년에는 예년 수준으로 잠시 회복했으나 2020년 코로나로 인하여 다시 또 하락

<br>
<p align="center"> <img src='https://user-images.githubusercontent.com/72845895/121806278-2d26c200-cc8a-11eb-9602-8bf4752a0a98.png' width='415'> <img src='https://user-images.githubusercontent.com/72845895/121806287-3021b280-cc8a-11eb-9138-74e6ef904df3.png' width='415'> <img src='https://user-images.githubusercontent.com/72845895/121806279-2dbf5880-cc8a-11eb-94f0-51377412d4a8.png'></p>
<br>

- 전체 개봉 영화 중에는 청소년 관람불가 작품이 많고 장르적으로는 독립/예술 영화는 다큐멘터리, 일반 영화는 멜로/로맨스 작품이 가장 많이 개봉
- 특이하게 일반영화에서는 성인물 작품 수가 많음 

<br>
<p align="center"> <img src='https://user-images.githubusercontent.com/72845895/121806296-32840c80-cc8a-11eb-9013-0de07d06b5c4.png' width='415'> <img src='https://user-images.githubusercontent.com/72845895/121806292-31eb7600-cc8a-11eb-9b54-6805641d530b.png' width='415'> <img src='https://user-images.githubusercontent.com/72845895/121806285-2ef08580-cc8a-11eb-81b0-fa6eba5b6d7f.png'></p>
<br>


- 매출액이 높은 영화의 장르는 드라마, 범죄, 사극 순으로 많이 분포하였고 12세와 15세이상 관람가 작품이 많음
- 그러나 장르와 등급의 상관관계에서는 액션과 15세이상 관람가, 액션과 12세이상 관람가, 애니메이션과 전체관람가가 가장 유의미하게 상관이 있다는 결과가 나옴

<br>
<p align='center'> <img src='https://user-images.githubusercontent.com/72845895/121806277-2c8e2b80-cc8a-11eb-946e-c3a80edf4c2a.png'></p>
<br>

- 독립/예술 영화의 서울 관객수와 전국 관객수는 크게 차이가 나지 않음

<br>
<p align="center"> <img width="415" height='300' alt="rosechart1" src="https://user-images.githubusercontent.com/72845895/121806288-30ba4900-cc8a-11eb-9c0d-3ca90876e5e8.png"><img width="415" height='300' alt="rosechart2" src="https://user-images.githubusercontent.com/72845895/121806291-3152df80-cc8a-11eb-89ea-fdbf9162fb45.png"></p>
<br>


- 유독 멜로/로맨스 작품에 청소년 관람불가 작품이 많았던 이유는 실제적으로는 성인물(에로) 작품이 멜로/로맨스 장르로 구분되어 있기 때문
- 이런 영화들은 주로 스크린수가 1개로 잡혀있는데 이것을 “형식 개봉 영화” 라고 하고 한국과 일본 국적을 가진 영화가 가장 많음

<br>
<p>
<img src='https://user-images.githubusercontent.com/72845895/121806299-33b53980-cc8a-11eb-87ae-7c3280329bce.png' width='415'><img src='https://user-images.githubusercontent.com/72845895/121806298-331ca300-cc8a-11eb-80b9-b2668d59aeb6.png' width='415'></p>
<br>


- 국내 독립/예술 영화는 관객들에게 감성적으로 어필



<br>
<h2><b> 과정 </b></h2>

- 모듈
> - rate.py : 영화 등급 분류

<img src ='https://postfiles.pstatic.net/MjAyMTA0MjBfNTcg/MDAxNjE4OTI2Mjc2NDk4.CPVi8k_VRC_K296mTCFMcKwAMhext4GEvAlRdU3htrYg.NwslRAo0A0qvIY95SXM2K8sStzjlSEgyOveNnJH2qwsg.PNG.greatsol98/rate.png?type=w966'>

> - wordcount.py : 가장 많이 사용된 20개의 단어 추출

<img src = 'https://postfiles.pstatic.net/MjAyMTA0MjBfNzMg/MDAxNjE4OTI2Mjc2MjEy.jWPEqqvz9mkC03BdX4e4MGq6EWhqJOeYXHYjK-cVWUcg.1drQ_DQny32hBPFGb4Xq0xQBbGmKKbQwXP0KyTHr13Eg.PNG.greatsol98/wordcount.png?type=w966'>

<br>
<h2><b> 개선사항 </b></h2>

- 아웃라이어 제거 후 EDA 진행

<br>
<h2><b> 참고 </b></h2>

- <독립 영화에 대한 현실적인 고민> 부산일보 : http://weekly.pusan.ac.kr/news/articleView.html?idxno=10499
- <절반으로 뚝 떨어진 독립영화 관객... 한국영화의 '위기'> 오마이뉴스 : http://star.ohmynews.com/NWS_Web/OhmyStar/at_pg.aspx?CNTN_CD=A0002512932&CMPT_CD=P0010&utm_source=naver&utm_medium=newsearch&utm_campaign=naver_news
- <서울 사람들이 지방 사람들보다 외국영화를 더 좋아한다고?> 경향신문 : https://www.khan.co.kr/culture/movie/article/201310141616491#csidx8d78a29a592e1c4bdd413f232e29a8e
- <영화 제목으로 본 영한 번역기법 연구: 영화 산업 정책과 언어 정책의 변화를 중심으로> 권유진, 2020
