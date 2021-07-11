# warning 제거 
import warnings 
warnings.filterwarnings(action='ignore')

# 글꼴 설정 
from module.Font import Fontmanager
path = Fontmanager()

# module 
import pandas as pd
import missingno as msno
import datetime
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from wordcloud import ImageColorGenerator
from pyecharts.charts import Pie
from pyecharts import options as opts
from pyecharts.render import make_snapshot

from module.rate import rating
from module.BubbleChart import BubbleChart
from module.ColumnNanInfo import ColumnNanInfo
from module.DeleteVal import DeleteVal
from module.WordCount import WordCount

df = pd.read_csv('./data/KoreaMovieData.csv', thousands = ',' )

# data column cleaning 
df.columns = df.columns.str.replace('\n','')

# index order by "순번"
df.set_index("순번", inplace = True)

# Filling missing value : "정보없음"
df = df.fillna("정보없음")

# 서울매출액만 정보 없음 값 0으로 대체
df["서울매출액"][df["서울매출액"] == "정보없음"] = 0 
df["서울매출액"] = df["서울매출액"].astype(int)

# 등급 변경 
raking_df = []
for i in df["등급"]:
    raking_df.append(rating(i))
    
df['등급'] = raking_df

# 일반 영화, 독립 영화 나눠주기 
indiefilm = df[df["영화구분"] == "독립/예술영화"]
film = df[df["영화구분"] == "일반영화"]

ratio = df['영화구분'].value_counts()
labels = df['영화구분'].value_counts().index
wedgeprops = {'width':0.7, 'edgecolor':'w', 'linewidth':5}
colors = ['seagreen','mediumpurple']
explode = [0, 0.10]

# 독립영화, 일반 영화 전체 개봉 작품수 비교 
plt.figure(figsize=(15,8))

plt.pie(ratio, labels=labels, autopct='%.0f%%', startangle=180, counterclock=True,
       wedgeprops=wedgeprops, colors=colors, textprops={'fontsize': 20}, explode = explode)

plt.text(1.4, 1, f"독립영화 작품수 {df['영화구분'].value_counts().values[1]}", fontsize=15)
plt.text(1.4, 0.8, f"일반영화 작품수 {df['영화구분'].value_counts().values[0]}", fontsize=15)

plt.axis('equal')
plt.title('영화 구분에 따른 전체 개봉영화 작품수', fontsize=20)
plt.savefig("./images/작품수(영화구분).png", facecolor='#ffffff')
plt.show()

# 월별 전체 영화 개봉수
# 개봉일 정보 없는 데이터들은 drop
df_open = df[df["개봉일"] != "정보없음"]

# 개봉일에 따른 영화 갯수 파악을 위한 전처리 작업 
df_open["개봉일"] = df_open["개봉일"].str.split("-").str[0] + df_open["개봉일"].str.split("-").str[1]
df_open["개봉일"] = df_open["개봉일"].astype(int)
df_open = df_open[df_open["개봉일"] >= 201101]

# 데이트 타임으로 형 변환 후 그래프화 작업
df_open["개봉일"] = pd.to_datetime(df_open["개봉일"], format="%Y%m")
df_open1 = df_open[df_open["영화구분"] == "일반영화"]
df_open2 = df_open[df_open["영화구분"] != "일반영화"]


# 연도별 전체 영화 개봉작품 수
plt.figure(figsize=(15,8))
plt.rc('font',size=20)

plt.plot(df_open1.groupby(df_open1["개봉일"]).size(),color = 'darkkhaki', linewidth=5, linestyle=':', label='일반영화')
plt.plot(df_open2.groupby(df_open2["개봉일"]).size(),color ='teal', linewidth=5, label='독립/예술영화')

plt.legend(loc=8) 
plt.grid(True, axis='y')
plt.title("월별 전체 영화 개봉작품수", fontsize=20)
plt.xlabel('개봉일')
plt.ylabel('영화 개수')
plt.vlines([17470], 10,200, linestyle = ':')
plt.annotate("", xy=(18700,10),xytext=(17470,125),arrowprops={
    'facecolor':'b', "edgecolor":'b','shrink' : 0.1, 'alpha':0.5 })
plt.savefig("./images/개봉작품수(영화구분).png", facecolor='#ffffff')
plt.show()

# 영화 등급 분포

datas_grade_1 = indiefilm[['등급']].groupby('등급').size().reset_index(name="작품수")
datas_grade_1 = datas_grade_1[datas_grade_1["등급"] != "정보없음"]

datas_grade_2 = film[['등급']].groupby('등급').size().reset_index(name="작품수")
datas_grade_2 = datas_grade_2[datas_grade_2["등급"] != "정보없음"]

datas_grade_1["영화구분"] = "독립/예술영화"
datas_grade_2["영화구분"] = "일반영화"
datas = pd.concat([datas_grade_1, datas_grade_2], 0)

# 독립/예술영화, 일반영화 등급 분포 그래프
plt.figure(figsize=(14,10))
parameters = {'axes.labelsize': 10,
              'axes.titlesize': 10,
              "xtick.labelsize":10,
              "ytick.labelsize":10}

plt.rcParams.update(parameters)
sns.barplot(x = datas['작품수'], y = datas['등급'], hue = datas["영화구분"], palette="rocket_r")
plt.title('전체 영화 등급 분포')
plt.savefig("./images/등급분포(영화구분).png", facecolor='#ffffff')
plt.show()

# 장르 분포
# 독립/예술영화, 일반영화 장르 분포 그래프
datas_genre_1 = indiefilm[['장르']].groupby('장르').size().reset_index(name="count")
datas_genre_2 = film[['장르']].groupby('장르').size().reset_index(name="count")

datas_genre_1["영화구분"] = "독립/예술영화"
datas_genre_2["영화구분"] = "일반영화"
datas_genre = pd.concat([datas_genre_1, datas_genre_2], 0)

interest = ["멜로/로맨스", "다큐멘터리", "애니메이션", "액션", "공포(호러)", "성인물(에로)"]
datas_interest = datas_genre[datas_genre["장르"].isin(interest)]
datas_interest_pivot = datas_interest.pivot(index='장르', columns='영화구분', values='count')

sns.set_palette(sns.color_palette('Dark2'))
datas_interest_pivot.plot.bar(stacked=True, rot=0, figsize=(20,10))
plt.title('전체 영화 장르 분포', fontsize=20)
plt.savefig("./images/장르분포.png", facecolor='#ffffff')
plt.show();


