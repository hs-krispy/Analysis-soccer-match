## EDA

승, 무, 패 세 가지 경우에 대해 각각 특정 column 별 확률밀도 확인

```python
def plot(df, column, idx):
    global set_width
    y_Max = 0
    x_Max = 0
    x_Min = 100
    for i in range(3):
        values = df[df["RESULT"] == i][column].value_counts().values / df[df["RESULT"] == i][column].shape[0]
        y_Max = max(y_Max, values.max())
        x_Max = max(x_Max, df[df["RESULT"] == i][column].value_counts().index.max())
        x_Min = min(x_Min, df[df["RESULT"] == i][column].value_counts().index.min())
        set_width = max(df[df["RESULT"] == i][column].value_counts().index)
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    for i, (r, ax, c) in enumerate(zip(res, axes, color)):
        print(df[df["RESULT"] == i][column].value_counts())
        index = df[df["RESULT"] == i][column].value_counts().index.tolist()
        values = df[df["RESULT"] == i][column].value_counts().values / df[df["RESULT"] == i][column].shape[
            0]
        ax.bar(index, values, color=c, label="{}".format(r))
        # if set_width >= 2.5:
        #     ax.bar(index, values, width=0.2, color=c, label="{}".format(r))
        # else:
        #     ax.bar(index, values, width=0.1, color=c, label="{}".format(r))
        ax.set_xlabel("{}".format(column))
        ax.set_ylabel("bins")
        ax.set_xlim(x_Min, x_Max)
        ax.set_ylim(0, y_Max)
        ax.legend()
    plt.show()
```

#### raw data

**FTHG (풀타임 홈팀의 골 수)**

<img src="https://user-images.githubusercontent.com/58063806/117828439-c5591200-b2ac-11eb-915e-d7ac6c980150.png" width=90% />

**FTAG (풀타임 원정팀의 골 수)**

<img src="https://user-images.githubusercontent.com/58063806/117828574-e588d100-b2ac-11eb-99b7-a97000864c83.png" width=90%/>

**HTHG (전반 홈팀의 골 수)**

<img src="https://user-images.githubusercontent.com/58063806/117828676-018c7280-b2ad-11eb-838a-7ce987c76ace.png" width=90% />

**HTAG (전반 원정팀의 골 수)**

<img src="https://user-images.githubusercontent.com/58063806/117828897-3993b580-b2ad-11eb-9ae9-608ac1ce079e.png" width=90% />

**HS (홈팀의 슈팅 수)**

<img src="https://user-images.githubusercontent.com/58063806/117829049-592ade00-b2ad-11eb-9198-f9224dde7fcf.png" width=90% />

**AS (원정팀의 슈팅 수)**

<img src="https://user-images.githubusercontent.com/58063806/117829193-7d86ba80-b2ad-11eb-943e-78645cf60986.png" width=90% />

**HST (홈팀의 유효슈팅 수)**

<img src="https://user-images.githubusercontent.com/58063806/117829320-9e4f1000-b2ad-11eb-8a7c-5fa330c6fbed.png" width=90%/>

**AST (원정팀의 유효슈팅 수)**

<img src="https://user-images.githubusercontent.com/58063806/117829448-ba52b180-b2ad-11eb-8d66-b06363bf5085.png" width=90% />

**HF (홈팀이 얻은 파울 수)**

<img src="https://user-images.githubusercontent.com/58063806/117829540-d22a3580-b2ad-11eb-8dea-d510abc93e2c.png" width=90% />

**AF (원정팀이 얻은 파울 수)**

<img src="https://user-images.githubusercontent.com/58063806/117829801-0d2c6900-b2ae-11eb-8a80-65888232cf28.png" width=90% />

**HC (홈팀이 얻은 코너킥 수)**

<img src="https://user-images.githubusercontent.com/58063806/117829915-2503ed00-b2ae-11eb-9135-79265b00f6a2.png" width=90% />

**AC (원정팀이 얻은 코너킥 수)**

<img src="https://user-images.githubusercontent.com/58063806/117830125-4c5aba00-b2ae-11eb-8a44-6dbf53956f1c.png" width=90% />

**HY (홈팀의 옐로우카드 수)**

<img src="https://user-images.githubusercontent.com/58063806/117830196-5da3c680-b2ae-11eb-868e-24ff14392c13.png" width=90% />

**AY (원정팀의 옐로우카드 수)**

<img src="https://user-images.githubusercontent.com/58063806/117830431-993e9080-b2ae-11eb-9bd2-f6a77130bef2.png" width=90% />

**HR (홈팀의 레드카드 수)**

<img src="https://user-images.githubusercontent.com/58063806/117830533-b07d7e00-b2ae-11eb-81a5-3b7d5b21749c.png" width=90% />

**AR (원정팀의 레드카드 수)**

<img src="https://user-images.githubusercontent.com/58063806/117830666-d014a680-b2ae-11eb-966b-f2af033a1817.png" width=90% />

**<u>골에 대한 데이터들을 제외하고는 승, 무, 패 데이터 간의 차이가 미미함</u>**

#### 분산분석(ANOVA)을 통해 확인 

- 각 집단의 데이터 개수가 비슷하고 데이터 분포가 정규 분포를 이루는 경우에 신뢰도가 높음
- StandardScaler로 정규화를 진행 **(정규분포화)**
- SMOTE로 oversampling 진행 **(데이터 개수 맞춤)**

```python
scaler = StandardScaler()
scaled_train = scaler.fit_transform(train.iloc[:, 3:-1])
scaled_train = pd.DataFrame(scaled_train, columns=data.columns[3:-1])
sampler = SMOTE(random_state=42)
df = sampler.fit_resample(scaled_train, train.iloc[:, [-1]])
df = pd.concat([df[0], df[1]], axis=1)
fstat, p_val = f_oneway(df.loc[df["RESULT"] == 0, df.columns[:-1]],
                        df.loc[df["RESULT"] == 1, df.columns[:-1]],
                        df.loc[df["RESULT"] == 2, df.columns[:-1]])
```

AC (원정팀이 얻은 코너킥 수) 데이터에 대해 p-value 0.73으로 유의수준 0.05을 초과 (귀무가설 채택)

> 각 집단의 평균이 동일 (집단의 분류에 있어서 중요도가 떨어짐)

#### 사후검정을 통해 확인

- 특성별로 각 집단간의 차이 유무를 확인하기 위함

```python
for i in range(3, 19):
    print(train.columns[i])
    posthoc = pairwise_tukeyhsd(train.iloc[:, [i]], train.iloc[:, [-1]], alpha=0.05)
    print(posthoc)
    plt.figure(figsize=(10, 10))
    posthoc.plot_simultaneous()
    plt.title("{}".format(train.columns[i]))
    plt.show()
```

<img src="https://user-images.githubusercontent.com/58063806/117858511-eaf41480-b2c8-11eb-8203-f606ae93ef87.png" width=40% />

<img src="https://user-images.githubusercontent.com/58063806/117858638-09f2a680-b2c9-11eb-8471-103b4f4dee19.png" width=40% />

<img src="https://user-images.githubusercontent.com/58063806/117858762-2989cf00-b2c9-11eb-8ffe-1e70712828bc.png" width=40% />

- 위와 같이 파울, 코너킥, 옐로카드에 해당하는 특성들의 일부 집단이 동일한 것을 볼 수 있음
- EX

<img src="https://user-images.githubusercontent.com/58063806/117859224-b03eac00-b2c9-11eb-8399-f49f1f7b6327.png" width=70%/>

- HF 특성은 승, 무, 패의 모든 경우에서 겹침 (모든 집단의 평균이 거의 동일하다고 판단)
- **데이터의 분포를 시각화 했던 것과 유사한 결과가 나옴 (골과 슈팅을 제외한 데이터들은 각 집단간의 분포 차이가 크지 않음)** 



**최근 5경기 상대전적**

<img src="https://user-images.githubusercontent.com/58063806/116851771-00fd3780-ac2e-11eb-98cc-b3d2786ec66d.png" width=90%/>

5경기 상대전적이 없는 -1 값을 제외하고

패 : 0 ~ 1.2, 1.3 정도에 밀집

무 : 0.8 ~ 2 정도에 밀집

승 : 0.8 ~ 2.7 정도에 밀집 (3인 경우가 존재) 

**홈팀의 이전 5경기 성적**

<img src="https://user-images.githubusercontent.com/58063806/116852006-8254ca00-ac2e-11eb-8da8-cfd03b026d5f.png" width=90% />

패 : 0.5 ~ 1.3 정도에 밀집

무 : 0.7 ~ 1.7 정도에 밀집 (3인 경우 2.5% 정도 존재)

승 : 0.9 ~ 2.1 정도에 밀집 (3인 경우 7.5% 정도 존재)

**원정팀의 이전 5경기 성적**

<img src="https://user-images.githubusercontent.com/58063806/116852690-bc729b80-ac2f-11eb-86a0-44724a31585f.png" width=90% />

패 : 0.5 ~ 2.5 정도에 밀집 (3인 경우가 10% 존재)

무 : 패배와 비슷한 분포를 보이지만 3인 경우가 현저히 줄어듬

승 : 0 ~ 2.3 정도에 밀집 (무승부와 비슷한 분포를 보이지만 조금 더 왼쪽으로 이동한 경향)

**홈팀의 이전 5경기 득점**

<img src="https://user-images.githubusercontent.com/58063806/116853080-694d1880-ac30-11eb-8227-9fa1b03e6fb0.png" width=90%/>

패 : 0.5 ~ 2.3 정도에 밀집

무 : 패배와 비슷한 분포를 보이지만 1.5 이상 빈도가 약간 상승

승 : 마찬가지로 무승부에서 1.5 이상 빈도가 약간 상승한 모습 (3.0 이상의 값들도 어느 정도 존재)

**홈팀의 이전 5경기 실점**

<img src="https://user-images.githubusercontent.com/58063806/116853414-f6906d00-ac30-11eb-8e68-ec83f8e5f585.png" width=90% />

패 : 0.5 ~ 2 정도에 밀집 (3.0 이상의 값들이 어느 정도 존재)

무 : 0.4 ~ 1.7 정도에 밀집 (3.0을 넘는 값들이 존재하지 않음)

승 : 0.4 ~ 1.8 정도에 밀집 (대부분 2.7 이하의 값들로 구성)

**원정팀의 이전 5경기 득점**

<img src="https://user-images.githubusercontent.com/58063806/116853790-98b05500-ac31-11eb-8ca1-d8ca9fe72288.png" width=90% />

패 : 0.5 ~ 2.5 정도에 밀집 (2.5 이상의 값들도 10% 이상 존재)

무 : 0.3 ~ 1.8 정도에 밀집 (2.3 이상의 값들이 존재하지 않음)

승 : 0 ~ 1.8 정도에 밀집 (무승부에 비해 그래프가 왼쪽으로 조금 치우침)

**원정팀의 이전 5경기 실점**

<img src="https://user-images.githubusercontent.com/58063806/116854085-14aa9d00-ac32-11eb-861a-ecf8357288ea.png" width=90% />

패 : 0.3 ~ 2.0 정도에 밀집 (2.7 이상의 값들이 존재하지 않음)

무 : 0.7 ~ 2.3 정도에 밀집 

승 : 0.7 ~ 2.4 정도에 밀집

**홈팀의 풀타임 골 수의 평균치**

<img src="https://user-images.githubusercontent.com/58063806/116854553-d497ea00-ac32-11eb-818d-1fd067134f91.png" width=90% />

패 : 1 ~ 1.2 정도에 밀집 (1.2가 40%의 비율을 차지함)

무 :  패배와 비슷한 분포를 보이지만 1.2의 비율이 작어지고 1.9의 비율이 늘어남

승 : 무승부에 비해 1.2의 비율이 더욱 작아지고 1.9와 2.1의 비율이 늘어남

**홈팀의 하프타임 골 수의 평균치**

<img src="https://user-images.githubusercontent.com/58063806/116854593-e37e9c80-ac32-11eb-8067-30010e6f7ee9.png" width=90% />

풀타임 골 수 피처와 비슷한 양상으로 패배에서 승리로 갈수록 더 높은 수치의 비율이 높아지는 경향

**원정팀의 풀타임 골 수의 평균치**

<img src="C:\Users\salmon11\AppData\Roaming\Typora\typora-user-images\image-20210503171403922.png" width=90% />

**원정팀의 하프타임 골 수의 평균치**

<img src="https://user-images.githubusercontent.com/58063806/116854664-04df8880-ac33-11eb-9fff-32e97fe06ea4.png" width=90% />

홈팀 피처의 경우와 반대의 경향

**홈팀의 슈팅 평균치**

<img src="https://user-images.githubusercontent.com/58063806/116854813-42dcac80-ac33-11eb-9f24-72a4a37dd82d.png" width=90% />

**홈팀의 유효슈팅 평균치**

<img src="C:\Users\salmon11\AppData\Roaming\Typora\typora-user-images\image-20210503171645399.png" width=90% />

**홈팀의 얻은 파울 평균치** 

<img src="https://user-images.githubusercontent.com/58063806/116854898-6273d500-ac33-11eb-8004-a0e14e631055.png" width=90% />

**홈팀의 얻은 코너킥 평균치**

<img src="https://user-images.githubusercontent.com/58063806/116855612-a4514b00-ac34-11eb-9cbf-eab23f5b8c85.png" width=90% />

**홈팀의 옐로 카드 평균치**

<img src="https://user-images.githubusercontent.com/58063806/116855648-b206d080-ac34-11eb-9c06-42023dbeec06.png" width=90% />

**홈팀의 레드 카드 평균치**

<img src="https://user-images.githubusercontent.com/58063806/116855694-c21eb000-ac34-11eb-927c-12789c4118dc.png" width=90% />

**원정팀의 슈팅 평균치**

<img src="https://user-images.githubusercontent.com/58063806/116855174-ea59df00-ac33-11eb-89c3-eb0495726203.png" width=90% />

**원정팀의 유효슈팅 평균치**

<img src="https://user-images.githubusercontent.com/58063806/116855219-fcd41880-ac33-11eb-884d-0c69407f202f.png" width=90% />

**원정팀의 얻은 파울 평균치** 

<img src="https://user-images.githubusercontent.com/58063806/116855249-0a899e00-ac34-11eb-830a-a7228db5905e.png" width=90% />

**원정팀의 얻은 코너킥 평균치**

<img src="https://user-images.githubusercontent.com/58063806/116855282-19705080-ac34-11eb-8c48-b811562d548a.png" width=90%/>

**원정팀의 옐로 카드 평균치**

<img src="https://user-images.githubusercontent.com/58063806/116855324-29883000-ac34-11eb-92f0-953f5566368f.png" width=90%/>

**원정팀의 레드 카드 평균치**

<img src="https://user-images.githubusercontent.com/58063806/116855364-373db580-ac34-11eb-90f7-4bf518123fd0.png" width=90% />



- 슈팅과 유효슈팅에 있어서는 패배시에 비해 무승부와 승리시에 더 높은 수치를 기록하는 비율이 상승하는 것을 볼 수 있음 
- 피파울, 얻은 코너킥, 옐로 카드에 있어서는 승리, 무승부, 패배 시에 두드러진 분포의 변화가 없고 크게 차이가 없음

- 레드 카드는 일반적으로 많이 발생하지 않음에 따라 전체 평균치는 가치가 없다고 판단 (이전 5경기와 같은 식으로 수정하거나 제외 고려)
  - 이전 5경기에서 받은 레드카드 수로 변경 (성능에 미치는 영향 미미)
- 대부분의 오분류는 패배나 무승부를 승리로 예측하는 경우

<img src="https://user-images.githubusercontent.com/58063806/117242911-2cbc2f80-ae71-11eb-86ff-0217ed0c2690.png" width=10% />

데이터가 많지 않은 상황에서 무승부나 패배에 비해 승리 데이터가 약 1700 ~ 2000개 가량 많은 것이 가장 큰 이유로 보임 

- 또한 중요하다고 생각되는 H2H_record 값이 -1로 관측되는 데이터가 5767개로 절반이 넘어가는 문제가 있는데 이 부분은 상대전적 경기수를 3경기로 줄여서 데이터를 다시 생성할 필요가 있음
  
- 상대전적 경기수를 3경기로 줄여도 H2H_record 값이 -1로 관측되는 데이터가 3754개 발생
  
- H2H_record , L5_home_record, L5_away_record, L5_home_GF, L5_home_GA, L5_away_GF, L5_away_GA와 같이 이전 5경기를 기준으로 생성한 피처들은 이전 경기수가 이에 미치지 못하는 데이터들에 있어서는 -1로 일괄처리하는 대신 이전 경기에 대해 같은 방식을 적용해 볼 필요가 있음   
  - H2H_record는 여전히 1324개의 -1 값이 존재
  - <img src="https://user-images.githubusercontent.com/58063806/117410381-4d59b780-af4d-11eb-80d8-49a67c9a44f1.png" width=20% />
  - 나머지 피처들도 위와 같은 양의 결측치가 발생
  
- 또한 2부리그 경기에 대해서는 일괄적으로 0.8로 감소시키는 것보다 2부리그에 있던 팀이 프리미어리그에 승격해서 치르는 경기에 한정해서 피처 부분적으로 감소와 증가를 시켜볼 필요가 있음 **(리그 수준에 따른 가중치를 부여)**
  - 피처를 생성할 때 **해당 시점에 1부 리그에 있는 상태(1부 리그 경기)일 때 지난 경기 중 2부 리그에서 치른 경기에 대해 가중치 부여**
    - 승점, 득점, 슈팅, 유효슈팅은 80%로 감소
    - 실점은 120%로 증가
  - 반대로 **해당 시점에 2부 리그에 있는 상태(2부 리그 경기)일 때 지난 경기 중 1부 리그에서 치른 경기에 대해 가중치 부여**
    - 승점, 득점, 슈팅, 유효슈팅은 120%로 증가
    - 실점은 80%로 감소

- 전반적인 피처 수정 이후에도 성능에 큰 변화는 없음

  - 추가적인 피처 생성과 데이터를 더 늘릴 필요가 있어보임
  - championship의 기록이 있는 0405 시즌부터 1819 시즌까지 train dataset (13774 row)

  <img src="https://user-images.githubusercontent.com/58063806/117671758-1bae4e00-b1e4-11eb-9835-815221650b0a.png" width=13%/>

  - 1920 시즌과 현재까지 기록이 있는 2021 시즌의 프리미어리그 경기를 대상으로 test dataset구성 (717 row)

  <img src="https://user-images.githubusercontent.com/58063806/117671915-44cede80-b1e4-11eb-92b8-d885cf7781cd.png" width=13% />

   

#### Feature importance

- Random forest

<img src="https://user-images.githubusercontent.com/58063806/117673922-3681c200-b1e6-11eb-8768-300aa09624db.png" width=60% />

- XGBoost

<img src="https://user-images.githubusercontent.com/58063806/117674145-616c1600-b1e6-11eb-9df6-2f8d8391da83.png" width=60% />

- LGBM

<img src="https://user-images.githubusercontent.com/58063806/117674299-8496c580-b1e6-11eb-9983-f91b808c6a96.png" width=60% />

- 레드카드에 대한 피처는 대부분 낮은 중요도를 보임
- 예상외로 이전 5경기를 대상으로 구성한 피처들보다 전체의 평균치 피처들이 더 높은 중요도를 나타냄
  - 이전 5경기를 대상으로 구성한 피처들의 결측치 때문으로 예상 (H2H_record는 1466개의 -1 값 존재)

