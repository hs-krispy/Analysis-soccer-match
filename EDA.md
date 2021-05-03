## EDA

승, 무, 패 세 가지 경우에 대해 각각 특정 column 별 확률밀도 확인

```python
def plot(column, idx):
    global set_width
    y_Max = 0
    x_Max = 0
    x_Min = 100
    for i in range(3):
        values = np.round(TEST[TEST["RESULT"] == i][column], 1).value_counts().values / TEST[TEST["RESULT"] == i][column].shape[0]
        y_Max = max(y_Max, np.round(values.max(), 1))
        x_Max = max(x_Max, np.round(TEST[TEST["RESULT"] == i][column], 1).value_counts().index.max())
        x_Min = min(x_Min, np.round(TEST[TEST["RESULT"] == i][column], 1).value_counts().index.min())
        set_width = max(np.round(TEST[TEST["RESULT"] == i][column], 1).value_counts().index)
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    for i, (r, ax, c) in enumerate(zip(res, axes, color)):
        print(np.round(TEST[TEST["RESULT"] == i][column], 1).value_counts())
        index = np.round(TEST[TEST["RESULT"] == i][column], 1).value_counts().index.tolist()
        values = np.round(TEST[TEST["RESULT"] == i][column], 1).value_counts().values / TEST[TEST["RESULT"] == i][column].shape[0]
        if set_width >= 2.5:
            ax.bar(index, values, width=0.2, color=c, label="{}".format(r))
        else:
            ax.bar(index, values, width=0.1, color=c, label="{}".format(r))
        ax.set_xlabel("{}".format(column))
        ax.set_ylabel("bins")
        ax.set_xlim(x_Min, x_Max)
        ax.set_ylim(0, y_Max)
        ax.legend()
    plt.show()
```

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

**홈팀의 하프타임 골 수의 평균치**

<img src="https://user-images.githubusercontent.com/58063806/116854593-e37e9c80-ac32-11eb-8067-30010e6f7ee9.png" width=90% />

**원정팀의 풀타임 골 수의 평균치**

<img src="C:\Users\salmon11\AppData\Roaming\Typora\typora-user-images\image-20210503171403922.png" width=90% />

**원정팀의 하프타임 골 수의 평균치**

<img src="https://user-images.githubusercontent.com/58063806/116854664-04df8880-ac33-11eb-9fff-32e97fe06ea4.png" width=90% />

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



레드 카드는 일반적으로 많이 발생하지 않음에 따라 전체 평균치는 가치가 없다고 판단 (이전 5경기와 같은 식으로 수정하거나 제외 고려)