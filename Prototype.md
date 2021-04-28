## Prototype

<img src="https://user-images.githubusercontent.com/58063806/113880113-27c67a80-97f6-11eb-88f0-48aa1da3a115.png" width=100% height=100% />

- **H2H_record : 두 팀간의 이전 5경기 상대전적**
  -  홈팀 기준으로 5경기 획득 승점 평균 (데이터셋 내에서 전적이 5경기에 못미치면 -1)
- **L5_home_record, L5_away_record : 홈팀과 원정팀의 이전 5경기 성적**
  - 홈팀과 원정팀에 대해 각각 이전 5경기에 획득한 승점 평균 (데이터셋 내에서 이전 경기수가 5경기에 못미치면 -1)
- **L5_home_GF, L5_home_GA, L5_away_GF, L5_away_GA : 홈팀과 원정팀의 이전 5경기 득점과 실점 평균 **
  - 홈팀은 최근 홈 5경기, 원정팀은 최근 원정 5경기를 대상으로 함
- **Home_AVG_FT_G, Home_AVG_HT_G, Away_AVG_FT_G, Away_AVG_FT_G  : 홈팀과 원정팀의 풀타임, 하프타임 골 수에 대한 평균치**
- **Avg_HS, Avg_HST, Avg_HF, Avg_HC, Avg_HY, Avg_HR ... : 홈팀과 원정팀의 슈팅, 유효슈팅, 얻은 파울, 코너킥, 옐로 카드, 레드 카드에 대한 평균치**
  - 위의 두 항목은 데이터의 시작인 0708 시즌부터 마지막인 1920 시즌까지 한 경기를 치를때마다 계속 피처의 값을 갱신

**0708 ~ 1819 시즌 데이터를 이용해서 대략적인 성능확인**

**shape - 11183 x 24**

- DecisionTree - 47.105%
- Randomforest - 50.526%
- LogisticRegression - 52.368% (무승부는 예측하지 못함)
- xgboost - 50.0%
- lightgbm - 52.105%
- SVC - 51.842% (무승부는 예측하지 못함)
- catboost - 51.579%

DecisionTree를 제외하고 파라미터를 튜닝하지 않은 **대부분의 모델들이 50 ~ 52% 정도의 Accuracy를 보임**

**0708 ~ 1819 시즌 데이터 중 프리미어리그에 해당하는 데이터들로만 학습**

**shape - 4560 x 24**

```python
df = df[df["Div"] == "E0"]
```

- DecisionTree - 43.158%
- Randomforest - 50.0%
- LogisticRegression - 51.842% (무승부는 예측하지 못함)
- xgboost - 49.211%
- lightgbm - 47.632%
- SVC - 52.105% (무승부는 예측하지 못함)
- catboost - 48.947%

DecisionTree를 제외하고 파라미터를 튜닝하지 않은 **대부분의 모델들이 48 ~ 52% 정도의 Accuracy를 보임** (전체 데이터를 사용했을때에 비해서 성능이 약간 하락)

> 예측 경향을 보았을때 무승부에 대한 부분을 거의 예측하지 못하고 대부분이 승과 패로 나뉘는 것을 볼 수 있었는데  feature engineering과 EDA를 통해 이 부분을 중심적으로 보완해야 함 

