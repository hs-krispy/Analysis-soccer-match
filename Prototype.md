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

**0708 ~ 1819 시즌 데이터를 대상으로 대략적인 성능확인**

```python
X_train, X_val, y_train, y_val = train_test_split(df.iloc[:, :-1], np.ravel(df.iloc[:, [-1]]), test_size=0.2, stratify=np.ravel(df.iloc[:, [-1]]), random_state=42)
print(X_train.shape, X_val.shape)
# (8946, 23) (2237, 23)
```

- DecisionTree - 38.936%
- Randomforest - 46.67%
- LogisticRegression - 47.743%
- xgboost - 43.987%
- lightgbm - 45.597%
- SVC - 46.312%
- catboost - 46.133%

DecisionTree를 제외하고 파라미터를 튜닝하지 않은 **대부분의 모델들이 45 ~ 47% 정도의 Accuracy를 보임**

> 추후에 피처 엔지니어링과 모델 튜닝을 통해 성능 향상을 기대 가능