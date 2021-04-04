#### 잉글리쉬 프리미어리그 경기 결과에 미치는 요인 분석 및 예측



**keyword - Feature Engineering, Database, ML, 경기 결과 예측**

- 경기 결과에 미치는 요인들을 분석하고 결과를 정확도 60% 이상으로 예측하고자 함



## Feature list

- Average Halftime goal
- Average Fulltime goal



one-hot-encoding 필요

- Halftime result
- Fulltime result
  - 홈 팀 기준으로 이기다가 지면 : 0 
  - 이기다가 비기면 : 1 
  - 비기다가 지면 : 2
  - 전 후반 모두 리드 : 3
  - 비기다가 이기면 : 4
  - 지다가 이기면 : 5



- Average shooting
- Average shoot on target
- Average Fouls commited (얻은 파울)
- Average get Corners



- Average yellow cards

- Average red cards

  - Average Booking Points (10 = yellow, 25 = red)

  

- 최근 5경기 상대전적

- 최근 5경기 성적

  - 승 - 3, 무 - 1, 패 - 0

  

- 최근 5경기 골득실

  - 5경기 평균 득점, 5경기 평균 실점 (홈팀은 최근 홈 5경기, 원정팀은 최근 원정 5경기)

- pezzali score