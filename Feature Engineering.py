import pandas as pd
import os
from tqdm import tqdm
import numpy as np

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

# train ~ 1819_season, test - 1920_season
DATA = []
# concat 0708 ~ 1920 season
# for idx, data in enumerate(os.listdir("dataset")):
#     if data == "1920(championship).csv":
#         continue
#     data_path = "dataset/" + data
#     df = pd.read_csv(data_path)
#     df = df[df.columns[:23]].drop(columns=["Date", "Referee"])
#     df.dropna(inplace=True)
#     # if "championship" in data:
#     #     D1_data = df[df.columns.difference(["HomeTeam", "AwayTeam", "Div", "FTR", "HTR", "HR", "AR"])] * 0.8
#     #     df[df.columns.difference(["HomeTeam", "AwayTeam", "Div", "FTR", "HTR"])] = D1_data
#     if len(DATA) == 0:
#         DATA = df
#     else:
#         DATA = pd.concat([DATA, df], ignore_index=True, axis=0)
#     if data == "1920.csv":
#         DATA.to_csv("Update_0720(Div0+Div1).csv", index=False)
#         break

path = "0719(Div0+Div1).csv"
past_data = pd.read_csv(path)
start_index = past_data.shape[0]
data = pd.read_csv("Update_0720(Div0+Div1).csv")
# 전 후반 결과 수치화 (승점으로)
data["FTR"].replace("A", 0, inplace=True)
data["FTR"].replace("D", 1, inplace=True)
data["FTR"].replace("H", 3, inplace=True)
data["HTR"].replace("A", 0, inplace=True)
data["HTR"].replace("D", 1, inplace=True)
data["HTR"].replace("H", 3, inplace=True)


# 최근 5경기 상대전적
def H2H(df, home, away, index):
    selected_df = df[df.index < index]
    record = selected_df[((selected_df['HomeTeam'] == home) & (selected_df['AwayTeam'] == away)) | (
            (selected_df['HomeTeam'] == away) & (selected_df['AwayTeam'] == home))].copy()
    # 최근 상대전적이 5경기 이상인 데이터
    if record.shape[0] >= 5:
        record = record[-5:]
        record.loc[(record['AwayTeam'] == home) & (record['FTR'] != 1), ['FTR']] = \
            abs(record.loc[(record['AwayTeam'] == home) & (record['FTR'] != 1), ['FTR']] - 3)
        df.loc[[index], ["H2H_record"]] = record['FTR'].mean()
    # 최근 상대전적이 5경기에 미치지 못하는 데이터 (존재하는 데이터 만큼만 같은 방식으로 적용)
    elif record.shape[0] >= 1:
        record.loc[(record['AwayTeam'] == home) & (record['FTR'] != 1), ['FTR']] = \
            abs(record.loc[(record['AwayTeam'] == home) & (record['FTR'] != 1), ['FTR']] - 3)
        df.loc[[index], ["H2H_record"]] = record['FTR'].mean()



# data['H2H_record'] = -1
# for i in tqdm(range(data.shape[0])):
#     home = data.loc[[i], ["HomeTeam"]].values.ravel()[0]
#     away = data.loc[[i], ["AwayTeam"]].values.ravel()[0]
#     H2H(data, home, away, i)


# data.to_csv("0719_H2H.csv", index=False)


# 지난 5경기 성적
def Last_5(df, home, away, index):
    selected_df = df[df.index < index]
    home_record = selected_df[((selected_df['HomeTeam'] == home) | (selected_df['AwayTeam'] == home))].copy()
    away_record = selected_df[((selected_df['HomeTeam'] == away) | (selected_df['AwayTeam'] == away))].copy()
    if home_record.shape[0] >= 5:
        home_record = home_record[-5:]
        home_record.loc[(home_record['AwayTeam'] == home) & (home_record['FTR'] != 1), ['FTR']] = \
            abs(home_record.loc[(home_record['AwayTeam'] == home) & (home_record['FTR'] != 1), ['FTR']] - 3)
        # 해당 시점에 1부 리그에 있는 상태(1부 리그 경기)일 때 지난 5경기 중 2부 리그에서 치른 경기에 대해 가중치 부여
        if df.loc[[index], "Div"].values == "E0" and home_record.loc[home_record["Div"] == "E1", "FTR"].shape[0] > 0:
            home_record.loc[home_record["Div"] == "E1", "FTR"] = home_record.loc[
                                                                     home_record["Div"] == "E1", "FTR"] * 0.8
        # 해당 시점에 2부 리그에 있는 상태(2부 리그 경기)일 때 지난 5경기 중 1부 리그에서 치른 경기에 대해 가중치 부여
        elif df.loc[[index], "Div"].values == "E1" and home_record.loc[home_record["Div"] == "E0", "FTR"].shape[0] > 0:
            home_record.loc[home_record["Div"] == "E0", "FTR"] = home_record.loc[
                                                                     home_record["Div"] == "E0", "FTR"] * 1.2
        df.loc[[index], ["L5_home_record"]] = home_record['FTR'].mean()
    elif home_record.shape[0] < 5:
        home_record.loc[(home_record['AwayTeam'] == home) & (home_record['FTR'] != 1), ['FTR']] = \
            abs(home_record.loc[(home_record['AwayTeam'] == home) & (home_record['FTR'] != 1), ['FTR']] - 3)
        if df.loc[[index], "Div"].values == "E0" and home_record.loc[home_record["Div"] == "E1", "FTR"].shape[0] > 0:
            home_record.loc[home_record["Div"] == "E1", "FTR"] = home_record.loc[
                                                                     home_record["Div"] == "E1", "FTR"] * 0.8
        elif df.loc[[index], "Div"].values == "E1" and home_record.loc[home_record["Div"] == "E0", "FTR"].shape[0] > 0:
            home_record.loc[home_record["Div"] == "E0", "FTR"] = home_record.loc[
                                                                     home_record["Div"] == "E0", "FTR"] * 1.2
        df.loc[[index], ["L5_home_record"]] = home_record['FTR'].mean()

    if away_record.shape[0] >= 5:
        away_record = away_record[-5:]
        away_record.loc[(away_record['AwayTeam'] == away) & (away_record['FTR'] != 1), ['FTR']] = \
            abs(away_record.loc[(away_record['AwayTeam'] == away) & (away_record['FTR'] != 1), ['FTR']] - 3)
        if df.loc[[index], "Div"].values == "E0" and away_record.loc[away_record["Div"] == "E1", "FTR"].shape[0] > 0:
            away_record.loc[away_record["Div"] == "E1", "FTR"] = away_record.loc[
                                                                     away_record["Div"] == "E1", "FTR"] * 0.8
        elif df.loc[[index], "Div"].values == "E1" and away_record.loc[away_record["Div"] == "E0", "FTR"].shape[0] > 0:
            away_record.loc[away_record["Div"] == "E0", "FTR"] = away_record.loc[
                                                                     away_record["Div"] == "E0", "FTR"] * 1.2
        df.loc[[index], ["L5_away_record"]] = away_record['FTR'].mean()
    elif away_record.shape[0] < 5:
        away_record.loc[(away_record['AwayTeam'] == away) & (away_record['FTR'] != 1), ['FTR']] = \
            abs(away_record.loc[(away_record['AwayTeam'] == away) & (away_record['FTR'] != 1), ['FTR']] - 3)
        if df.loc[[index], "Div"].values == "E0" and away_record.loc[away_record["Div"] == "E1", "FTR"].shape[0] > 0:
            away_record.loc[away_record["Div"] == "E1", "FTR"] = away_record.loc[
                                                                     away_record["Div"] == "E1", "FTR"] * 0.8
        elif df.loc[[index], "Div"].values == "E1" and away_record.loc[away_record["Div"] == "E0", "FTR"].shape[0] > 0:
            away_record.loc[away_record["Div"] == "E0", "FTR"] = away_record.loc[
                                                                     away_record["Div"] == "E0", "FTR"] * 1.2
        df.loc[[index], ["L5_away_record"]] = away_record['FTR'].mean()


data['L5_home_record'] = data["L5_away_record"] = -1
for i in tqdm(range(data.shape[0])):
    home = data.loc[[i], ["HomeTeam"]].values.ravel()[0]
    away = data.loc[[i], ["AwayTeam"]].values.ravel()[0]
    Last_5(data, home, away, i)


# data.to_csv("0719_H2H_L5.csv")


# 지난 5경기 평균득점, 평균실점
def Last_5_GF_GA(df, home, away, index):
    selected_df = df[df.index < index]
    home_record = selected_df[selected_df['HomeTeam'] == home].copy()
    away_record = selected_df[selected_df['AwayTeam'] == away].copy()

    if home_record.shape[0] >= 5:
        home_record = home_record[-5:]
        # 해당 시점에 1부 리그에 있는 상태(1부 리그 경기)일 때 지난 5경기 중 2부 리그에서 치른 경기에 대해 가중치 부여
        # 득점은 0.8, 실점은 1.2의 가중치
        if df.loc[[index], "Div"].values == "E0" and home_record.loc[home_record["Div"] == "E1"].shape[0] > 0:
            home_record.loc[home_record["Div"] == "E1", "FTHG"] = home_record.loc[
                                                                      home_record["Div"] == "E1", "FTHG"] * 0.8
            home_record.loc[home_record["Div"] == "E1", "FTAG"] = home_record.loc[
                                                                      home_record["Div"] == "E1", "FTAG"] * 1.2
        # 해당 시점에 2부 리그에 있는 상태(2부 리그 경기)일 때 지난 5경기 중 1부 리그에서 치른 경기에 대해 가중치 부여
        # 반대로 득점은 1.2, 실점은 0.8의 가중치
        elif df.loc[[index], "Div"].values == "E1" and home_record.loc[home_record["Div"] == "E0"].shape[0] > 0:
            home_record.loc[home_record["Div"] == "E0", "FTHG"] = home_record.loc[
                                                                      home_record["Div"] == "E0", "FTHG"] * 1.2
            home_record.loc[home_record["Div"] == "E0", "FTAG"] = home_record.loc[
                                                                      home_record["Div"] == "E0", "FTAG"] * 0.8
        df.loc[[index], ["L5_home_GF"]] = home_record['FTHG'].mean()
        df.loc[[index], ["L5_home_GA"]] = home_record['FTAG'].mean()
    elif home_record.shape[0] < 5:
        if df.loc[[index], "Div"].values == "E0" and home_record.loc[home_record["Div"] == "E1"].shape[0] > 0:
            home_record.loc[home_record["Div"] == "E1", "FTHG"] = home_record.loc[
                                                                      home_record["Div"] == "E1", "FTHG"] * 0.8
            home_record.loc[home_record["Div"] == "E1", "FTAG"] = home_record.loc[
                                                                      home_record["Div"] == "E1", "FTAG"] * 1.2
        elif df.loc[[index], "Div"].values == "E1" and home_record.loc[home_record["Div"] == "E0"].shape[0] > 0:
            home_record.loc[home_record["Div"] == "E0", "FTHG"] = home_record.loc[
                                                                      home_record["Div"] == "E0", "FTHG"] * 1.2
            home_record.loc[home_record["Div"] == "E0", "FTAG"] = home_record.loc[
                                                                      home_record["Div"] == "E0", "FTAG"] * 0.8
        df.loc[[index], ["L5_home_GF"]] = home_record['FTHG'].mean()
        df.loc[[index], ["L5_home_GA"]] = home_record['FTAG'].mean()

    if away_record.shape[0] >= 5:
        away_record = away_record[-5:]
        if df.loc[[index], "Div"].values == "E0" and away_record.loc[away_record["Div"] == "E1"].shape[0] > 0:
            away_record.loc[away_record["Div"] == "E1", "FTHG"] = away_record.loc[
                                                                      away_record["Div"] == "E1", "FTHG"] * 0.8
            away_record.loc[away_record["Div"] == "E1", "FTAG"] = away_record.loc[
                                                                      away_record["Div"] == "E1", "FTAG"] * 1.2
        elif df.loc[[index], "Div"].values == "E1" and away_record.loc[away_record["Div"] == "E0"].shape[0] > 0:
            away_record.loc[away_record["Div"] == "E0", "FTHG"] = away_record.loc[
                                                                      away_record["Div"] == "E0", "FTHG"] * 1.2
            away_record.loc[away_record["Div"] == "E0", "FTAG"] = away_record.loc[
                                                                      away_record["Div"] == "E0", "FTAG"] * 0.8
        df.loc[[index], ["L5_away_GF"]] = away_record['FTAG'].mean()
        df.loc[[index], ["L5_away_GA"]] = away_record['FTHG'].mean()
    elif away_record.shape[0] < 5:
        if df.loc[[index], "Div"].values == "E0" and away_record.loc[away_record["Div"] == "E1"].shape[0] > 0:
            away_record.loc[away_record["Div"] == "E1", "FTHG"] = away_record.loc[
                                                                      away_record["Div"] == "E1", "FTHG"] * 0.8
            away_record.loc[away_record["Div"] == "E1", "FTAG"] = away_record.loc[
                                                                      away_record["Div"] == "E1", "FTAG"] * 1.2
        elif df.loc[[index], "Div"].values == "E1" and away_record.loc[away_record["Div"] == "E0"].shape[0] > 0:
            away_record.loc[away_record["Div"] == "E0", "FTHG"] = away_record.loc[
                                                                      away_record["Div"] == "E0", "FTHG"] * 1.2
            away_record.loc[away_record["Div"] == "E0", "FTAG"] = away_record.loc[
                                                                      away_record["Div"] == "E0", "FTAG"] * 0.8
        df.loc[[index], ["L5_away_GF"]] = away_record['FTAG'].mean()
        df.loc[[index], ["L5_away_GA"]] = away_record['FTHG'].mean()


data['L5_home_GF'] = data["L5_home_GA"] = data["L5_away_GF"] = data["L5_away_GA"] = -1
for i in tqdm(range(data.shape[0])):
    home = data.loc[[i], ["HomeTeam"]].values.ravel()[0]
    away = data.loc[[i], ["AwayTeam"]].values.ravel()[0]
    Last_5_GF_GA(data, home, away, i)


# data.to_csv("0719_H2H_L5_GF_GD.csv")

# 지난 5경기 레드카드 수
def Last_5_RED(df, home, away, index):
    selected_df = df[df.index < index]
    home_record = selected_df[((selected_df['HomeTeam'] == home) | (selected_df['AwayTeam'] == home))].copy()
    away_record = selected_df[((selected_df['HomeTeam'] == away) | (selected_df['AwayTeam'] == away))].copy()
    if home_record.shape[0] >= 5:
        home_record = home_record[-5:]
        df.loc[[index], ["L5_home_Red"]] = home_record.loc[home_record['HomeTeam'] == home, ["HR"]].sum().values + \
                                           home_record.loc[home_record["AwayTeam"] == home, ["AR"]].sum().values
    if away_record.shape[0] >= 5:
        away_record = away_record[-5:]
        df.loc[[index], ["L5_away_Red"]] = away_record.loc[away_record['HomeTeam'] == away, ["HR"]].sum().values + \
                                           away_record.loc[away_record["AwayTeam"] == away, ["AR"]].sum().values


data["L5_home_Red"] = data["L5_away_Red"] = 0
for i in tqdm(range(data.shape[0])):
    home = data.loc[[i], ["HomeTeam"]].values.ravel()[0]
    away = data.loc[[i], ["AwayTeam"]].values.ravel()[0]
    Last_5_RED(data, home, away, i)
#
# data.loc[:, ["H2H_record", "L5_home_Red", "L5_away_Red"]].to_csv("H2H_L5_RED.csv", index=False)


# 평균 FT, HT 득점
def AVG_FT_HT_G(df, home, away, index):
    selected_df = df[df.index < index]
    home_record = selected_df[((selected_df['HomeTeam'] == home) | (selected_df['AwayTeam'] == home))].copy()
    # 해당 경기의 홈팀이 지난 경기들 중 원정팀인 경우에 대해 계산을 용이하게 하기위해 데이터를 수정
    h_FTAG = home_record.loc[home_record['AwayTeam'] == home]['FTAG']
    h_HTAG = home_record.loc[home_record['AwayTeam'] == home]['HTAG']
    home_record.loc[home_record['AwayTeam'] == home, ['FTHG']] = h_FTAG
    home_record.loc[home_record['AwayTeam'] == home, ['HTHG']] = h_HTAG
    if df.loc[[index], "Div"].values == "E0" and home_record.loc[home_record["Div"] == "E1"].shape[0] > 0:
        home_record.loc[home_record["Div"] == "E1", "FTHG"] = home_record.loc[home_record["Div"] == "E1", "FTHG"] * 0.8
        home_record.loc[home_record["Div"] == "E1", "HTHG"] = home_record.loc[home_record["Div"] == "E1", "HTHG"] * 0.8
    elif df.loc[[index], "Div"].values == "E1" and home_record.loc[home_record["Div"] == "E0"].shape[0] > 0:
        home_record.loc[home_record["Div"] == "E0", "FTHG"] = home_record.loc[home_record["Div"] == "E0", "FTHG"] * 1.2
        home_record.loc[home_record["Div"] == "E0", "HTHG"] = home_record.loc[home_record["Div"] == "E0", "HTHG"] * 1.2

    away_record = selected_df[((selected_df['HomeTeam'] == away) | (selected_df['AwayTeam'] == away))].copy()
    a_FTAG = away_record.loc[away_record['AwayTeam'] == away]['FTAG']
    a_HTAG = away_record.loc[away_record['AwayTeam'] == away]['HTAG']
    away_record.loc[away_record['AwayTeam'] == away, ['FTHG']] = a_FTAG
    away_record.loc[away_record['AwayTeam'] == away, ['HTHG']] = a_HTAG
    if df.loc[[index], "Div"].values == "E0" and away_record.loc[away_record["Div"] == "E1"].shape[0] > 0:
        away_record.loc[away_record["Div"] == "E1", "FTHG"] = away_record.loc[away_record["Div"] == "E1", "FTHG"] * 0.8
        away_record.loc[away_record["Div"] == "E1", "HTHG"] = away_record.loc[away_record["Div"] == "E1", "HTHG"] * 0.8
    elif df.loc[[index], "Div"].values == "E1" and away_record.loc[away_record["Div"] == "E0"].shape[0] > 0:
        away_record.loc[away_record["Div"] == "E0", "FTHG"] = away_record.loc[away_record["Div"] == "E0", "FTHG"] * 1.2
        away_record.loc[away_record["Div"] == "E0", "HTHG"] = away_record.loc[away_record["Div"] == "E0", "HTHG"] * 1.2

    if home_record.shape[0] >= 1:
        df.loc[[index], ["Home_AVG_FT_G"]] = home_record['FTHG'].mean()
        df.loc[[index], ["Home_AVG_HT_G"]] = home_record['HTHG'].mean()
    if away_record.shape[0] >= 1:
        df.loc[[index], ["Away_AVG_FT_G"]] = away_record['FTHG'].mean()
        df.loc[[index], ["Away_AVG_HT_G"]] = away_record['HTHG'].mean()



data["Away_AVG_FT_G"] = data["Away_AVG_HT_G"] = 0
for i in tqdm(range(data.shape[0])):
    home = data.loc[[i], ["HomeTeam"]].values.ravel()[0]
    away = data.loc[[i], ["AwayTeam"]].values.ravel()[0]
    AVG_FT_HT_G(data, home, away, i)


# data.to_csv("0719_H2H_L5_GF_GD_FTG_HTG.csv", index=False)

def AVG_feature(df, home, away, index):
    selected_df = df[df.index < index]
    home_record = selected_df[((selected_df['HomeTeam'] == home) | (selected_df['AwayTeam'] == home))].copy()
    away_record = selected_df[((selected_df['HomeTeam'] == away) | (selected_df['AwayTeam'] == away))].copy()
    # 슈팅에 관련된 피처에 대해서만 리그 수준에 따른 가중치를 부여
    if home_record.shape[0] >= 1:
        home_record.loc[home_record['AwayTeam'] == home, ["HS", "HST", "HF", "HC", "HY"]] = \
            home_record.loc[home_record['AwayTeam'] == home][["AS", "AST", "AF", "AC", "AY"]].values
        if df.loc[[index], "Div"].values == "E0" and home_record.loc[home_record["Div"] == "E1"].shape[0] > 0:
            home_record.loc[home_record["Div"] == "E1", "HS"] = home_record.loc[home_record["Div"] == "E1", "HS"] * 0.8
            home_record.loc[home_record["Div"] == "E1", "HST"] = home_record.loc[
                                                                     home_record["Div"] == "E1", "HST"] * 0.8
        elif df.loc[[index], "Div"].values == "E1" and home_record.loc[home_record["Div"] == "E0"].shape[0] > 0:
            home_record.loc[home_record["Div"] == "E0", "HS"] = home_record.loc[home_record["Div"] == "E0", "HS"] * 1.2
            home_record.loc[home_record["Div"] == "E0", "HST"] = home_record.loc[
                                                                     home_record["Div"] == "E0", "HST"] * 1.2
        df.loc[[index], ["Avg_HS", "Avg_HST", "Avg_HF", "Avg_HC", "Avg_HY"]] = home_record.loc[:,
                                                                               ["HS", "HST", "HF", "HC", "HY"]].mean(
            axis=0).values

    if away_record.shape[0] >= 1:
        away_record.loc[selected_df['AwayTeam'] == away, ["HS", "HST", "HF", "HC", "HY"]] = \
            away_record.loc[selected_df['AwayTeam'] == away][["AS", "AST", "AF", "AC", "AY"]].values
        if df.loc[[index], "Div"].values == "E0" and home_record.loc[home_record["Div"] == "E1"].shape[0] > 0:
            away_record.loc[away_record["Div"] == "E1", "HS"] = away_record.loc[away_record["Div"] == "E1", "HS"] * 0.8
            away_record.loc[away_record["Div"] == "E1", "HST"] = away_record.loc[
                                                                     away_record["Div"] == "E1", "HST"] * 0.8
        elif df.loc[[index], "Div"].values == "E1" and home_record.loc[home_record["Div"] == "E0"].shape[0] > 0:
            away_record.loc[away_record["Div"] == "E0", "HS"] = away_record.loc[away_record["Div"] == "E0", "HS"] * 1.2
            away_record.loc[away_record["Div"] == "E0", "HST"] = away_record.loc[
                                                                     away_record["Div"] == "E0", "HST"] * 1.2
        df.loc[[index], ["Avg_AS", "Avg_AST", "Avg_AF", "Avg_AC", "Avg_AY"]] = away_record.loc[:,
                                                                               ["HS", "HST", "HF", "HC", "HY"]].mean(
            axis=0).values


data[["Avg_HS", "Avg_HST", "Avg_HF", "Avg_HC", "Avg_HY", "Avg_AS", "Avg_AST", "Avg_AF", "Avg_AC", "Avg_AY"]] = 0
for i in tqdm(range(data.shape[0])):
    home = data.loc[[i], ["HomeTeam"]].values.ravel()[0]
    away = data.loc[[i], ["AwayTeam"]].values.ravel()[0]
    AVG_feature(data, home, away, i)


# data.iloc[:start_index, :].to_csv("Update_0719(train).csv", index=False)
# data.iloc[start_index:, :].to_csv("Update_1920(test).csv", index=False)
