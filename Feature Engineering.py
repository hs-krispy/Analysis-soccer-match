import pandas as pd
import os
import mglearn
import seaborn as sns
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway
from sklearn.svm import SVC
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import ExtraTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectPercentile, f_classif
from collections import Counter

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

train_index = 13774

DATA = pd.read_csv("new_0421(Div0+Div1).csv")
DATA.drop(columns=["FTR"], inplace=True)
# DATA.loc[DATA.RESULT == 1, "RESULT"] = 0
DATA.loc[DATA["HTR"] == "A", "HTR"] = 0
DATA.loc[DATA["HTR"] == "D", "HTR"] = 1
DATA.loc[DATA["HTR"] == "H", "HTR"] = 3
DATA["HTR"] = DATA["HTR"].astype(np.float32)


# ------------------ merge 0708 ~ 2021 season ------------------
# train 0708 ~ 1819_season, test - 1920, 2021_season
# DATA = []
# for idx, data in enumerate(os.listdir("dataset")):
#     if data == "1920(championship).csv" or data == "2021(championship).csv":
#         continue
#     data_path = "dataset/" + data
#     df = pd.read_csv(data_path, error_bad_lines=False, encoding="cp949")
#     print(data, "\n", df.head())
#     df = df[df.columns[:23]].drop(columns=["Date", "Referee"])
#     df.dropna(inplace=True)
#     # if "championship" in data:
#     #     D1_data = df[df.columns.difference(["HomeTeam", "AwayTeam", "Div", "FTR", "HTR", "HR", "AR"])] * 0.8
#     #     df[df.columns.difference(["HomeTeam", "AwayTeam", "Div", "FTR", "HTR"])] = D1_data
#     if len(DATA) == 0:
#         DATA = df
#     else:
#         DATA = pd.concat([DATA, df], ignore_index=True, axis=0)
#     if data == "2021.csv":
#         DATA.to_csv("new_0421(Div0+Div1).csv", index=False)
#         break
#     if data == "1819.csv":
#         print(DATA.shape[0])

class Predict_football():
    def __init__(self, raw_data, model):
        plt.rc("font", family="Malgun Gothic")
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(10, 10))
        self.data = raw_data
        self.scaler = StandardScaler()
        self.sampler = SMOTE(random_state=42)
        self.pca = PCA(random_state=42)
        self.add = 0.1
        self.skip = []
        self.columns = None
        self.clf = model
        self.data["2HTHG"] = self.data["FTHG"] - self.data["HTHG"]
        self.data["2HTAG"] = self.data["FTAG"] - self.data["HTAG"]
        self.data["FGD"] = self.data["FTHG"] - self.data["FTAG"]
        self.data["2HGD"] = self.data["2HTHG"] - self.data["2HTAG"]
        self.data["HGD"] = self.data["HTHG"] - self.data["HTAG"]
        self.data["SD"] = self.data["HS"] - self.data["AS"]
        self.data["STD"] = self.data["HST"] - self.data["AST"]
        self.data["Pezzali"] = (self.data["FTHG"] + self.add) / (self.data["HS"] + self.add) * (
                self.data["AS"] + self.add) / (self.data["FTAG"] + self.add)
        self.COL = ["FTHG", "2HTHG", "HTHG", "HS", "HST", "FTAG", "2HTAG", "HTAG", "AS", "AST"]
        self.r_COL = ["FGD", "2HGD", "HGD", "SD", "STD", "Pezzali"]
        self.df = self.data.copy()

    # ------------------ ANOVA ------------------

    def ANOVA(self):
        df = self.data.drop(columns=["Div", "HomeTeam", "AwayTeam", "RESULT"])
        df = df.iloc[:train_index, :]
        scaled_train = self.scaler.fit_transform(df)
        scaled_train = pd.DataFrame(scaled_train, columns=df.columns)
        df = self.sampler.fit_resample(scaled_train, self.data.loc[:train_index, "RESULT"])
        df = pd.concat([df[0], df[1]], axis=1)
        fstat, p_val = f_oneway(df.loc[df["RESULT"] == 0, df.columns[:-1]],
                                df.loc[df["RESULT"] == 1, df.columns[:-1]],
                                df.loc[df["RESULT"] == 2, df.columns[:-1]])
        print(p_val)
        print(df.columns[:-1][p_val > 0.05])

    # ------------------ Post-hoc ------------------

    def PH(self):
        df = self.data.drop(columns=["Div", "HomeTeam", "AwayTeam", "RESULT"])
        for i in df.columns:
            posthoc = pairwise_tukeyhsd(self.data.iloc[:, [i]], self.data["RESULT"], alpha=0.05)
            plt.figure(figsize=(10, 10))
            posthoc.plot_simultaneous()
            plt.title("{}".format(self.data.columns[i]))
            plt.show()

    # ------------------ plot data pdf(probability density function) ------------------

    def plot(self):
        res = ["패", "무", "승"]
        color = ["r", "g", "b"]
        y_Max = 0
        x_Max = 0
        x_Min = 100
        for col in self.data.columns[3:-1]:
            for i in range(3):
                values = self.data[self.data["RESULT"] == i][col].value_counts().values / \
                         self.data[self.data["RESULT"] == i][col].shape[0]
                y_Max = max(y_Max, values.max())
                x_Max = max(x_Max, self.data[self.data["RESULT"] == i][col].max())
                x_Min = min(x_Min, self.data[self.data["RESULT"] == i][col].min())
            fig, axes = plt.subplots(1, 3, figsize=(15, 8))
            for i, (r, ax, c) in enumerate(zip(res, axes, color)):
                print(self.data[self.data["RESULT"] == i][col].value_counts())
                index = self.data[self.data["RESULT"] == i][col].value_counts().index.tolist()
                values = self.data[self.data["RESULT"] == i][col].value_counts().values / \
                         self.data[self.data["RESULT"] == i][col].shape[0]
                ax.bar(index, values, color=c, label="{}".format(r))
                ax.set_xlabel("{}".format(col))
                ax.set_ylabel("bins")
                ax.set_xlim(x_Min, x_Max)
                ax.set_ylim(0, y_Max)
                ax.legend()
            plt.show()

    # ------------------ 해당 경기 홈팀과 원정팀의 이전 5경기 맞대결 데이터들의 평균 값 ------------------

    def H2H(self, home, away, index, ratio=1):
        selected_df = self.data[self.data.index < index]
        record = selected_df[((selected_df['HomeTeam'] == home) & (selected_df['AwayTeam'] == away)) | (
                (selected_df['HomeTeam'] == away) & (selected_df['AwayTeam'] == home))].copy()
        # 승리와 패배의 경우 현재 경기의 홈팀이 과거에 원정에서 치른 경기의 HTR과 RESULT를 홈과 바꿈 (계산 편리)
        record.loc[(record['AwayTeam'] == home) & (record['HTR'] != 1), ['HTR']] = \
            3 - record.loc[(record['AwayTeam'] == home) & (record['HTR'] != 1), ['HTR']]
        record.loc[(record['AwayTeam'] == home) & (record['RESULT'] != 1), ['RESULT']] = \
            2 - record.loc[(record['AwayTeam'] == home) & (record['RESULT'] != 1), ['RESULT']]
        # 현재 경기의 홈팀이 과거에 원정에서 치른 경기의 feature들을 홈팀기준으로 변경, Pezzali는 -가 아닌 역수를 취함
        record.loc[record["AwayTeam"] == home, self.r_COL[:-1]] = -record.loc[
            record["AwayTeam"] == home, self.r_COL[:-1]]
        record.loc[record["AwayTeam"] == home, ["Pezzali"]] = 1 / record.loc[record["AwayTeam"] == home, ["Pezzali"]]
        temp = record.loc[record["AwayTeam"] == home, self.COL[:5]].values
        record.loc[record["AwayTeam"] == home, self.COL[:5]] = record.loc[
            record["AwayTeam"] == home, self.COL[5:]].values
        record.loc[record["AwayTeam"] == home, self.COL[5:]] = temp

        if record.shape[0] == 0:
            self.skip.append(index)
            return
        div = 0
        if record.shape[0] >= 5:
            record = record[-5:]
        INDEX = record["RESULT"].value_counts().index
        VALUES = record["RESULT"].value_counts().values
        for idx, val in zip(INDEX, VALUES):
            record.loc[record["RESULT"] == idx, self.r_COL] = record.loc[record["RESULT"] == idx, self.r_COL] * val
            record.loc[record["RESULT"] == idx, self.COL] = record.loc[record["RESULT"] == idx, self.COL] * val
            div += val ** 2
        # 다른 방식과 혼합해서 사용할 경우 비율을 조정
        self.df.loc[[index], self.r_COL] = record[self.r_COL].sum(axis=0).values * ratio
        self.df.loc[[index], self.COL] = record[self.COL].sum(axis=0).values * ratio
        # self.df.loc[[index], self.r_self.COL] = record[self.r_self.COL].ewm(span=record.shape[0], adjust=True).mean().sum().values * ratio
        # self.df.loc[[index], self.COL] = record[self.COL].ewm(span=record.shape[0], adjust=True).mean().mean().values * ratio
        self.df.loc[[index], ["HTR"]] = np.ravel(record["HTR"].mean(axis=0)) * ratio

    # ------------------ 해당 경기 홈팀의 이전 5경기 데이터 평균 값 - 해당 경기 원정팀의 이전 5경기 데이터 평균 값  ------------------
    
    def Last_5(self, home, away, index, ratio=0.2):
        selected_df = self.data[self.data.index < index]
        home_record = selected_df[((selected_df['HomeTeam'] == home) | (selected_df['AwayTeam'] == home))].copy()
        away_record = selected_df[((selected_df['HomeTeam'] == away) | (selected_df['AwayTeam'] == away))].copy()
        home_record["RESULT"].replace(2, 3, inplace=True)
        away_record["RESULT"].replace(2, 3, inplace=True)
        # 현재 경기의 홈팀이 과거에 원정에서 치른 경기의 HTR과 RESULT를 홈과 바꿈
        home_record.loc[(home_record['AwayTeam'] == home) & (home_record['HTR'] != 1), ['HTR']] = \
            3 - home_record.loc[(home_record['AwayTeam'] == home) & (home_record['HTR'] != 1), ['HTR']]
        home_record.loc[(home_record['AwayTeam'] == home) & (home_record['RESULT'] != 1), ['RESULT']] = \
            3 - home_record.loc[(home_record['AwayTeam'] == home) & (home_record['RESULT'] != 1), ['RESULT']]
        home_record.loc[home_record["AwayTeam"] == home, ["Pezzali"]] = 1 / home_record.loc[
            home_record["AwayTeam"] == home, ["Pezzali"]]
        # 현재 경기의 원정팀이 과거에 원정에서 치른 경기의 HTR과 RESULT를 홈과 바꿈
        away_record.loc[(away_record['AwayTeam'] == away) & (away_record['HTR'] != 1), ['HTR']] = \
            3 - away_record.loc[(away_record['AwayTeam'] == away) & (away_record['HTR'] != 1), ['HTR']]
        away_record.loc[(away_record['AwayTeam'] == away) & (away_record['RESULT'] != 1), ['RESULT']] = \
            3 - away_record.loc[(away_record['AwayTeam'] == away) & (away_record['RESULT'] != 1), ['RESULT']]

        if index in self.skip:
            self.df.loc[[index], ["HTR"] + self.r_COL] = 0
            ratio = 0.5
        # 이전 10 경기 획득 승점
        if home_record.shape[0] >= 10:
            home_record = home_record[-10:]
        # 2부리그 경기에 대해 0.8의 가중치
        home_record.loc[home_record["Div"] == "E1", "RESULT"] *= 0.8
        self.df.loc[[index], "HP"] = home_record["RESULT"].sum(axis=0)

        if away_record.shape[0] >= 10:
            away_record = away_record[-10:]
        away_record.loc[away_record["Div"] == "E1", "RESULT"] *= 0.8
        self.df.loc[[index], "AP"] = away_record["RESULT"].sum(axis=0)

        if home_record.shape[0] >= 5:
            home_record = home_record[-5:]
        if away_record.shape[0] >= 5:
            away_record = away_record[-5:]
        # 현재 경기의 홈팀이 과거에 원정에서 치른 경기의 피처들을 홈을 기준으로 변환
        home_record.loc[home_record["AwayTeam"] == home, self.r_COL[:-1]] = -home_record.loc[
            home_record["AwayTeam"] == home, self.r_COL[:-1]]
        home_record.loc[home_record["AwayTeam"] == home, ["Pezzali"]] = 1 / home_record.loc[
            home_record["AwayTeam"] == home, ["Pezzali"]]
        INDEX = home_record["RESULT"].value_counts().index
        VALUES = home_record["RESULT"].value_counts().values
        for idx, val in zip(INDEX, VALUES):
            home_record.loc[home_record["RESULT"] == idx, self.r_COL] *= val
        # df.loc[[index], self.r_COL + ["HTR"]] += home_record[self.r_COL + ["HTR"]].ewm(
        #     span=home_record.shape[0]).mean().mean().values
        self.df.loc[[index], self.r_COL] += home_record[self.r_COL].sum(axis=0).values * ratio
        H_HTR = np.ravel(home_record["HTR"].mean(axis=0))

        # 현재 경기의 원정팀이 과거에 원정에서 치른 경기의 피처들을 홈을 기준으로 변환
        away_record.loc[away_record["AwayTeam"] == away, self.r_COL[:-1]] = -away_record.loc[
            away_record["AwayTeam"] == away, self.r_COL[:-1]]
        away_record.loc[away_record["AwayTeam"] == away, ["Pezzali"]] = 1 / away_record.loc[
            away_record["AwayTeam"] == away, ["Pezzali"]]
        INDEX2 = away_record["RESULT"].value_counts().index
        VALUES2 = away_record["RESULT"].value_counts().values
        for idx, val in zip(INDEX2, VALUES2):
            away_record.loc[away_record["RESULT"] == idx, self.r_COL] *= val
        # self.df.loc[[index], self.r_COL + ["HTR"]] += away_record[self.r_COL + ["HTR"]].ewm(
        #     span=away_record.shape[0]).mean().mean().values
        self.df.loc[[index], self.r_COL] -= away_record[self.r_COL].sum(axis=0).values * ratio
        A_HTR = np.ravel(away_record["HTR"].mean(axis=0))
        self.df.loc[[index], ["HTR"]] += (H_HTR - A_HTR + 3) / 2 * ratio

    # ------------------ 해당 경기 홈팀의 이전 홈 5경기 데이터 평균 값 - 해당 경기 원정팀이 이전 원정 5경기 데이터 평균 값 ------------------

    def Last_5_GF_GA(self, home, away, index, ratio=0.2):
        selected_df = self.data[self.data.index < index].copy()
        home_record = selected_df[selected_df['HomeTeam'] == home].copy()
        away_record = selected_df[selected_df['AwayTeam'] == away].copy()
        # 승점 등의 기준으 홈팀으로 설정되어 있으므로 원정팀에 해당하는 형식으로 바꿈
        away_record.loc[:, ["Pezzali"]] = 1 / away_record["Pezzali"]
        away_record.loc[away_record['HTR'] != 1, ['HTR']] = \
            3 - away_record.loc[away_record['HTR'] != 1, ['HTR']]
        away_record.loc[away_record['RESULT'] != 1, ['RESULT']] = \
            2 - away_record.loc[away_record['RESULT'] != 1, ['RESULT']]
        if home_record.shape[0] == 0 & away_record.shape[0] == 0:
            return
        if home_record.shape[0] >= 5:
            home_record = home_record[-5:]
        if away_record.shape[0] >= 5:
            away_record = away_record[-5:]
        # 해당 방식을 단독으로 사용할 경우
        # df.loc[[index], self.r_self.COL + ['HTR']] = 0
        # 현재 경기가 상대전적 데이터가 없는 팀간의 경기일 경우
        if index in self.skip:
            self.df.loc[[index], ["HTR"] + self.r_COL] = 0
            ratio = 1
        INDEX = home_record["RESULT"].value_counts().index
        VALUES = home_record["RESULT"].value_counts().values
        # 빈도에 따른 가중치 부여
        for idx, val in zip(INDEX, VALUES):
            home_record.loc[home_record["RESULT"] == idx, self.COL[:5] + ["Pezzali"]] *= val
        # df.loc[[index], self.r_self.COL] += home_record[self.COL[:5] + ["Pezzali"]].ewm(
        #     span=home_record.shape[0]).mean().sum().values * ratio
        self.df.loc[[index], self.r_COL] += home_record[self.COL[:5] + ["Pezzali"]].sum(axis=0).values * ratio
        self.df.loc[[index], self.COL[:5]] += home_record[self.COL[:5]].sum(axis=0).values
        INDEX2 = away_record["RESULT"].value_counts().index
        VALUES2 = away_record["RESULT"].value_counts().values
        for idx, val in zip(INDEX2, VALUES2):
            away_record.loc[away_record["RESULT"] == idx, self.COL[5:] + ["Pezzali"]] *= val
        # self.df.loc[[index], self.r_COL] -= away_record[self.COL[:5] + ["Pezzali"]].ewm(
        #     span=away_record.shape[0]).mean().sum().values * ratio
        self.df.loc[[index], self.r_COL] -= away_record[self.COL[5:] + ["Pezzali"]].sum(axis=0).values * ratio
        self.df.loc[[index], self.COL[:5]] += away_record[self.COL[5:]].sum(axis=0).values
        H_HTR = home_record[["HTR"]].mean(axis=0)
        A_HTR = away_record["HTR"].mean(axis=0)
        # 0 ~ 3으로 정규화
        val = (H_HTR - A_HTR + 3) / 2 * ratio
        self.df.loc[[index], "HTR"] += np.ravel(val)

    def remove_draw(self):
        # 무승부 데이터 제거
        self.df = self.df[self.df.RESULT != 1]

    def train_test_split(self):
        train = self.df.loc[self.df.index < train_index, ["HTR"] + self.r_COL]
        train_label = self.df.loc[self.df.index < train_index, ["RESULT"]]
        test = self.df.loc[self.df.index >= train_index, ["HTR"] + self.r_COL]
        test_label = self.df.loc[self.df.index >= train_index, ["RESULT"]]
        self.columns = train.columns
        print("------------------ trainset example ------------------\n", train.head(20))
        print("------------------ testset example ------------------\n", test.head(20))
        print("------------------ train_label counts ------------------\n", train_label.value_counts())
        print("------------------ test_label counts ------------------\n", test_label.value_counts())
        train = self.scaler.fit_transform(train)
        test = self.scaler.transform(test)

        return train, test, train_label, test_label

    def corr(self):
        sns.heatmap(data=self.df.corr(), annot=True, fmt=".2f")
        plt.savefig("corr.jpg")
        sns.pairplot(self.df, height=3, hue="RESULT")
        plt.savefig("pairplot.jpg")

    # ------------------ oversampling ------------------

    def oversampling(self, data, label, n=5):
        self.sampler.k_neighbors = n
        resampled_data, resampled_label = self.sampler.fit_resample(data, label)

        return resampled_data, resampled_label

    def Train(self, data, label):
        self.clf.fit(data, np.ravel(label))

    def prediction(self, data, label):
        print(classification_report(np.ravel(label), self.clf.predict(data)))
        print(confusion_matrix(np.ravel(label), self.clf.predict(data)))
        print("{}%".format(np.round(self.clf.score(data, np.ravel(label)) * 100, 3)))

    # ------------------ PCA ------------------

    def D_red(self, data1, data2, n=2):
        self.pca.n_components = n
        pca_train = self.pca.fit_transform(data1)
        pca_test = self.pca.transform(data2)
        print(np.round(self.pca.explained_variance_, 3))
        print(np.round(self.pca.explained_variance_ratio_, 3))

        return pca_train, pca_test

    def plot_dist(self, data, label, db=False):
        plt.xlim(data[:, 0].min(), data[:, 0].max() + 1)
        plt.ylim(data[:, 1].min(), data[:, 1].max() + 1)
        mglearn.discrete_scatter(data[:, 0], data[:, 1],
                                 np.ravel(label.values.reshape(1, -1)).astype(np.int32),
                                 alpha=0.7)
        plt.legend(["패", "무", "승"])
        if db:
            mglearn.plots.plot_2d_classification(self.clf, data, fill=True, alpha=.7)
        plt.show()


forest = RandomForestClassifier(random_state=42)
result_prediction = Predict_football(DATA, forest)

# 상대전적
for i, (home, away) in tqdm(enumerate(zip(DATA.HomeTeam[train_index:], DATA.AwayTeam[train_index:])), total=DATA[train_index:].shape[0]):
    result_prediction.H2H(home, away, i + train_index, 0.8)
# 홈, 원정 개별
for i, (home, away) in tqdm(enumerate(zip(DATA.HomeTeam[train_index:], DATA.AwayTeam[train_index:])), total=DATA[train_index:].shape[0]):
    result_prediction.Last_5_GF_GA(home, away, i + train_index, 0.2)

result_prediction.remove_draw()
train, test, train_label, test_label = result_prediction.train_test_split()
# resampled_train, resampled_train_label = result_prediction.oversampling(train, train_label, 2)
# pca_train, pca_test = result_prediction.D_red(train, test)
result_prediction.Train(train, train_label)
result_prediction.prediction(test, test_label)
# result_prediction.plot_dist(test, test_label)
