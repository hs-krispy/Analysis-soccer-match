import pandas as pd
import os
import mglearn
import seaborn as sns
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
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
add = 0.1

train = pd.read_csv("Raw train data")
test = pd.read_csv("Raw test data")
train.drop(columns=["FTR"], inplace=True)
test.drop(columns=["FTR"], inplace=True)
# train.loc[train.RESULT == 1, "RESULT"] = 0
train.loc[train["HTR"] == "A", "HTR"] = test.loc[test["HTR"] == "A", "HTR"] = 0
train.loc[train["HTR"] == "D", "HTR"] = test.loc[test["HTR"] == "D", "HTR"] = 1
train.loc[train["HTR"] == "H", "HTR"] = test.loc[test["HTR"] == "H", "HTR"] = 3
train["HTR"] = train["HTR"].astype(np.float32)
train["2HTHG"], test["2HTHG"] = train["FTHG"] - train["HTHG"], test["FTHG"] - test["HTHG"]
train["2HTAG"], test["2HTAG"] = train["FTAG"] - train["HTAG"], test["FTAG"] - test["HTAG"]
train["FGD"], test["FGD"] = train["FTHG"] - train["FTAG"], test["FTHG"] - test["FTAG"]
train["2HGD"], test["2HGD"] = train["2HTHG"] - train["2HTAG"], test["2HTHG"] - test["2HTAG"]
train["HGD"], test["HGD"] = train["HTHG"] - train["HTAG"], test["HTHG"] - test["HTAG"]
train["SD"], test["SD"] = train["HS"] - train["AS"], test["HS"] - test["AS"]
train["STD"], test["STD"] = train["HST"] - train["AST"], test["HST"] - test["AST"]
train["Pezzali"] = (train["FTHG"] + add) / (train["HS"] + add) * (train["AS"] + add) / (train["FTAG"] + add)
test["Pezzali"] = (test["FTHG"] + add) / (test["HS"] + add) * (test["AS"] + add) / (test["FTAG"] + add)


class Predict_football():
    def __init__(self, raw_train, raw_test, model):
        plt.rc("font", family="Malgun Gothic")
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(10, 10))
        self.train = raw_train.copy()
        self.test = raw_test.copy()
        self.scaler = StandardScaler()
        self.sampler = SMOTE(random_state=42)
        self.pca = PCA(random_state=42)
        self.skip = []
        self.columns = None
        self.clf = model
        self.COL = ["FTHG", "2HTHG", "HTHG", "HS", "HST", "FTAG", "2HTAG", "HTAG", "AS", "AST"]
        self.r_COL = ["FGD", "2HGD", "HGD", "SD", "STD", "Pezzali"]

    # ------------------ ANOVA ------------------

    def ANOVA(self, data):
        df = data.drop(columns=["Div", "HomeTeam", "AwayTeam", "RESULT"])
        df = df.iloc[:train_index, :]
        scaled_train = self.scaler.fit_transform(df)
        scaled_train = pd.DataFrame(scaled_train, columns=df.columns)
        df = self.sampler.fit_resample(scaled_train, data.loc[:train_index, "RESULT"])
        df = pd.concat([df[0], df[1]], axis=1)
        fstat, p_val = f_oneway(df.loc[df["RESULT"] == 0, df.columns[:-1]],
                                df.loc[df["RESULT"] == 1, df.columns[:-1]],
                                df.loc[df["RESULT"] == 2, df.columns[:-1]])
        print(p_val)
        print(df.columns[:-1][p_val > 0.05])

    # ------------------ Post-hoc ------------------

    def PH(self, data):
        df = data.drop(columns=["Div", "HomeTeam", "AwayTeam", "RESULT"])
        for i in df.columns:
            posthoc = pairwise_tukeyhsd(data.iloc[:, [i]], data["RESULT"], alpha=0.05)
            plt.figure(figsize=(10, 10))
            posthoc.plot_simultaneous()
            plt.title("{}".format(data.columns[i]))
            plt.show()

    # ------------------ plot data pdf(probability density function) ------------------

    def plot(self, data):
        res = ["패", "무", "승"]
        color = ["r", "g", "b"]
        y_Max = 0
        x_Max = 0
        x_Min = 100
        for col in self.r_COL + ["HTR"]:
            for i in range(3):
                values = data[data["RESULT"] == i][col].value_counts().values / \
                         data[data["RESULT"] == i][col].shape[0]
                y_Max = max(y_Max, values.max())
                x_Max = max(x_Max, data[data["RESULT"] == i][col].max())
                x_Min = min(x_Min, data[data["RESULT"] == i][col].min())
            fig, axes = plt.subplots(1, 3, figsize=(15, 8))
            for i, (r, ax, c) in enumerate(zip(res, axes, color)):
                print(data[data["RESULT"] == i][col].value_counts())
                index = data[data["RESULT"] == i][col].value_counts().index.tolist()
                values = data[data["RESULT"] == i][col].value_counts().values / \
                         data[data["RESULT"] == i][col].shape[0]
                ax.bar(index, values, color=c, label="{}".format(r))
                ax.set_xlabel("{}".format(col))
                ax.set_ylabel("bins")
                ax.set_xlim(x_Min, x_Max)
                ax.set_ylim(0, y_Max)
                ax.legend()
            plt.show()

    # ------------------ 해당 경기 홈팀과 원정팀의 이전 5경기 맞대결 데이터들의 평균 값 ------------------

    def H2H(self, home, away, index, ratio=1):
        record = self.train[((self.train['HomeTeam'] == home) & (self.train['AwayTeam'] == away)) | (
                (self.train['HomeTeam'] == away) & (self.train['AwayTeam'] == home))].copy()
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
        self.test.loc[[index], self.r_COL] = record[self.r_COL].sum(axis=0).values * ratio
        self.test.loc[[index], self.COL] = record[self.COL].sum(axis=0).values * ratio
        # self.test.loc[[index], self.r_self.COL] = record[self.r_self.COL].ewm(span=record.shape[0], adjust=True).mean().sum().values * ratio
        # self.test.loc[[index], self.COL] = record[self.COL].ewm(span=record.shape[0], adjust=True).mean().mean().values * ratio
        self.test.loc[[index], ["HTR"]] = np.ravel(record["HTR"].mean(axis=0)) * ratio

    # ------------------ 해당 경기 홈팀의 이전 5경기 데이터 평균 값 - 해당 경기 원정팀의 이전 5경기 데이터 평균 값  ------------------

    def Last_5(self, home, away, index, ratio=0.2):
        home_record = self.train[((self.train['HomeTeam'] == home) | (self.train['AwayTeam'] == home))].copy()
        away_record = self.train[((self.train['HomeTeam'] == away) | (self.train['AwayTeam'] == away))].copy()
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
            self.test.loc[[index], ["HTR"] + self.r_COL] = 0
            ratio = 0.5

        # 이전 10 경기 획득 승점
        if home_record.shape[0] >= 10:
            home_record = home_record[-10:]
        # 2부리그 경기에 대해 0.8의 가중치
        home_record.loc[home_record["Div"] == "E1", "RESULT"] *= 0.8
        self.test.loc[[index], "HP"] = home_record["RESULT"].sum(axis=0)

        if away_record.shape[0] >= 10:
            away_record = away_record[-10:]
        away_record.loc[away_record["Div"] == "E1", "RESULT"] *= 0.8
        self.test.loc[[index], "AP"] = away_record["RESULT"].sum(axis=0)

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
        self.test.loc[[index], self.r_COL] += home_record[self.r_COL].sum(axis=0).values * ratio
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
        # self.test.loc[[index], self.r_COL + ["HTR"]] += away_record[self.r_COL + ["HTR"]].ewm(
        #     span=away_record.shape[0]).mean().mean().values
        self.test.loc[[index], self.r_COL] -= away_record[self.r_COL].sum(axis=0).values * ratio
        A_HTR = np.ravel(away_record["HTR"].mean(axis=0))
        self.test.loc[[index], ["HTR"]] += (H_HTR - A_HTR + 3) / 2 * ratio

    # ------------------ 해당 경기 홈팀의 이전 홈 5경기 데이터 평균 값 - 해당 경기 원정팀이 이전 원정 5경기 데이터 평균 값 ------------------

    def Last_5_GF_GA(self, home, away, index, ratio=0.2):
        home_record = self.train[self.train['HomeTeam'] == home].copy()
        away_record = self.train[self.train['AwayTeam'] == away].copy()
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
            self.test.loc[[index], ["HTR"] + self.r_COL] = 0
            ratio = 1
        INDEX = home_record["RESULT"].value_counts().index
        VALUES = home_record["RESULT"].value_counts().values
        # 빈도에 따른 가중치 부여
        for idx, val in zip(INDEX, VALUES):
            home_record.loc[home_record["RESULT"] == idx, self.COL[:5] + ["Pezzali"]] *= val
        # df.loc[[index], self.r_self.COL] += home_record[self.COL[:5] + ["Pezzali"]].ewm(
        #     span=home_record.shape[0]).mean().sum().values * ratio
        self.test.loc[[index], self.r_COL] += home_record[self.COL[:5] + ["Pezzali"]].sum(axis=0).values * ratio
        self.test.loc[[index], self.COL[:5]] += home_record[self.COL[:5]].sum(axis=0).values
        INDEX2 = away_record["RESULT"].value_counts().index
        VALUES2 = away_record["RESULT"].value_counts().values
        for idx, val in zip(INDEX2, VALUES2):
            away_record.loc[away_record["RESULT"] == idx, self.COL[5:] + ["Pezzali"]] *= val
        # self.test.loc[[index], self.r_COL] -= away_record[self.COL[:5] + ["Pezzali"]].ewm(
        #     span=away_record.shape[0]).mean().sum().values * ratio
        self.test.loc[[index], self.r_COL] -= away_record[self.COL[5:] + ["Pezzali"]].sum(axis=0).values * ratio
        self.test.loc[[index], self.COL[:5]] += away_record[self.COL[5:]].sum(axis=0).values
        H_HTR = home_record[["HTR"]].mean(axis=0)
        A_HTR = away_record["HTR"].mean(axis=0)
        # 0 ~ 3으로 정규화
        val = (H_HTR - A_HTR + 3) / 2 * ratio
        self.test.loc[[index], "HTR"] += np.ravel(val)

    def remove_draw(self):
        # 무승부 데이터 제거
        self.train = self.train[self.train.RESULT != 1]
        self.test = self.test[self.test.RESULT != 1]

    def making_train_test(self):
        train = self.train[["HTR"] + self.r_COL]
        train_label = self.train[["RESULT"]]
        test = self.test[["HTR"] + self.r_COL]
        test_label = self.test[["RESULT"]]
        self.columns = train.columns
        print("------------------ trainset example ------------------\n", train.head(20))
        print("------------------ testset example ------------------\n", test.head(20))
        print("------------------ train_label counts ------------------\n", train_label.value_counts())
        print("------------------ test_label counts ------------------\n", test_label.value_counts())
        train = self.scaler.fit_transform(train)
        test = self.scaler.transform(test)

        return train, test, train_label, test_label

    def corr(self, data):
        sns.heatmap(data=data.corr(), annot=True, fmt=".2f")
        plt.savefig("corr.jpg")
        sns.pairplot(data, height=3, hue="RESULT")
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
result_prediction = Predict_football(train, test, forest)
# 예측할 데이터에 대해 home, away 팀을 key로 test data 생성
# 상대전적
for i, (home, away) in tqdm(enumerate(zip(test.HomeTeam, test.AwayTeam)), total=test.shape[0]):
    result_prediction.H2H(home, away, i + train_index, 0.8)
# 홈, 원정 개별
for i, (home, away) in tqdm(enumerate(zip(test.HomeTeam, test.AwayTeam)), total=test.shape[0]):
    result_prediction.Last_5_GF_GA(home, away, i + train_index, 0.2)

result_prediction.remove_draw()
train, test, train_label, test_label = result_prediction.making_train_test()
# resampled_train, resampled_train_label = result_prediction.oversampling(train, train_label, 2)
# pca_train, pca_test = result_prediction.D_red(train, test)
result_prediction.Train(train, train_label)
result_prediction.prediction(test, test_label)
# result_prediction.plot_dist(test, test_label)
