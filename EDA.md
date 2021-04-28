## EDA

승, 무, 패 세 가지 경우에 대해 각각 특정 column 별 확률밀도 확인

```python
def plot(column):
    fig, axes = plt.subplots(1, 3, figsize=(12, 8))
    for i, (r, ax, c) in enumerate(zip(res, axes, color)):
        index = TEST[TEST["RESULT"] == i][column].value_counts().index
        values = TEST[TEST["RESULT"] == i][column].value_counts().values / TEST[TEST["RESULT"] == i][
            column].value_counts().values.sum()
        ax.bar(index, values, color=c, label="{}".format(r))
        ax.set_xlabel("{}".format(column))
        ax.set_ylabel("bins")
        ax.set_xlim(0, 3)
        ax.legend()
    plt.show()


plot("H2H_record")
```

최근 5경기 상대전적에 대한 시각화

<img src="https://user-images.githubusercontent.com/58063806/116423984-1e23b600-a87c-11eb-91c0-fc9256a0200e.png" width=80% />