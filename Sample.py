import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, metrics


if __name__ == '__main__':
    # csv読み込み
    data = pd.read_csv('sample.csv')
    # ラベル抽出
    label = data['room']
    # 訓練データの前処理（要らない特徴量の削除）
    data2 = data[["mean", "std", "min", "median", "max", "diff",
                  "preMean", "preStd", "preMin", "preMedian", "preMax", "preDiff"]]
    # 訓練用と試験用のデータ生成
    train_x, test_y, ans_x, ans_y = cross_validation.train_test_split(data2, label, test_size=0.2, random_state=0)
    # n_estimators => 作成する木の数
    random_forest = RandomForestClassifier(n_estimators=10)
    # 学習
    random_forest.fit(train_x, ans_x)
    # 予測
    predicted = random_forest.predict(test_y)
    # 結果
    print(metrics.confusion_matrix(predicted, ans_y))
    print(metrics.accuracy_score(predicted, ans_y))
