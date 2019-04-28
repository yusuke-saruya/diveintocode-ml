#ライブラリインポート
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import numpy as np

# スコア（指標値）出力
def score_print(y_test, y_pred):
    """
    算出した予測値をもとに４種類の指標値を出力する

    Parameters
    ----------------
    y_test : ndarray, shape(n_samples)
        正解値
    y_pred : ndarray, shape(n_samples)
        予測値
    
    Return
    ----------------
    score(accuracy,precision,recall,f値)

    """
    #accuracyの計算
    print("accuracy:{}".format(accuracy_score(y_test, y_pred)))

    #precisionの計算
    print("precision:{}".format(precision_score(y_test, y_pred)))

    #recallの計算
    print("recall:{}".format(recall_score(y_test, y_pred)))

    #f値の計算
    print("f値:{}".format(f1_score(y_test, y_pred)))
    

#決定領域の出力    
def decision_region(X_train, y_train, model, step=0.01, title='decision region', xlabel='feature1', ylabel='feature2', target_names=['target1', 'target2']):
    """
    2値分類を2次元の特徴量で学習したモデルの決定領域を描く。
    背景の色が学習したモデルによる推定値から描画される。
    散布図の点は学習用データである。

    Parameters
    ----------------
    X_train : ndarray, shape(n_samples, 2)
        学習用データの特徴量
    y_train : ndarray, shape(n_samples,)
        学習用データの正解値
    model : object
        学習したモデルのインスンタスを入れる
    step : float, (default : 0.1)
        推定値を計算する間隔を設定する
    title : str
        グラフのタイトルの文章を与える
    xlabel, ylabel : str
        軸ラベルの文章を与える
    target_names= : list of str
        凡例の一覧を与える
    """
    # setting
    scatter_color = ['red', 'blue']
    contourf_color = ['pink', 'skyblue']
    n_class = 2

    # pred
    mesh_f0, mesh_f1  = np.meshgrid(np.arange(np.min(X_train[:,0])-0.5, np.max(X_train[:,0])+0.5, step), np.arange(np.min(X_train[:,1])-0.5, np.max(X_train[:,1])+0.5, step))
    mesh = np.c_[np.ravel(mesh_f0),np.ravel(mesh_f1)]
    pred = model.predict(mesh).reshape(mesh_f0.shape)

    # plot
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.contourf(mesh_f0, mesh_f1, pred, n_class-1, cmap=ListedColormap(contourf_color))
    plt.contour(mesh_f0, mesh_f1, pred, n_class-1, colors='y', linewidths=3, alpha=0.5)
    for i, target in enumerate(set(y_train)):
        plt.scatter(X_train[y_train==target][:, 0], X_train[y_train==target][:, 1], s=80, color=scatter_color[i], label=target_names[i], marker='o')
    patches = [mpatches.Patch(color=scatter_color[i], label=target_names[i]) for i in range(n_class)]
    plt.legend(handles=patches)
    plt.legend()
    plt.show()
    

#ロジスティック回帰
def logistic_regression(X_train, X_test, y_train, y_test):

    #インスタンス生成
    log_model = LogisticRegression(solver='lbfgs')
    
    #モデル作成
    log_model.fit(X_train, y_train)

    #検証用データを予測
    y_pred_LR = log_model.predict(X_test)
    
    #指標値を出力
    print("<LogisticRegression>")
    score_print(y_test, y_pred_LR)
    
    #特徴量が二つの場合、決定領域を表示する
    if X_train.shape[1] == 2:
        decision_region(
            X_train=X_train, 
            y_train=y_train.reshape(-1),   #set型に対応するために一次元配列にする
            model=log_model, #インスタンスの決定領域を表示
            title='decision region of LogisticRegression', 
        )
    
# SVM    
def svm(X_train, X_test, y_train, y_test):
    
    #インスタンス生成
    SVM_model = SVC(gamma='auto')
    
    #モデル作成
    SVM_model.fit(X_train, y_train.reshape(-1)) 

    #検証用データを予測
    y_pred_SVM = SVM_model.predict(X_test)

    #指標値を出力
    print("<SVM>")
    score_print(y_test, y_pred_SVM)
    
    #特徴量が二つの場合、決定領域を表示する
    if X_train.shape[1] == 2:
        decision_region(
            X_train=X_train, 
            y_train=y_train.reshape(-1),   #set型に対応するために一次元配列にする
            model=SVM_model, #インスタンスの決定領域を表示
            title='decision region of SVC', 
        )


#決定木
def decision_tree(X_train, X_test, y_train, y_test):
    
    #インスタンス生成
    DTC_model = DecisionTreeClassifier()

    #学習用データでモデル作成
    DTC_model.fit(X_train, y_train) 

    #検証用データを予測
    y_pred_DTC = DTC_model.predict(X_test)

    #指標値を出力
    print("<DecisionTreeClassifier>")
    score_print(y_test, y_pred_DTC)

    #特徴量が二つの場合、決定領域を表示する
    if X_train.shape[1] == 2:
        decision_region(
            X_train=X_train, 
            y_train=y_train.reshape(-1),   #set型に対応するために一次元配列にする
            model=DTC_model, #インスタンスの決定領域を表示
            title='decision region of DecisionTreeClassifier', 
        )

            
if __name__ == '__main__':
    logistic_regression()