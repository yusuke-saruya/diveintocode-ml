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
import pandas as pd
from pandas import DataFrame
from scipy import integrate


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
    print("precision:{}".format(precision_score(y_test, y_pred, average='micro')))

    #recallの計算
    print("recall:{}".format(recall_score(y_test, y_pred, average='micro')))

    #f値の計算
    print("f値:{}".format(f1_score(y_test, y_pred, average='micro')))
    

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

        
class ScratchLogisticRegression():
    """
    ロジスティック回帰のスクラッチ実装

    Parameters
    ----------
    num_iter : int
      イテレーション数
    lr : float
      学習率
    no_bias : bool
      バイアス項を入れない場合はTrue
    verbose : bool
      学習過程を出力する場合はTrue
    C : int
        正則化パラメータ(デフォルト値10)

    Attributes
    ----------
    self.coef_ : 次の形のndarray, shape (n_features,)
      パラメータ
    self.loss : 次の形のndarray, shape (self.iter,)
      学習用データに対する損失の記録
    self.val_loss : 次の形のndarray, shape (self.iter,)
      検証用データに対する損失の記録

    """

    def __init__(self, num_iter, lr, no_bias, verbose, C=10):
        # ハイパーパラメータを属性として記録
        self.iter = num_iter                            #イテレーション数
        self.lr = lr                                             #学習率
        self.no_bias = no_bias                      #True:バイアス項なし、　False:バイアス項あり
        self.verbose = verbose                     #True:表示、False:非表示
        self.C = C                                            #正則化パラメータ（デフォルト値10）
        # 損失を記録する配列を用意
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)
        
        
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        線形回帰を学習する。検証用データが入力された場合はそれに対する損失と精度もイテレーションごとに計算する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
        y : 次の形のndarray, shape (n_samples, )
            学習用データの正解値
        X_val : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量
        y_val : 次の形のndarray, shape (n_samples, )
            検証用データの正解値
            
        Returns
        -------
        self : returns an instance of self.            
            
        """
        #pandasをnp.arrayに変換
        X = np.array(X)
        y = np.array(y)
        
        #学習データが一次元の場合、次元変換する
        if X.ndim == 1:
            X = X.reshape(len(X), 1)
        
        #バイアス項を含める場合は、
        if self.no_bias is False:
            #x0（全て1）を０列目に挿入する
            X_0 = np.insert(X, 0, 1, axis=1)
        #バイアス項を入れない場合、0を挿入する
        else:
            X_0 = np.insert(X, 0, 0, axis=1)

        #n(特徴量の数+1)を算出
        n = X_0.shape[1]

        # パラメータの初期化
        self.coef_ = np.random.rand(1, n)
                
        #イテレーションの数だけパラメータを更新する
        for i in range(self.iter):

            #仮定関数を求める
            hypothesis = self._linear_hypothesis(X_0)
            
            error = hypothesis - y

            # 損失関数
            cost_function = self.objective(hypothesis, y)
            
            #損失関数を記録する
            self.loss[i] = cost_function
            
            #最後の一回はパラメータ更新なし
            if i <= self.iter - 2:
                #最急降下法によりパラメータ更新
                self.coef_ = self._gradient_descent(X_0, error)
                
                #　一時的に学習したパラメータを退避（検証データで学習された際に元に戻す）
                tmp_coef_ = self.coef_

        if self.verbose:
            #verboseをTrueにした際は学習過程を出力
                map_result = map(str, self.loss)
                result = ',\n'.join(map_result)                
                print('学習データ　目的関数 : \n{}'.format(result))
                
        if X_val is not None:
            #pandasをnp.arrayに変換
            X_val = np.array(X_val)
            y_val = np.array(y_val)
            
            #学習データが一次元の場合、次元変換する
            if X_val.ndim == 1:
                X_val = X_val.reshape(len(X_val), 1)

            #バイアス項を含める場合は、
            if self.no_bias is False:
                #x0（全て1）を０列目に挿入する
                X_val_0 = np.insert(X_val, 0, 1, axis=1)
            #バイアス項を入れない場合、0を挿入する
            else:
                X_val_0 = np.insert(X_val, 0, 0, axis=1)

            #n(特徴量の数+1)を算出
            n = X_val_0.shape[1]

            # パラメータの初期化
            self.coef_ = np.random.rand(1, n)

            m = len(X_val_0)

            #イテレーションの数だけパラメータを更新する
            for i in range(self.iter):

                #仮定関数を求める
                hypothesis_val = self._linear_hypothesis(X_val_0)

                error_val = hypothesis_val - y_val

                # 損失関数
                cost_function_val = self.objective(hypothesis_val, y_val)

                #損失関数を記録する
                self.val_loss[i] = cost_function_val

                #最後の一回はパラメータ更新なし
                if i <= self.iter - 2:
                    #最急降下法によりパラメータ更新
                    self.coef_ = self._gradient_descent(X_val_0, error_val)

            if self.verbose:
                #verboseをTrueにした際は学習過程を出力
                map_result = map(str, self.val_loss)
                result = ',\n'.join(map_result)                
                print('検証データ　目的関数 : \n{}'.format(result))
        
        self.coef_ = tmp_coef_
    

    def predict(self, X):
        """
        ロジスティック回帰を使い分類する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
            次の形のndarray, shape (n_samples, 1)
            ロジスティック回帰による分類結果
            (予測確率が0.5以上であれば、1を返す)
        """
        #学習データが一次元の場合、次元変換する
        if X.ndim == 1:
            X = X.reshape(len(X), 1)
        
        #バイアス項を含める場合は、
        if self.no_bias is False:
            #x0（全て1）を０列目に挿入する
            X_0 = np.insert(X, 0, 1, axis=1)
        #バイアス項を入れない場合、0を挿入する
        else:
            X_0 = np.insert(X, 0, 0, axis=1)

        return (self._linear_hypothesis(X_0)>=0.5).astype(np.int)
    
    def predict_proba(self, X):
        """
        ロジスティック回帰を使い分類する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
            次の形のndarray, shape [n_samples, n_classes]
            ロジスティック回帰による分類結果
        """
        #学習データが一次元の場合、次元変換する
        if X.ndim == 1:
            X = X.reshape(len(X), 1)
        
        #バイアス項を含める場合は、
        if self.no_bias is False:
            #x0（全て1）を０列目に挿入する
            X_0 = np.insert(X, 0, 1, axis=1)
        #バイアス項を入れない場合、0を挿入する
        else:
            X_0 =np.insert(X, 0, 0, axis=1)
        
        classes0 = 1- self._linear_hypothesis(X_0)
        classes1 = self._linear_hypothesis(X_0)
        
        return np.concatenate([classes0, classes1], 1)
    
    
    #シグモイド関数
    def _sigmoid(self, z):
        """
        ロジスティック回帰のシグモイド関数を計算する

        Parameters
        ----------
        z : 次の形のndarray, shape (n_samples, 1)
          学習データ

        Returns
        -------
        sigmoid : 次の形のndarray, shape (n_samples, 1)
          入力値からシグモイド関数の結果を返す

        """        
        return 1 / (1 + np.exp(-z))        
        
        
        
    def _linear_hypothesis(self, X):
        """
        ロジスティック回帰の仮定関数を計算する

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ

        Returns
        -------
          次の形のndarray, shape (n_samples, 1)
          仮定関数による推定結果

        """
        #仮定関数を求める
        return self._sigmoid(np.dot(X, self.coef_.T))
    

    def objective(self, y_pred, y):
    
        """
        目的関数の計算

        Parameters
        ----------
        y : 次の形のndarray, shape (n_samples,)
          正解値

        Returns
        ----------
        objective : numpy.float
          目的関数の結果
        """    
        return (
            ((np.dot(-y.T, np.log(y_pred)) - np.dot((1 - y).T, np.log(1- y_pred))) / len(y))
                + (np.sum(self.coef_ ** 2) *  self.C / (2 * len(y)))
        )
    
    def _gradient_descent(self, X, error):
        """
        最急降下法にてパラメータを更新する

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ
        error : 次の形のndarray, shape (n_samples,)
          仮定関数から目的関数を差し引いたもの

        Returns
        ----------
        self.coef_ : 次の形のndarray, shape (n_features,)
          更新後のパラメータ
        """

        update_0 = ((np.dot(X.T, error)) / len(X))[0]

        update_j = (((np.dot(X.T, error)) / len(X)) + (((self.C / len(X)) * self.coef_).T))[1:].reshape(-1)

        update = np.concatenate([update_0, update_j])

        self.coef_ = self.coef_ - self.lr * update

        return self.coef_
    
    #学習曲線を表示する
    def plot_learning_curve(self):
        """
        学習曲線を表示する

        Parameters
        ----------

        Returns
        ----------
        学習時における損失の推移をプロットする
        fit時に検証データが入力されている場合、val_lossもプロットする
        """
        #trainデータの学習曲線を表示する
        plt.plot(self.loss, label='train_loss')
        
        #valデータが入力されている場合、表示する
        if np.all(self.val_loss!=0):
            plt.plot(self.val_loss, label='val_loss')

        plt.title('Learning curve')
        plt.xlabel('iter')
        plt.ylabel('loss')
        plt.legend()

        plt.show()

        
    #混同行列を計算する    
    def confusion_matrix(self, y_true, y_pred):
        """
        予測値より混同行列を計算する

        Parameters
        ----------
        y_true : 次の形のndarray, shape (n_samples,)
          正解値
        y_pred : 次の形のndarray, shape (n_samples,)
          予測値

        Returns
        ----------
        confusion_matrix : 次の形のndarray, shape (n_classes,n_classes)
          正解値と予測値における混同行列
        """
        #正解値と予測値を結合する
        y_true_and_pred = np.concatenate([y_true, y_pred], 1)

        # 目的変数でのユニークな要素をリスト化
        class_list = np.unique(y_true_and_pred)

        #クラスの数を数える
        n_classes = len(class_list)

        #クラスの数だけ空のの混同行列を作成
        c_matrix = np.zeros([n_classes,n_classes])

        #正解値と予測値の結果に応じて、混同行列を更新する
        for row, class1 in enumerate(class_list):
            for column, class2 in enumerate(class_list):
                for i in range(len(y_true)):
                    if (y_true[i] == class_list[class1]) and (y_pred[i] == class_list[class2]):
                        c_matrix[row][column] += 1
                
        return c_matrix

    
    def accuracy_score(self, y_true, y_pred):
        """
        予測値より正解率(accuracy)を計算する

        Parameters
        ----------
        y_true : 次の形のndarray, shape (n_samples,)
          正解値
        y_pred : 次の形のndarray, shape (n_samples,)
          予測値

        Returns
        ----------
        accuracy_score : float
          正解値と予測値における正解率(Accuracy)
        """
        
        #混同行列を算出
        conf_mat = self.confusion_matrix(y_true, y_pred)
        
        #混同行列の合計を算出
        sumconf = np.sum(conf_mat)
        
        #TP+TNを計算する
        tp_tn = 0
        for i in range(len(conf_mat)):
            tp_tn += conf_mat[i][i]

        #accuracy_scoreを算出
        return tp_tn / sumconf

    def precision_score(self, y_true, y_pred):
        """
        予測値より精度(precision)を計算する
        ※二値分類のみ対応

        Parameters
        ----------
        y_true : 次の形のndarray, shape (n_samples,)
          正解値
        y_pred : 次の形のndarray, shape (n_samples,)
          予測値

        Returns
        ----------
        precision_score : float
          正解値と予測値における精度(Precision)
        """
        
        #混同行列を算出
        conf_mat = self.confusion_matrix(y_true, y_pred)
        
        #TP+FPを計算
        tp_fp = 0
        for i in range(len(conf_mat)):
            tp_fp += conf_mat[i][0]
            
        #TPをセット
        tp = conf_mat[0][0]
        
        #precisionを算出
        return tp / tp_fp
    

    def recall_score(self, y_true, y_pred):
        """
        予測値より検出率(recall)を計算する
        ※二値分類のみ対応

        Parameters
        ----------
        y_true : 次の形のndarray, shape (n_samples,)
          正解値
        y_pred : 次の形のndarray, shape (n_samples,)
          予測値

        Returns
        ----------
        recall_score : float
          正解値と予測値における検出率(Recall)
        """
        
        #混同行列を算出
        conf_mat = self.confusion_matrix(y_true, y_pred)
        
        #TP+FNを計算
        tp_fn = 0
        for i in range(len(conf_mat)):
            tp_fn += conf_mat[0][i]
            
        #TPをセット
        tp = conf_mat[0][0]
        
        #recallを算出
        return tp / tp_fn



    def f1_score(self, y_true, y_pred):
        """
        予測値よりF値を計算する
        ※二値分類のみ対応
        F1 = 2 * (precision * recall) / (precision + recall)

        Parameters
        ----------
        y_true : 次の形のndarray, shape (n_samples,)
          正解値
        y_pred : 次の形のndarray, shape (n_samples,)
          予測値

        Returns
        ----------
        recall_score : float
          正解値と予測値における検出率(Recall)
        """
        #精度を計算        
        precision = self.precision_score(y_true, y_pred)
        
        #検出率を計算
        recall = self.recall_score(y_true, y_pred)
               
        #F値を計算する
        return 2 * (precision * recall) / (precision + recall)
    

    def decision_region(self, X_train, y_train, step=0.01, title='decision region', xlabel='xlabel', ylabel='ylabel', target_names=['versicolor', 'virginica']):
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
        pred = self.predict(mesh).reshape(mesh_f0.shape)

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
        

        
        
class ScratchSVMClassifier():
    """
    SVM分類のスクラッチ実装

    Parameters
    ----------
    kernel : str
        カーネルを選択する（linear,,,）(デフォルト'linear')
    num_iter : int
      イテレーション数(学習回数)(デフォルト10)
    lr : float
      学習率（デフォルト0.1）
    C : float
        正則化パラメータ(デフォルト値10)
    threshold : flaot
        サポートベクターを決定する閾値（デフォルト値1e-5）

    Attributes
    ----------
    self.coef_ : 次の形のndarray, shape (n_features,)
      パラメータ
    self.intercept_ 次の形のndarray, shape (1,)
     切片
    self.X_sv : 次の形のndarry, shape(n_samples(サポートベクトルの数), n_classes)
        サポートベクトル
    """

    def __init__(self, kernel='linear', num_iter=10, lr=0.1, C=10, threshold=1e-5, gamma=1, coef0=0, d=2):
        # ハイパーパラメータを属性として記録
        self.kernel = kernel                             #カーネル関数を選択（linear:線形カーネル）
        self.iter = num_iter                            #イテレーション数
        self.lr = lr                                             #学習率
        self.C = C                                            #正則化パラメータ（デフォルト値10）
        self.threshold = threshold                #サポートベクターを決定する閾値(デフォルト値1e-5)
        self.gamma = gamma                       #多項式カーネル関数の係数(デフォルト値1)
        self.coef0 = coef0                            #多項式カーネルの切片(デフォルト値0)
        self.d = d                                            #多項式カーネルの累乗数(デフォルト値1)
                
    def fit(self, X, y):
        """
        SVM分類を学習する。
        
        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
        y : 次の形のndarray, shape (n_samples, )
            学習用データの正解値(二値)
            
        Returns
        -------
        self : returns an instance of self.            
            
        """
        #pandasをnp.arrayに変換
        X = np.array(X)
        y = np.array(y)
        
        #predict用にy(目的変数)のクラスを取得
        self.y_classes = np.unique(y)
        
        #正解値が-1と1のクラスになっていない場合、変換する
        if self.y_classes[0] != -1 or self.y_classes[1] != 1:
            #目的変数の0番目を-1,1番目を1に置換
            y  = np.where(y==self.y_classes[0], -1, 1)
        
        #学習データが一次元の場合、次元変換する
        if X.ndim == 1:
            X = X.reshape(len(X), 1)
                    
        #選択したカーネルを関数に渡す        
        if self.kernel == 'linear':
            #線形カーネル関数
            self.kernel_object = self._linear_kernel(X, X)
        elif self.kernel == 'poly':
            self.kernel_object = self._poly_kernel(X, X)
        
        #最急降下法にてパラメータを更新する
        lam = self._gradient_descent(X, y)
                            
        # しきい値以上の値となったラグランジュ乗数をサポートベクトルとして取り出す
        sv = np.where(lam > self.threshold)[0]

        # サポートベクトルの数
        nsupport = len(sv)

        #サポートベクトルの配列を作成
        X_sv = X[sv,:]
        lam_sv = lam[sv]
        y_sv = y[sv]
        
        #グラフ表示用にサポートベクトルをattribute化
        self.X_sv = X_sv.copy()
        self.lam_sv = lam_sv.copy()
        self.y_sv = y_sv.copy()
        
        """退避
        #パラメータを初期化
        self.coef_ = 0
        
        #thetaパラメータを更新
        for i in range(nsupport):
            self.coef_ += lam_sv[i] * y_sv[i] * X_sv[i]
        
        #切片theta0を更新
        self.intercept_ = np.sum(y_sv - (np.dot(X_sv, self.coef_.reshape(X.shape[1],1)))) / nsupport
        """
    

    def predict(self, X):
        """
        SVMを使い分類予測する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル
        y : 次の形のndarray, shape (n_samples, )
            正解値

        Returns
        -------
            次の形のndarray, shape (n_samples, 1)
            svmによる分類結果
            (予測確率が0より小さい場合-1、そうでない場合1)
        """
        #選択したカーネルを関数に渡す        
        if self.kernel == 'linear':
            #線形カーネル関数
            estimate_kernel = self._linear_kernel(X, self.X_sv)
        elif self.kernel == 'poly':
            estimate_kernel = self._poly_kernel(X, self.X_sv)

        #推定を行う
        estimate = np.zeros((X.shape[0],))
        for i in range(len(self.lam_sv)):
            estimate += self.lam_sv[i] * self.y_sv[i] * estimate_kernel[:, i]


        #決定関数より分類結果を返す
        return np.where(estimate.reshape(X.shape[0],) < 0, self.y_classes[0], self.y_classes[1])

        

        
    def _linear_kernel(self, Xi, Xj):
        """
        線形カーネル関数を計算する

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ

        Returns
        -------
          次の形のndarray, shape (n_samples, 1)
          仮定関数による推定結果

        """
        #線形カーネル関数を求める
        return np.dot(Xi, Xj.T)

    def _poly_kernel(self, Xi, Xj):
        """
        多項式カーネル関数を計算する

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ

        Returns
        -------
          次の形のndarray, shape (n_samples, 1)
          仮定関数による推定結果

        """
        #多項式カーネル関数を求める
        return self.gamma * ((np.dot(Xi, Xj.T) + self.coef0) ** self.d)

  
    def _gradient_descent(self, X, y):
        """
        最急降下法にてパラメータを更新する

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ
        y : 次の形のndarray, shape (n_samples, )
            正解値

        Returns
        ----------
        self.coef_ : 次の形のndarray, shape (n_features,)
          更新後のパラメータ
        """
        #サンプル数を一時変数に格納（処理が早くなる）
        n_samples = X.shape[0]
        
        #ラグランジュ乗数を初期化する
        lam = np.ones((n_samples,1))

        #イテレーション数だけ学習を繰り返す
        for count in range(self.iter):
            #パラメータの更新式のsigma以降の計算を行う
            for i in range(n_samples):
                tmp_lam = 0
                for j in range(n_samples):
                    tmp_lam += lam[j] * y[i] * y[j] * self.kernel_object[i, j]

                # サンプルごとのラムダを更新する
                lam[i] += self.lr *(1 - tmp_lam)

                # ラムダが0より小さい場合、0に更新する
                if lam[i] < 0:
                    lam[i] = 0

        #学習後のラグランジュ乗数を返す
        return lam
    
        
    
    def decision_region(self, X_train, y_train, step=0.01, title='decision region', xlabel='xlabel', ylabel='ylabel', target_names=['target1', 'target2']):
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
        pred = self.predict(mesh).reshape(mesh_f0.shape)
        
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
        
        #サポートベクターをプロットする
        plt.scatter(self.X_sv[:, 0], self.X_sv[:, 1], color='y')

        plt.show()





            
if __name__ == '__main__':
    logistic_regression()