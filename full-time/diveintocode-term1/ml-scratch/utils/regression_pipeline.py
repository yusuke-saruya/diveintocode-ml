from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error


def regression(X_train, X_test, y_train, y_test, reg):
    #回帰モデルを生成する
    reg.fit(X_train, y_train)

    # 回帰モデルを利用して予測する
    y_train_pred = reg.predict(X_train)
    y_test_pred = reg.predict(X_test)

    # 学習用、検証用データに関して平均二乗誤差を出力
    print('[MSE] 学習用データ : %.3f, 検証用データ : %.3f' % (
        mean_squared_error(
            y_train, 
            y_train_pred), 
        mean_squared_error(
            y_test, 
            y_test_pred)
    ))
    # 学習用、検証用データに関してR^2を出力
    print('[R^2] 学習用データ : %.3f, 検証用データ : %.3f' % (
        r2_score(
            y_train, 
            y_train_pred), 
        r2_score(
            y_test, 
            y_test_pred)
    ))

    '''
    残差プロットを表示する
    '''

    # 学習用、検証用それぞれで残差をプロット
    plt.scatter(y_train_pred, y_train_pred - y_train, c = 'blue', marker = 'o', label = 'Train Data')
    plt.scatter(y_test_pred, y_test_pred - y_test, c = 'lightgreen', marker = 's', label = 'Test Data')

    #x,yラベルを設定
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')

    # 凡例を左上に表示
    plt.legend(loc = 'upper left')

    # y = 0に直線を引く
    plt.hlines(y = 0, xmin = 0, xmax = 600000, lw = 2, color = 'red')
    plt.xlim([0, 600000])

    plt.show()



    
class ScratchLinearRegression():
    """
    線形回帰のスクラッチ実装

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

    Attributes
    ----------
    self.coef_ : 次の形のndarray, shape (n_features,)
      パラメータ
    self.loss : 次の形のndarray, shape (self.iter,)
      学習用データに対する損失の記録
    self.val_loss : 次の形のndarray, shape (self.iter,)
      検証用データに対する損失の記録

    """

    def __init__(self, num_iter, lr, bias, verbose):
        # ハイパーパラメータを属性として記録
        self.iter = num_iter
        self.lr = lr
        self.bias = bias
        self.verbose = verbose
        # 損失を記録する配列を用意
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)
    
    
        
    def _linear_hypothesis(self, X):
        """
        線形の仮定関数を計算する

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ

        Returns
        -------
          次の形のndarray, shape (n_samples, 1)
          線形の仮定関数による推定結果

        """
        #仮定関数を求める
        hypothesis = np.dot(X, self.coef_.T).reshape(-1)

        return hypothesis
    
    
    
    def MSE(self, y_pred, y):
        """
        平均二乗誤差の計算

        Parameters
        ----------
        y_pred : 次の形のndarray, shape (n_samples,)
          推定した値
        y : 次の形のndarray, shape (n_samples,)
          正解値

        Returns
        ----------
        mse : numpy.float
          平均二乗誤差
        """    
        mse = np.sum((y_pred - y) ** 2 ) / 2 * len(y)

        return mse
    
    
    
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

        #最急降下法によりパラメータ更新
        self.coef_ = self.coef_ - (np.dot(error, X) * self.lr / len(X))

        return self.coef_


    
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
        #バイアス項を含める場合は、
        if self.bias is False:
            #x0（全て1）を０列目に挿入する
            X_0 = np.insert(X, 0, 1, axis=1)
        #バイアス項を入れない場合はそのまま
        else:
            X_0 = X

        #n(特徴量の数+1)を算出
        n = X_0.shape[1]

        # パラメータの初期化
        self.coef_ = np.random.rand(1, n)
        
        m = len(X_0)
        
        #イテレーションの数だけパラメータを更新する
        for i in range(self.iter):

            #仮定関数を求める
            hypothesis = self._linear_hypothesis(X_0)
            
            error = hypothesis - y

            # 損失関数
            cost_function = self.MSE(hypothesis, y)
            
            #損失関数を記録する
            self.loss[i] = cost_function
            
            #最後の一回はパラメータ更新なし
            if i <= self.iter - 1:
                #最急降下法によりパラメータ更新
                self.coef_ = self._gradient_descent(X_0, error)

        if self.verbose:
            #verboseをTrueにした際は学習過程を出力
                map_result = map(str, self.loss)
                result = ',\n'.join(map_result)                
                print('学習データ　MSE : \n{}'.format(result))
                
        if X_val is not None:
            #バイアス項を含める場合は、
            if self.bias is False:
                #x0（全て1）を０列目に挿入する
                X_val_0 = np.insert(X_val, 0, 1, axis=1)
            #バイアス項を入れない場合はそのまま
            else:
                X_val_0 = X_val

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
                cost_function_val = self.MSE(hypothesis_val, y_val)

                #損失関数を記録する
                self.val_loss[i] = cost_function_val

                #最後の一回はパラメータ更新なし
                if i <= self.iter - 1:
                    #最急降下法によりパラメータ更新
                    self.coef_ = self._gradient_descent(X_val_0, error_val)

            if self.verbose:
                #verboseをTrueにした際は学習過程を出力
                map_result = map(str, self.val_loss)
                result = ',\n'.join(map_result)                
                print('検証データ　MSE : \n{}'.format(result))
        
    

    def predict(self, X):
        """
        線形回帰を使い推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
            次の形のndarray, shape (n_samples, 1)
            線形回帰による推定結果
        """
        #バイアス項を含める場合は、
        if self.bias is False:
            #x0（全て1）を０列目に挿入する
            X_0 = np.insert(X, 0, 1, axis=1)
        #バイアス項を入れない場合はそのまま
        else:
            X_0 = X

        return np.dot(X_0, self.coef_.T)