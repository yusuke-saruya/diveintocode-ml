#ライブラリインポート
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class ScratchSimpleNeuralNetrowkClassifier():
    """
    シンプルな三層ニューラルネットワーク分類器

    Parameters
    ----------
    verbose : bool
        学習過程を出力する場合はTrue
    n_hidden : int
        隠れ層のノード数
    l2 : float
        正則化係数
    epochs : int
        エポック数（イテレーション数）
    eta : float
        学習率
    minibatch_size : int
        ミニバッチサイズ

    Attributes
    ----------
    self.b_h : ndarray,shape(n_hidden, )
        隠れ層のバイアス
    self.w_h : ndarray,shape(n_features, n_hidden)
        隠れ層の重み
    self.b_out : ndarray,shape(n_output, )
        出力層のバイアス
    self.w_out : ndarray,shape(n_hidden, n_output)
        出力層の重み
    self.evals_ : dict
        イテレーションごとのcostとaccuracy
    """

    def __init__(self, verbose = True, n_hidden1=100, n_hidden2=30, epochs=100, eta=0.001, minibatch_size=10, seed=None):
        self.verbose = verbose                                        #True(default):学習過程を表示、False:非表示
        self.random=np.random.RandomState(seed)  #初期化する
        self.n_hidden1 = n_hidden1                                   #1層目のノード数(default:100)
        self.n_hidden2 = n_hidden2                                   #2層目のノード数(default:30)
        self.epochs = epochs                                          #エポック数(default:100)
        self.eta = eta                                                       #学習率(default:0.001)
        self.minibatch_size = minibatch_size              #ミニバッチを行うサイズ(default:10)
    
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        ニューラルネットワーク分類器を学習する。

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
        """
        #クラスラベルの個数
        n_output = np.unique(y).shape[0]
        
        #入力層の特徴量
        n_features = X.shape[1]
                
        #1層目の重みを初期化
        self.b_1 = np.zeros(self.n_hidden1)
        self.w_1 = self.random.normal(loc=0.0, scale=0.1, size=(n_features, self.n_hidden1))

        #2層目の重みを初期化
        self.b_2 = np.zeros(self.n_hidden2)
        self.w_2 = self.random.normal(loc=0.0, scale=0.1, size=(self.n_hidden1, self.n_hidden2))

        #3層目(出力層)の重みを初期化
        self.b_3 = np.zeros(n_output)
        self.w_3 = self.random.normal(loc=0.0, scale=0.1, size=(self.n_hidden2, n_output))
        
        #学習過程データの格納用dictionary
        self.evals_ = {'cost' : [], 'cost_val' : [], 'train_acc' : [], 'valid_acc' : []}
        
        #正解データをワンホットエンコーディング
        y_enc = self._onehot(y)
        
        #エポック数だけトレーニングを繰り返す
        for i in range(self.epochs):
            
            #minibatchデータを生成
            get_mini_batch = GetMiniBatch(X, y_enc, batch_size=self.minibatch_size)
            
            #引数で設定したminibatch数の単位で学習を行う
            for mini_X_train, mini_y_train in get_mini_batch:
                                            
                #フォワードプロパゲーション
                #a:活性化関数に代入するもの、z:活性化関数の計算結果
                z_1, a_1, z_2, a_2, z_3, a_3 = self._forward(mini_X_train)
                
                #バックプロパゲーション                
                delta_w_1, delta_b_1, delta_w_2, delta_b_2, delta_w_3, delta_b_3 = self._back(
                    mini_X_train, mini_y_train, z_1, a_1, z_2, a_2, z_3, a_3)
                
                #１層目の重みの更新
                self.w_1 -= self.eta * delta_w_1
                self.b_1 -= self.eta * delta_b_1
                
                #2層目の重みの更新
                self.w_2 -= self.eta * delta_w_2
                self.b_2 -= self.eta * delta_b_2

                #3層目の重みの更新
                self.w_3 -= self.eta * delta_w_3
                self.b_3 -= self.eta * delta_b_3
                
                
            ############
            # 評価
            ############
            
            #イテレーションごとに評価を行う
            z_1, a_1, z_2, a_2, z_3, a_3 = self._forward(X)
            
            #交差エントロピー誤差を計算
            cost = self._compute_cost(y=y_enc, y_pred=z_3)

            #誤差を格納
            self.evals_['cost'].append(cost)
            
            #推定を行い、accuracyを計算する
            y_pred = self.predict(X)
            train_acc = \
                ((np.sum(y == y_pred)).astype(np.float) / X.shape[0])
            self.evals_['train_acc'].append(train_acc)
            
            #検証用データが引数にある場合、処理を行う
            if X_val is not None:
                
                #y_valのワンホットエンコーディング
                y_val_enc = self._onehot(y_val)
                
                #イテレーションごとに評価を行う
                z_1_val, a_1_val, z_2_val, a_2_val, z_3_val, a_3_val = self._forward(X_val)
                
                #検証用データの交差エントロピー誤差を計算
                cost_val = self._compute_cost(y=y_val_enc, y_pred=z_3_val)

                #誤差を格納
                self.evals_['cost_val'].append(cost_val)
                
                #推定を行い、accuracyを計算する
                y_val_pred = self.predict(X_val)
                valid_acc = \
                    ((np.sum(y_val == y_val_pred)).astype(np.float) / X_val.shape[0])
                self.evals_['valid_acc'].append(valid_acc)
            
            

            #verboseをTrueにした際は学習過程などを出力する
            if self.verbose:
                #一度だけ、'Cross Entropy Error'を出力
                if i == 0:
                    print('Cross Entropy Error')
                    
                #エポックごとのコスト関数を出力
                print('epoch{} : {}'.format(i+1, np.mean(cost)))
                
                #検証用データがある場合、そのコスト関数も出力
                if X_val is not None:
                    print('epoch_val{} : {}'.format(i+1, np.mean(cost_val)))

        return self


    def predict(self, X):
        """
        ニューラルネットワーク分類器を使い推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
        y_pred :  次の形のndarray, shape (n_samples, 1)
            推定結果
        """
        #フォワードプロパゲーション
        z_1, a_1, z_2, a_2, z_3, a_3 = self._forward(X)
        
        #出力層の確率から、最大値をそのクラスとする
        y_pred = np.argmax(z_3, axis=1)
        
        return y_pred
    
    def _onehot(self, y):
        """
        多クラス分類を行う際のone-hot表現に変換

        Parameters
        ----------
        y : 次の形のndarray, shape (n_samples, )
            サンプル

        Returns
        -------
        y_one_hot : 次の形のndarray, shape (n_samples, n_classes)
            推定結果
        """
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        y_one_hot = enc.fit_transform(y[:, np.newaxis])
        
        return y_one_hot
    
    
    def _sigmoid(self, z):
        """
        活性化関数sigmoidを計算する

        Parameters
        ----------
        z : 次の形のndarray, shape ((batch_size, n_nodes)
            サンプル

        Returns
        -------

        """
        
        return 1 / (1 + np.exp(-z))
    
    def _softmax(self, x):
        """
        ソフトマックスを計算する

        Parameters
        ----------
        x : 次の形のndarray, shape (batch_size, n_output)
            サンプル

        Returns
        -------
         次の形のndarray, shape (batch_size, n_output)
            ソフトマックス計算結果
        """
        #オーバーフロー対策
        max_x = np.max(x)

        #最大要素を引いてからexpをかけることでオーバーフローを回避
        exp_x = np.exp(x - max_x)

         #和を計算
        sum_exp_x = np.sum(exp_x, axis=1).reshape(-1, 1)

        return exp_x / sum_exp_x

    
    def _forward(self, X):
        """
        フォワードプロパゲーションの計算

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
         次の形のndarray, shape (batch_size, n_nodes)
            sigmoid計算結果
        """
        
        #1層目
        #[n_samples, n_features] dot [n_features, n_hidden1]
        #→ [n_samples, n_hiddne1]
        a_1 = (X @ self.w_1) + self.b_1
        
        #1層目の活性化関数
        z_1 = self._sigmoid(a_1)

        #2層目
        #[n_samples, n_features] dot [n_features, n_hidden1]
        #→ [n_samples, n_hiddne1]
        a_2 = (z_1 @ self.w_2) + self.b_2
        
        #2層目の活性化関数
        z_2 = self._sigmoid(a_2)

        #3層目(出力層)
        #[n_samples, n_hiddne] dot [n_hidden, n_classlabels]
        #→ [n_samples, n_classlabls]
        a_3 = (z_2 @ self.w_3) + self.b_3
        
        #3層目出力層の活性化関数
        z_3 = self._softmax(a_3)
        
        return z_1, a_1, z_2, a_2, z_3, a_3
    
    def _back(self, X, y, z_1, a_1, z_2, a_2, z_3, a_3):
        """
        バックプロパゲーションの計算

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
         次の形のndarray, shape (batch_size, n_nodes)
            sigmoid計算結果
        """
        #3層目
        #[n_samples, n_output]
        delta_a_3 = z_3 - y

        #[n_hidden, n_samples] dot [n_samples, n_output]
        #→ [n_hiddne, n_output]
        delta_w_3 = z_2.T @ delta_a_3

        #[n_output,]
        delta_b_3 = np.mean(delta_a_3, axis=0)

        #2層目
        #[n_samples, n_classlabels] dot [n_classlabels, n_hidden]
        #→ [n_samples, n_hidden]
        delta_a_2 = ((delta_a_3 @ self.w_3.T) * ((1 - self._sigmoid(a_2)) * self._sigmoid(a_2)))

        #[n_features, n_samples] dot [n_samples, n_hidden]
        #→ [n_features, n_hidden]
        delta_w_2 = z_1.T @ delta_a_2

        #[n_hidden, ]
        delta_b_2 = np.mean(delta_a_2, axis=0)

        #1層目
        #[n_samples, n_classlabels] dot [n_classlabels, n_hidden]
        #→ [n_samples, n_hidden]
        delta_a_1 = ((delta_a_2 @ self.w_2.T) * ((1 - self._sigmoid(a_1)) * self._sigmoid(a_1)))

        #[n_features, n_samples] dot [n_samples, n_hidden]
        #→ [n_features, n_hidden]
        delta_w_1 = X.T @ delta_a_1

        #[n_hidden, ]
        delta_b_1 = np.mean(delta_a_1, axis=0)
        
        return delta_w_1, delta_b_1, delta_w_2, delta_b_2, delta_w_3, delta_b_3

    
    
    #交差エントロピー誤差
    def _compute_cost(self, y, y_pred):

        return - np.sum(y * np.log(y_pred), axis=1)
    
    
    
class GetMiniBatch:
    """
    ミニバッチを取得するイテレータ

    Parameters
    ----------
    X : 次の形のndarray, shape (n_samples, n_features)
      学習データ
    y : 次の形のndarray, shape (n_samples, 1)
      正解値
    batch_size : int
      バッチサイズ
    seed : int
      NumPyの乱数のシード
    """
    def __init__(self, X, y, batch_size = 10, seed=0):
        self.batch_size = batch_size
        np.random.seed(seed)
        shuffle_index = np.random.permutation(np.arange(X.shape[0]))
        self.X = X[shuffle_index]
        self.y = y[shuffle_index]
        self._stop = np.ceil(X.shape[0]/self.batch_size).astype(np.int)

    def __len__(self):
        return self._stop

    def __getitem__(self,item):
        p0 = item*self.batch_size
        p1 = item*self.batch_size + self.batch_size
        return self.X[p0:p1], self.y[p0:p1]        

    def __iter__(self):
        self._counter = 0
        return self

    def __next__(self):
        if self._counter >= self._stop:
            raise StopIteration()
        p0 = self._counter*self.batch_size
        p1 = self._counter*self.batch_size + self.batch_size
        self._counter += 1
        return self.X[p0:p1], self.y[p0:p1]