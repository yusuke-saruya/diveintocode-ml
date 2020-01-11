#ライブラリインポート
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class ScratchDeepNeuralNetrowkClassifier():
    """
    Deepニューラルネットワーク分類器

    Parameters
    ----------
    verbose : bool
        学習過程を出力する場合はTrue
    n_nodes1 : int(default:100)
        1層目のノード数
    n_nodes2 : int(default:30)
        2層目のノード数
    n_epochs : int(default:30)
        イテレーション数
    lr : flaot (default:1e-3)
        学習率
    batch : int(default :10)
        ミニバッチの単位数
    sigma : float(default:0.01)
        ガウス分布の標準偏差
    opt : str (default: 'sgd')
        最適化手法。'sgd'、'adagrad'より選択。
    act : str (default : 'relu')
        活性化関数。'relu', 'sigmoid', 'tanh'より選択。

    Attributes
    ----------
    self.loss : ndarray,shape(n_epochs, )
        エポックごとの誤差を格納
    self.val_loss : ndarray,shape(n_epochs, )
        エポックごとの誤差(検証用データ)を格納
    self.n_output : int
        出力層のノード数（クラス数）
    self.n_features : int
        入力層のノード数（特徴量の数）
        
        
    """

    def __init__(self, verbose = True, n_nodes1=100, n_nodes2=30, 
                 n_epochs=30, lr=1e-3, batch=10, sigma=0.01, opt='sgd', act='relu'):

        self.verbose = verbose             #True(default):学習過程を表示、False:非表示
        self.n_nodes1 = n_nodes1        #1層目のノード数(default:100)
        self.n_nodes2 = n_nodes2        #2層目のノード数(default:30)
        self.n_epochs = n_epochs        #エポック数(default:30)
        self.lr = lr                                #学習率(default:1e-3)
        self.batch = batch                   #ミニバッチを行うサイズ(default:10)
        self.sigma = sigma                  #ガウス分布の標準偏差
        self.opt = opt                          #最適化手法('sgd' or 'adagrad')
        self.act = act                          #活性化関数('relu' or 'sigmoid' or 'tanh')
        self.loss = np.zeros(n_epochs)
        self.val_loss = np.zeros(n_epochs)
    
    
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
        #出力層のノード数
        self.n_output = np.unique(y).shape[0]
        
        #正解データをワンホットエンコーディング
        y = self._onehot(y)
        #検証用データもあればワンホット
        if X_val is not None:
            y_val = self._onehot(y_val)


        #入力層のノード数
        self.n_features = X.shape[1]

        #minibatchデータを生成
        train_minibatch = GetMiniBatch(X, y, batch_size=self.batch)
        

        #最適化手法のインスタンスを選択、生成。
        if self.opt == 'sgd':
            optimizer = SGD(self.lr)
        elif self.opt == 'adagrad':
            optimizer = AdaGrad(self.lr)
        
        #活性化関数の選択
        #'relu'の場合、Heにて初期化
        if self.act == 'relu':
        
            #1層目
            #インスタンス生成、重みの初期化
            self.FC1 = FC(self.n_features, self.n_nodes1, HeInitializer(), optimizer)
            self.activation1 = Relu()        

            #２層目
            #インスタンス生成、重みの初期化
            self.FC2 = FC(self.n_nodes1, self.n_nodes2, HeInitializer(), optimizer)
            self.activation2 = Relu()        

            #3層目(出力層)
            #インスタンス生成、重みの初期化
            self.FC3 = FC(self.n_nodes2, self.n_output, HeInitializer(), optimizer)
            self.activation3 = Softmax()

        #'sigmoid'の場合、Xavierにて初期化
        elif self.act == 'sigmoid':
        
            #1層目
            #インスタンス生成、重みの初期化
            self.FC1 = FC(self.n_features, self.n_nodes1, XavierInitializer(), optimizer)
            self.activation1 = Sigmoid()        

            #２層目
            #インスタンス生成、重みの初期化
            self.FC2 = FC(self.n_nodes1, self.n_nodes2, XavierInitializer(), optimizer)
            self.activation2 = Sigmoid()        

            #3層目(出力層)
            #インスタンス生成、重みの初期化
            self.FC3 = FC(self.n_nodes2, self.n_output, XavierInitializer(), optimizer)
            self.activation3 = Softmax()

        #'tanh'の場合、Xavierにて初期化
        elif self.act == 'tanh':
        
            #1層目
            #インスタンス生成、重みの初期化
            self.FC1 = FC(self.n_features, self.n_nodes1, XavierInitializer(), optimizer)
            self.activation1 = Tanh()        

            #２層目
            #インスタンス生成、重みの初期化
            self.FC2 = FC(self.n_nodes1, self.n_nodes2, XavierInitializer(), optimizer)
            self.activation2 = Tanh()        

            #3層目(出力層)
            #インスタンス生成、重みの初期化
            self.FC3 = FC(self.n_nodes2, self.n_output, XavierInitializer(), optimizer)
            self.activation3 = Softmax()


        #エポック数だけトレーニングを繰り返す
        for epoch in range(self.n_epochs):
                            
            #引数で設定したminibatch数の単位で学習を行う
            for mini_X, mini_y in train_minibatch:
                X = mini_X.copy()
                Y = mini_y.copy()
                                            
                #フォワードプロパゲーション
                A1 = self.FC1.forward(X)
                Z1 = self.activation1.forward(A1)
                A2 = self.FC2.forward(Z1)
                Z2 = self.activation2.forward(A2)
                A3 = self.FC3.forward(Z2)
                Z3 = self.activation3.forward(A3)
                                
                #バックプロパゲーション
                dA3 = self.activation3.backward(Z3, Y) # 交差エントロピー誤差とソフトマックスを合わせている
                dZ2 = self.FC3.backward(dA3)
                dA2 = self.activation2.backward(dZ2)
                dZ1 = self.FC2.backward(dA2)
                dA1 = self.activation1.backward(dZ1)
                dZ0 = self.FC1.backward(dA1) # dZ0は使用しない
                    

            ############
            # 評価
            ############
            #誤差を格納
            self.loss[epoch] = np.mean(self.activation3.cost)
                        
            #検証用データが引数にある場合、処理を行う
            if X_val is not None:
                                
                #フォワードプロパゲーション
                A1_val = self.FC1.forward(X_val)
                Z1_val = self.activation1.forward(A1_val)
                A2_val = self.FC2.forward(Z1_val)
                Z2_val = self.activation2.forward(A2_val)
                A3_val = self.FC3.forward(Z2_val)
                Z3_val = self.activation3.forward(A3_val)
                
                #検証用データの交差エントロピー誤差を計算
                cost_val = self._compute_cost(y_val, Z3_val)

                #誤差を格納
                self.val_loss[epoch] = np.mean(cost_val)
                            

            #verboseをTrueにした際は学習過程などを出力する
            if self.verbose:
                #一度だけ、'Cross Entropy Error'を出力
                if epoch == 0:
                    print('Cross Entropy Error')
                    
                #エポックごとのコスト関数を出力
                print('epoch{} : {}'.format(epoch+1, np.mean(self.activation3.cost)))
                
                #検証用データがある場合、そのコスト関数も出力
                if X_val is not None:
                    print('epoch_val{} : {}'.format(epoch+1, np.mean(cost_val)))

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
        A1 = self.FC1.forward(X)
        Z1 = self.activation1.forward(A1)
        A2 = self.FC2.forward(Z1)
        Z2 = self.activation2.forward(A2)
        A3 = self.FC3.forward(Z2)
        Z3 = self.activation3.forward(A3)
        
        #出力層の確率から、最大値をそのクラスとする
        y_pred = np.argmax(Z3, axis=1)
        
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
    
    
class FC:
    """
    ノード数n_nodes1からn_nodes2への全結合層
    Parameters
    ----------
    n_nodes1 : int
      前の層のノード数
    n_nodes2 : int
      後の層のノード数
    initializer : 初期化方法のインスタンス
    optimizer : 最適化手法のインスタンス
    
    Attribute
    ------------
    self.W : 重み
    self.B : バイアス
    self.H_w : 前のイテレーションまでの勾配の(重み)二乗和(初期値0)
    self.H_b : 前のイテレーションまでの勾配の(バイアス)二乗和(初期値0)
    self.forward_Z : forward時の入力値(backward用に利用)
    self.dW : 重みの勾配
    self.dB : バイアスの勾配
    """
    def __init__(self, n_nodes1, n_nodes2, initializer, optimizer):
        self.optimizer = optimizer
        
        # 初期化
        # initializerのメソッドを使い、self.Wとself.Bを初期化する
        self.W = initializer.W(n_nodes1, n_nodes2)
        self.B = initializer.B(n_nodes2)
        
        #AdaGard用
        self.H_w = 0
        self.H_b = 0
        
        self.forward_Z=None

    def forward(self, Z):
        """
        フォワード
        Parameters
        ----------
        Z : 次の形のndarray, shape (batch_size, n_nodes1)
            入力
        Returns
        ----------
        A : 次の形のndarray, shape (batch_size, n_nodes2)
            出力
        """
        #backfoward用に保存
        self.forward_Z = Z.copy()
        
        A = (Z @ self.W) + self.B
        
        return A
    
    
    def backward(self, dA):
        """
        バックワード
        Parameters
        ----------
        dA : 次の形のndarray, shape (batch_size, n_nodes2)
            後ろから流れてきた勾配
        Returns
        ----------
        dZ : 次の形のndarray, shape (batch_size, n_nodes1)
            前に流す勾配
        """
        self.dB = dA
        self.dW = self.forward_Z.T @ dA
        
        #[batch_size, n_nodes2] dot [n_nodes2, n_nodes1]
        #→ [batch_size, n_nodes1]
        dZ = dA @ self.W.T 
        
        # 更新
        self = self.optimizer.update(self)
              
        return dZ
    
    
    
class SimpleInitializer:
    """
    ガウス分布によるシンプルな初期化
    Parameters
    ----------
    sigma : float
      ガウス分布の標準偏差
    """
    
    def __init__(self, sigma):
        self.sigma = sigma
        
    def W(self, n_nodes1, n_nodes2):
        """
        重みの初期化
        Parameters
        ----------
        n_nodes1 : int
          前の層のノード数
        n_nodes2 : int
          後の層のノード数

        Returns
        ----------
        W :
        """
        W = self.sigma * np.random.randn(n_nodes1, n_nodes2)

        return W
    
    def B(self, n_nodes2):
        """
        バイアスの初期化
        Parameters
        ----------
        n_nodes2 : int
          後の層のノード数

        Returns
        ----------
        B :
        """
        B = np.zeros(n_nodes2)

        return B

    
    
class SGD:
    """
    確率的勾配降下法
    Parameters
    ----------
    lr : 学習率
    """
    
    def __init__(self, lr):
        self.lr = lr
        
    def update(self, layer):
        """
        ある層の重みやバイアスの更新
        Parameters
        ----------
        layer : 更新前の層のインスタンス

        Returns
        ----------
        layer : 更新後の層のインスタンス
        """
        layer.W -= self.lr * layer.dW / layer.dB.shape[0]
        layer.B -= np.mean(self.lr * layer.dB, axis=0)
        
        return layer
    
    
class Sigmoid():
    """
    sigmoidの計算
    
    Parameters
    -----------
    
    Attribute
    -----------
    self.forward_A : forward時の入力値(backward用に利用)
    """
    
    def __init__(self):
        self.forward_A = None
        
    def forward(self, A):
        """
        フォワードプロパゲーションにおける活性化関数の計算
        
        Parameters
        -----------
        A : 活性化関数計算前
        
        Return
        -----------
        Z : 出力

        """
        #backfoward用に保存
        self.forward_A = A.copy()
        
        Z = 1 / (1 + np.exp(-A))
        
        return Z
    
    def backward(self, dZ):
        """
        バックプロパゲーションにおける活性化関数の計算
        
        Parameters
        -----------
        dZ : 活性化関数計算前
        
        Return
        -----------
        dA : 出力

        """
        sigmoid_A = 1 / (1 + np.exp(-self.forward_A))
        
        dA = dZ * ((1 - sigmoid_A)  * sigmoid_A)
        
        return dA
    
    
class Tanh():
    """
    ハイパボリックタンジェントの計算
    
    Parameters
    -----------
    
    Attribute
    -----------
    self.forward_A : forward時の入力値(backward用に利用)
    """
    
    def __init__(self):
        self.forward_A = None
        
    def forward(self, A):
        """
        フォワードプロパゲーションにおける活性化関数の計算
        
        Parameters
        -----------
        A : 活性化関数計算前
        
        Return
        -----------
        Z : 出力

        """
        #backfoward用に保存
        self.forward_A = A.copy()
        
        Z = np.tanh(A)
        
        return Z
    
    def backward(self, dZ):
        """
        バックプロパゲーションにおける活性化関数の計算
        
        Parameters
        -----------
        dZ : 活性化関数計算前
        
        Return
        -----------
        dA : 出力

        """
        dA = dZ * (1 - (np.tanh(self.forward_A) ** 2))
        
        return dA
    
    
class Softmax():
    """
    softmaxの計算
    
    Parameters
    -----------
    
    Attribute
    -----------
    self.cost : 交差エントロピー誤差を格納
    """
    
    def __init__(self):
        self.cost = None
        
    def forward(self, A):
        """
        フォワードにおけるソフトマックスの計算
        
        Parameters
        -----------
        A : 活性化関数計算前
        
        Return
        -----------
        Z : 出力
        """
        #オーバーフロー対策
        max_A = np.max(A)

        #最大要素を引いてからexpをかけることでオーバーフローを回避
        exp_A = np.exp(A - max_A)

         #分母を計算
        sum_exp_A = np.sum(exp_A, axis=1).reshape(-1, 1)
        
        Z = exp_A / sum_exp_A

        return Z
        
    
    def backward(self, Z, Y):
        """
        バックワードにおけるソフトマックスと交差エントロピー誤差
        
        Parameters
        -----------
        Z : 出力層で計算された出力
        Y : 正解値
        
        Return
        -----------
        dA : 出力

        """
        #交差エントロピー誤差
        self.cost = - np.sum(Y * np.log(Z), axis=1)

        #バックワード(出力層)
        dA = Z - Y
        
        return dA
    

    
    
class Relu():
    """
    ReLUの計算
    
    Parameters
    -----------
    
    Attribute
    -----------
    self.mask : 入力値の0以下を判定するboolリスト
    
    """
    
    def __init__(self):
        self.mask = None     
        
    def forward(self, A):
        """
        フォワードにおける活性化関数の計算
        
        Parameters
        -----------
        A : 活性化関数計算前
        
        Return
        -----------
        Z : 出力

        """
        self.mask = (A <= 0)
        
        Z = A.copy()
        
        Z[self.mask] = 0
        
        return Z
    
    def backward(self, dZ):
        """
        バックワードにおける活性化関数の計算
        
        Parameters
        -----------
        dZ : 活性化関数計算前
        
        Return
        -----------
        dA : 出力

        """
        dZ[self.mask] = 0
        
        dA = dZ
            
        return dA
    
    
class XavierInitializer:
    """
    Xavierによる初期化
    
    Parameters
    ----------
    
    """
    
    def __init__(self):
        pass
        
    def W(self, n_nodes1, n_nodes2):
        """
        重みの初期化
        Parameters
        ----------
        n_nodes1 : int
          前の層のノード数
        n_nodes2 : int
          後の層のノード数

        Returns
        ----------
        W :
        """
        W = np.random.randn(n_nodes1, n_nodes2) / np.sqrt(n_nodes1)

        return W
    
    def B(self, n_nodes2):
        """
        バイアスの初期化
        Parameters
        ----------
        n_nodes2 : int
          後の層のノード数

        Returns
        ----------
        B :
        """
        B = np.zeros(n_nodes2)

        return B
    
    
    
class HeInitializer:
    """
    Heによる初期化
    
    Parameters
    ----------
    
    """
    
    def __init__(self):
        pass
        
    def W(self, n_nodes1, n_nodes2):
        """
        重みの初期化
        Parameters
        ----------
        n_nodes1 : int
          前の層のノード数
        n_nodes2 : int
          後の層のノード数

        Returns
        ----------
        W :
        """
        W = np.random.randn(n_nodes1, n_nodes2) / np.sqrt(n_nodes1) * np.sqrt(2)

        return W
    
    def B(self, n_nodes2):
        """
        バイアスの初期化
        Parameters
        ----------
        n_nodes2 : int
          後の層のノード数

        Returns
        ----------
        B :
        """
        B = np.zeros(n_nodes2)

        return B
    
    
class AdaGrad:
    """
    AdaGradによる最適化手法
    Parameters
    ----------
    lr : 学習率
    """    
    def __init__(self, lr):
        self.lr = lr
        
    def update(self, layer):
        """
        ある層の重みやバイアスの更新
        Parameters
        ----------
        layer : 更新前の層のインスタンス

        Returns
        ----------
        layer : 更新後の層のインスタンス
        """
        #重みの更新
        layer.H_w += (layer.dW / layer.dB.shape[0]) ** 2
        layer.W -= self.lr * (1 / np.sqrt(layer.H_w + 1e-7)) * (layer.dW / layer.dB.shape[0])

        #バイアスの更新
        layer.H_b += (layer.dB) ** 2        
        layer.B -= np.mean(self.lr * (1 / np.sqrt(layer.H_b + 1e-7)) * layer.dB)

        
        return layer