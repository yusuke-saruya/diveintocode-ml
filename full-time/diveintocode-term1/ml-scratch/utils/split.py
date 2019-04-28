import numpy as np

def train_test_split(X, y, train_size=0.8, random_state=0):
    """
    学習用データを分割する。

    Parameters
    ----------
    X : 次の形のndarray, shape (n_samples, n_features)
      学習データ
    y : 次の形のndarray, shape (n_samples, )
      正解値
    train_size : float (0<train_size<1)
      何割をtrainとするか指定

    Returns
    ----------
    X_train : 次の形のndarray, shape (n_samples, n_features)
      学習データ
    X_test : 次の形のndarray, shape (n_samples, n_features)
      検証データ
    y_train : 次の形のndarray, shape (n_samples, )
      学習データの正解値
    y_test : 次の形のndarray, shape (n_samples, )
      検証データの正解値
    """
    #ここにコードを書く
    
    # 発生する乱数を指定する
    np.random.seed(random_state)
    # 順番をランダムに並び替え、その順番に並び替える
    p = np.random.permutation(len(X))
    X_random = X[p]
    y_random = y[p]
    
    #train_sizeに応じてデータを分割する
    X_train = X_random[:int(len(X) * train_size)]
    X_test = X_random[:len(X) - (int(len(X) * train_size))]
    y_train = y_random[:int(len(y) * train_size)]
    y_test = y_random[:len(y) - (int(len(y) * train_size))]

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    train_test_split()