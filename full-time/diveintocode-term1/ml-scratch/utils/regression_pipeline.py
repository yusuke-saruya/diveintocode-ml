from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

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


