from matplotlib import pyplot as plt
from data import get_data
from ARMA2_mine import *

datas = get_data()
# model = MARMA(datas)
# model = MAR(datas)
# model = MARIMA(datas)
model = MBayesian(datas)
predictions = model.get_predictions()

balances = []
balance = 1
close = datas['testx']['Close']
for i in range(0, len(close) - 1):
    if close[i] < predictions[i]:  # 做多
        balance = balance / close[i] * close[i + 1]
        balances.append(balance)
    else:
        balance = 2 * balance - balance / close[i] * close[i + 1]
        balances.append(balance)
plt.plot(balances)
plt.show()
