import json
import matplotlib.pyplot as plt
import numpy as np


def holt_winters_second_order_ewma(x, span, beta, skip=1):
    N = x.size
    alpha = 2.0 / (1 + span)
    s = np.zeros((N,))
    b = np.zeros((N,))
    s[:skip] = x[0:skip]
    for i in range(skip, N):
        s[i] = alpha * x[i] + (1 - alpha) * (s[i - 1] + b[i - 1])
        b[i] = beta * (s[i] - s[i - 1]) + (1 - beta) * b[i - 1]
    return s


with open('./model/full36_B32_60E_exp_nr/history.json') as f:
    history = json.load(f)

# summarize history for accuracy
plt.figure(0)
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.plot(holt_winters_second_order_ewma(np.array(history['val_acc']), 10, 0.1, 8))
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss
plt.figure(1)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.plot(holt_winters_second_order_ewma(np.array(history['val_loss']), 10, 0.1, 8))
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')


f, axarr = plt.subplots(2, sharex=True)
axarr[0].set_title('Accuracy')
axarr[0].plot(history['acc'])
axarr[0].plot(history['val_acc'])
axarr[0].plot(holt_winters_second_order_ewma(np.array(history['val_acc']), 10, 0.1, 8))
axarr[0].legend(['train', 'test'], loc='upper left')

axarr[1].set_title('Loss')
axarr[1].plot(history['loss'])
axarr[1].plot(history['val_loss'])
axarr[1].plot(holt_winters_second_order_ewma(np.array(history['val_loss']), 10, 0.1, 8))
axarr[1].legend(['train', 'test'], loc='upper left')
plt.show()