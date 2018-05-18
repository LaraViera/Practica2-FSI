import gzip
import pickle

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
f.close()

x_data_train, y_data_train = train_set
x_data_validate, y_data_validate = valid_set
x_data_test, y_data_test = test_set

y_data_train = one_hot(y_data_train, 10)
y_data_validate = one_hot(y_data_validate, 10)
y_data_test = one_hot(y_data_test, 10)

x = tf.placeholder(tf.float32, [None, 784])  # muestra
y_true = tf.placeholder(tf.float32, [None, 10])  # etiquetas

# TODO: the neural net!!

W1 = tf.Variable(np.float32(np.random.rand(784, 100)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(100)) * 0.1)
y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

# segunda capa
W2 = tf.Variable(np.float32(np.random.rand(100, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)
y2 = tf.nn.softmax(tf.matmul(y1, W2) + b2)

# a partir del elemento mayor de cada fila
y_sup = tf.argmax(y2, 1)
y_true_max = tf.argmax(y_true, 1)

# funcion de coste entropia cruzada y promedio de esta
cost = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y2), reduction_indices=[1]))

# funcion optimizadora
train = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

# inicializamos
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# comprobamos si las etiquetas sup y la true son iguales en un vector
correct_prediction = tf.equal(y_sup, y_true_max)
# para comprobar, pasamos booleamos a fraccion y calculamos la media
precision = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("----------------------")
print("   Start training...  ")
print("----------------------")

batch_size = 20
control = True
epoch = 0

error_prev = 1

epoch_list = []
error_list = []

while control and epoch < 100:
    for jj in range(int(len(x_data_train) / batch_size)):
        batch_xs = x_data_train[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data_train[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_true: batch_ys})

    # --- Imprimimos el errores y los resultados ---
    error = sess.run(cost, feed_dict={x: x_data_validate, y_true: y_data_validate})
    print("Epoch Validacion#: ", epoch, ", Error Previo: ", error_prev, ", Error: ", error)
    result = sess.run(y_sup, feed_dict={x: batch_xs})

    epoch += 1

    # if error < 0.1:
    #    control = False

    if error_prev - error < 0.001:
        control = False
    else:
        error_prev = error

    # --- Mostramos la grafica
    error_list.append(error)
    epoch_list.append(epoch)
    plt.plot(epoch_list, error_list, 'b')

print("----------------------")
print("        TEST          ")
print("----------------------")

plt.xlabel("N de intentos")
plt.ylabel("Error")
plt.show()

test = sess.run(cost, feed_dict={x: x_data_test, y_true: y_data_test})
print("Error del test: ", test)

percent = sess.run(precision, feed_dict={x: x_data_test, y_true: y_data_test}) * 100

print("Porcentaje de acierto:", percent)

# ---------------- Visualizing some element of the MNIST dataset --------------


plt.imshow(x_data_train[57].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()  # Let's see a sample
print(y_data_train[57])
