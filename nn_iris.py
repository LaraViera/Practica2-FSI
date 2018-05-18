import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]

# ---Codifica los resultados en 1 o 0 ---
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


# --- Obtenemos los datos para procesarlos en la neurona ---
data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data

x_data = data[:, 0:4].astype('f4')  # the samples are the four first rows of data
y_data = one_hot(data[:, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code

# --- Como _data es nuestro array ahora seleccionaremos los datos ---

# -- 70% para Train --
range_x_train = len(x_data) * 0.7  # nº de elementos
range_y_train = len(y_data) * 0.7

x_data_train = x_data[:int(range_x_train), ]  # datos
y_data_train = y_data[:int(range_y_train), ]

# -- 15% para validaciones --
range_x_validate = range_x_train + (len(x_data) * 0.15)  # nº ele
range_y_validate = range_y_train + (len(y_data) * 0.15)

x_data_validate = x_data[int(range_x_train):int(range_x_validate)]
y_data_validate = y_data[int(range_y_train):int(range_y_validate)]

# -- 15% para test --
x_data_test = x_data[int(range_x_validate):]
y_data_test = y_data[int(range_y_validate):]

# print("\nSome samples...")
# for i in range(20):
#    print(x_data[i], " -> ", y_data[i])
# print

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

# --- Entrada ---
# -- Ya que la capa intermedia tiene 5 neuronas, la matriz W1 sera de 4x5 --
W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

# -- Como la capa intermedia tiene 5 neuronas y hay 3 tipos de flores la matriz --
#     sera de 3x5 --
W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print("----------------------")
print("   Start training...  ")
print("----------------------")

# --- Mandamos lotes de 20 en 20
batch_size = 20

# ---Comenzamos el entrenamiento en el bucle
# como trabajamos en Python 3 cambiamos el xrange por range
for epoch in range(100):
    for jj in range(int(len(x_data_train) / batch_size)):
        batch_xs = x_data_train[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data_train[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    # --- Imprimimos el errores y los resultados ---
    error = sess.run(loss, feed_dict={x: x_data_validate, y_: y_data_validate})
    print("Epoch Validacion#: ", epoch, "Error: ", error)
    result = sess.run(y, feed_dict={x: batch_xs})

    for b, r in zip(batch_ys, result):
        print(b, "-->", r)
    print("----------------------------------------------------------------------------------")

print("---------------------")
print("---     Test      ---")
print("---------------------")

# --- Realizamos lo mismo que arriba pero cambiando el lote por el lote test
errorTest = sess.run(loss, feed_dict={x: x_data_test, y_: y_data_test})
print("Error del Test: ", errorTest)

resultTest = sess.run(y, feed_dict={x: x_data_test})
for b, r in zip(y_data_test, resultTest):
    print(b, "-->", r)
