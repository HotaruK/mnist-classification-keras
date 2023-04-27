from keras import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Flatten
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def get_filtered_mnist_data():
    """
    Obtain MNIST dataset from 1 to 3
    x is input image data, y is label
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train_filtered = x_train[np.isin(y_train, [1, 2, 3])]
    y_train_filtered = y_train[np.isin(y_train, [1, 2, 3])]

    x_test_filtered = x_test[np.isin(y_test, [1, 2, 3])]
    y_test_filtered = y_test[np.isin(y_test, [1, 2, 3])]

    return (x_train_filtered, y_train_filtered - 1), (x_test_filtered, y_test_filtered - 1)


if __name__ == '__main__':
    train, test = get_filtered_mnist_data()

    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train[0], train[1], epochs=5)

    test_loss, test_acc = model.evaluate(test[0], test[1])
    print('Test accuracy:', test_acc)

    # output result
    y_pred = model.predict(test[0])
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(test[1], y_pred_classes)
    print('Confusion matrix:')
    print(cm)

    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=[1, 2, 3], yticklabels=[1, 2, 3])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('result.png')

    # output layer
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        print(f'Layer {i}:')
        for j, w in enumerate(weights):
            print(f'  Weights {j}:')
            print(w)
