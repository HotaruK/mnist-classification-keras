from keras import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Flatten
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    train, test = mnist.load_data()

    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

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

    labels = list(range(10))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=labels, yticklabels=labels)
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
