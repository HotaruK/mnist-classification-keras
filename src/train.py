from keras import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Flatten

if __name__ == '__main__':
    train, test = mnist.load_data()

    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train[0], train[1], epochs=20)

    test_loss, test_acc = model.evaluate(test[0], test[1])
    print('Test accuracy:', test_acc)

    model.save('../mnist_model.h5')
