import os
import numpy as np
from keras.models import load_model
from keras.utils import load_img, img_to_array
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def read_handwritten_data():
    x = []
    y = []

    for i in range(10):
        class_folder = os.path.join('../handwritten', str(i))
        for file_name in os.listdir(class_folder):
            origin_img = load_img(os.path.join(class_folder, file_name),
                                  color_mode='grayscale')
            img = img_to_array(origin_img)
            img = img.reshape(28, 28)
            x.append(img)
            y.append(i)

    return (x, y)


if __name__ == '__main__':
    x, y = read_handwritten_data()

    x_test = np.array(x)
    y_test = np.array(y)

    model = load_model('../mnist_model.h5')
    predictions = model.predict(x_test)
    predicted_labels = np.argmax(predictions, axis=1)

    accuracy = np.mean(predicted_labels == y_test)
    print('Test accuracy:', accuracy)

    cm = confusion_matrix(y_test, predicted_labels)
    print('Confusion Matrix:\n', cm)
    labels = list(range(10))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusionmatrix.png')

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, predicted_labels)
    metrics_df = pd.DataFrame({'Precision': precision,
                               'Recall': recall,
                               'F1-Score': f1_score})
    metrics_df.to_csv('metrics.csv', index=False)
