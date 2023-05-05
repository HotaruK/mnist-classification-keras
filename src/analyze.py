import csv
from keras.models import load_model

if __name__ == '__main__':
    model = load_model('../mnist_model.h5')

    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        for j, w in enumerate(weights):
            with open(f'../analysis/layer_{i}_weights_{j}.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                if len(w.shape) == 1:
                    writer.writerow(w)
                else:
                    writer.writerows(w)
