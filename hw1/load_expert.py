import pickle
from sklearn.model_selection import train_test_split


def load_expert_data(filename):
    """
    input: filename, expert data
    output: x_train, x_test, y_train, y_test
    """
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())

    x = data['observations']
    y = data['actions']
    y = y.reshape(y.shape[0], -1)
    print('load expert data with observations shape', x.shape,
          'actions reshape as:', y.shape)

    return x, y
    

def split_expert_data(x, y, ratio=0.3):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=ratio)

    return x_train, x_test, y_train, y_test