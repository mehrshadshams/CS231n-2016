import numpy as np
import matplotlib.pyplot as plt


def show_decision_boundary(X, W, b, W2, b2):
    # plot the resulting classifier
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = np.dot(
        np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b), W2) + b2
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()
    # fig.savefig('spiral_net.png')


show_plots = True
N = 100  # number of points per class
D = 2  # dimensionality
K = 3  # number of classes
X = np.zeros((N * K, D))  # data matrix (each row = single example)
y = np.zeros(N * K, dtype='uint8')  # class labels
for j in xrange(K):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)  # radius
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix] = j
# lets visualize the data:
if show_plots:
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()

num_examples = X.shape[0]
num_epochs = 10000

h = 100

W1 = 0.01 * np.random.randn(D, h)
W2 = 0.01 * np.random.randn(h, K)
b1 = np.zeros((1, h))
b2 = np.zeros((1, K))

reg = 1e-3

step_size = 1e-0

for i in range(num_epochs):
    hidden = np.maximum(0, np.dot(X, W1) + b1)  # ReLU
    scores = np.dot(hidden, W2) + b2

    exp_scores = np.exp(scores)
    exp_scores_norm = exp_scores.sum(axis=1, keepdims=True)
    probs = exp_scores / exp_scores_norm
    correct_logprobs = -np.log(probs[range(num_examples), y])

    data_loss = np.sum(correct_logprobs) / num_examples
    reg_loss = 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

    loss = data_loss + reg_loss

    if i % 100 == 0:
        print 'Epoch %d, loss: %.2f' % (i, loss)

    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples

    dW2 = np.dot(hidden.T, dscores)
    dW2 += reg * W2

    db2 = np.sum(dscores, axis=0, keepdims=True)

    dhidden = np.dot(dscores, W2.T)
    dhidden[hidden <= 0] = 0

    dW1 = np.dot(X.T, dhidden)
    dW1 += reg * W1
    db1 = np.sum(dhidden, axis=0, keepdims=True)

    W2 += -step_size * dW2
    b2 += -step_size * db2
    W1 += -step_size * dW1
    b1 += -step_size * db1

# print loss
# expected_loss = -np.log(1.0/K)
# print np.abs(expected_loss - loss)

hidden = np.maximum(0, np.dot(X, W1) + b1)
scores = np.dot(hidden, W2) + b2
predicted_class = np.argmax(scores, axis=1)

print 'training accuracy: %.2f' % np.mean(predicted_class == y)
if show_plots:
    show_decision_boundary(X, W1, b1, W2, b2)
