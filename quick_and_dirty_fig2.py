import matplotlib.pyplot as plt
import numpy as np

dim = 4
s1, s2, s3 = np.mgrid[:dim, :dim, :dim]

to_remove = (
    (s1 == 0) & (s2 == dim - 1)  # rm the top-left corner of the s2-s1 plane
    | (s1 == dim - 1) & (s2 == 0)  # rm the bot-right corner of the s2-s1 plane
    | (s1 == 0) & (s3 == dim - 1)  # rm the top-left corner of the s3-s1 plane
    | (s1 == dim - 1) & (s3 == 0)  # rm the bot-right corner of the s3-s1 plane
)
s1 = s1[~to_remove]
s2 = s2[~to_remove]
s3 = s3[~to_remove]

true_y = (s1 >= dim / 2).astype(int)

ax = plt.subplot(111, projection="3d")
ax.scatter(s1, s2, s3, c=true_y)
ax.set_xlabel("s1")
ax.set_ylabel("s2")
ax.set_zlabel("s3")
plt.show()

print("=" * 20)
print("corr(s1,s2)", np.corrcoef(s1, s2)[0, 1])
print("corr(s1,s3)", np.corrcoef(s1, s3)[0, 1])
print("corr(s3,s2)", np.corrcoef(s3, s2)[0, 1])

# Stack and zscore
s_data = np.stack([s1.ravel(), s2.ravel(), s3.ravel()], axis=-1)
s_data = (s_data - s_data.mean(axis=0, keepdims=True)) / s_data.std(axis=0, keepdims=True)

# %% Define some models


class SuperSimpleModel(object):
    def __init__(self, weights):
        self.weights = list(map(np.array, weights))

    def __call__(self, x):
        activations = []
        for w in self.weights:
            x = x @ w.T
            activations.append(x)
        return activations


modelA = SuperSimpleModel([[[1, 0, 0], [1, 0, 0]], [[1, 0]]])
modelB = SuperSimpleModel([[[0, 1, 0], [0, 0, 1]], [[0, 1]]])
modelC = SuperSimpleModel([[[0, 1, 0], [0, 0, 1]], [[1, 0]]])
modelAC = SuperSimpleModel(modelA.weights[:1] + modelC.weights[1:])
modelBC = SuperSimpleModel(modelB.weights[:1] + modelC.weights[1:])


def evaluate_task(model, data, labels):
    hiddens = model(data)
    is_correct = (hiddens[-1].ravel() > 0) == (labels == 1)
    return np.mean(is_correct)

print("=" * 20)
print("Model A accuracy:", evaluate_task(modelA, s_data, true_y))
print("Model B accuracy:", evaluate_task(modelB, s_data, true_y))
print("Model C accuracy:", evaluate_task(modelC, s_data, true_y))
print("Model AC accuracy:", evaluate_task(modelAC, s_data, true_y))
print("Model BC accuracy:", evaluate_task(modelBC, s_data, true_y))

# %%



def _center(x):
    return x - np.mean(x, axis=0, keepdims=True)


def _hsic(x, y):
    xc = _center(x)
    yc = _center(y)
    cov_xy = np.einsum("bi,bj->ij", xc, yc) / len(x)
    return np.sum(cov_xy * cov_xy)


def evaluate_cka(model1, model2, data):
    hidden1 = model1(data)[0]
    hidden2 = model2(data)[0]
    cka = _hsic(hidden1, hidden2) / np.sqrt(_hsic(hidden1, hidden1) * _hsic(hidden2, hidden2))
    return cka


print("=" * 20)
print("CKA(A,B)", evaluate_cka(modelA, modelB, s_data))
print("CKA(A,C)", evaluate_cka(modelA, modelC, s_data))
print("CKA(B,C)", evaluate_cka(modelB, modelC, s_data))