import numpy as np

# np.save('./data', np.array([1, 2]))

H = np.array([[1, 0, 1], [0, 0, 0], [0, 9, 0]])
# img = np.array([[1, 1, 1]])
np.savez('./data.npz', H=H)
print(np.load('./data.npz')['H'])
print("--------------------------")
img = np.array([[1, 1, 1]])
np.savez('./data.npz', img=img)


