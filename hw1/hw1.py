import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

c = 2
total_iteration = 400
Epochs = 6


def loss_function(w, x, y):
    result = np.sqrt((1 / c / c) * pow((y - x @ w), 2) + 1) - 1
    result = np.mean(result)
    return result


def calculate_gradient(w, x, y):
    gradient_list = []
    for i in range(4):
        # np.dot(x,w)
        numerator = x[:, i].reshape([-1, 1]) * (np.dot(x, w) - y)
        denominator = c * np.sqrt((np.dot(x, w) - y) ** 2 + c * c)
        gradient = np.mean(numerator / denominator)
        gradient_list.append(gradient)
    return gradient_list


def pre_processing(data_csv):
    # Remove any rows of the data that contain a missing (‘NA’) value.
    data_temp = data_csv.dropna(axis=0, how="any")  # drop all rows that have any NaN values

    # List the indices of the removed data points.
    abc = data_csv.count(axis=1)
    index = []
    for i in range(len(abc)):
        if abc[i] < 7:
            index.append(i)
    print("The row index list of the removed data points is:\n ", index)
    print("-----------------------------------------------------------")

    # # deleting all festures from the dataset apart from: age, nearestMRT, nConvenience and price(as labels)
    data_features = data_temp.drop(["transactiondate", "latitude", "longitude"], axis=1)
    # Set a new index for data and delete the old index,
    data_new = data_features.reset_index(drop=True)

    # Q1. Pre-processing
    # (b) feature normalisation
    data_normal = (data_new - data_new.min()) / (data_new.max() - data_new.min())

    print("The mean values over the dataset above are:\n ", data_normal.mean())
    print("-----------------------------------------------------------")

    # Q2. Train and Test sets, half 50%-50%
    train_data, test_data = np.array_split(data_normal, 2)

    # train_data.iloc[1,:];  train_data.iloc[-1,:]
    # read the first and last rows of training sets
    train_top = train_data.head(1)
    train_tail = train_data.tail(1)

    # test_data.iloc[1,:]; test_data.iloc[-1,:]
    # read the first and last rows of test sets
    test_top = test_data.head(1)
    test_tail = test_data.tail(1)

    print("The first row of training set is:\n ", train_top)
    print()
    print("The last row of training set is:\n ", train_tail)
    print()

    print("The first row of test set is:\n ", test_top)
    print()
    print("The last row of test set is:\n ", test_tail)
    print()
    print("-----------------------------------------------------------")

    return train_data, test_data


def GD_GradientDescent(w, x, y, lr):
    weight_list = w
    Loss = []
    for i in range(total_iteration):
        w0 = w[0] - lr * calculate_gradient(w, x, y)[0]
        w1 = w[1] - lr * calculate_gradient(w, x, y)[1]
        w2 = w[2] - lr * calculate_gradient(w, x, y)[2]
        w3 = w[3] - lr * calculate_gradient(w, x, y)[3]
        w = np.array([w0, w1, w2, w3]).reshape([-1, 1])

        weight_list = np.hstack([weight_list, w])
        loss = loss_function(w, x, y)
        Loss.append(loss)

    return Loss, weight_list


def plot_differ_lr(x, y, w, tip):
    alphas = [10, 5, 2, 1, 0.5, 0.25, 0.1, 0.05, 0.01]
    num = 0

    for lr in alphas:
        if tip == 'GD':
            Loss, weight_list = GD_GradientDescent(w, x, y, lr)
        elif tip == 'SGD':
            Loss, weight_list = SGD_Stochastic_Gradient_Descent(w, x, y, lr)

        Loss = np.array(Loss).reshape([-1, 1])

        if num == 0:
            Loss_list = Loss
            num = 1
        else:
            Loss_list = np.hstack([Loss_list, Loss])

    # plot
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(ax.flat):
        ax.plot(Loss_list[:, i])
        ax.set_title(f"step size: {alphas[i]}")
    plt.tight_layout()  # plot formatting
    plt.show()


# Q6, Stochastic Gradient Descent Implementation,SGD
def SGD_Stochastic_Gradient_Descent(w, x, y, lr):
    weight_list = w
    Loss = []
    data = np.hstack([y, x])
    for i in range(Epochs):
        # permutation,shuffle
        random_data = np.random.permutation(data)
        x_ = random_data[:, 1:]
        y_ = random_data[:, 0]
        for j in range(len(y_)):
            numerator_part = np.dot(x_[j, :], w) - y_[j]
            denominator = c * np.sqrt((np.dot(x_[j, :], w) - y_[j]) ** 2 + c * c)


            w0 = w[0] - lr * x_[j, 0] * numerator_part / denominator
            w1 = w[1] - lr * x_[j, 1] * numerator_part / denominator
            w2 = w[2] - lr * x_[j, 2] * numerator_part / denominator
            w3 = w[3] - lr * x_[j, 3] * numerator_part / denominator

            w = np.array([w0, w1, w2, w3]).reshape([-1, 1])
            weight_list = np.hstack([weight_list, w])
            loss = loss_function(w, x, y)
            Loss.append(loss)

        return Loss, weight_list


# read csv file
data_csv = pd.read_csv("./real_estate.csv", header=0)
# data = pd.read_csv("./real_estate.csv",header =None)

train_data, test_data = pre_processing(data_csv)

train_x = train_data[['age', 'nearestMRT', 'nConvenience']]
train_y = train_data[['price']]
x = np.hstack([np.ones([len(train_x), 1]), train_x.values])
y = train_y.values.reshape([-1, 1])
w = np.array([1, 1, 1, 1]).reshape([-1, 1])

test_x = test_data[['age', 'nearestMRT', 'nConvenience']]
test_y = test_data[['price']]
t_x = np.hstack([np.ones([len(test_x), 1]), test_x.values])
t_y = test_y.values.reshape([-1, 1])
plot_differ_lr(x, y, w, tip='GD')

# q5.c
lr = 0.3
Loss, weights = GD_GradientDescent(w, x, y, lr)
plt.plot(weights.T)
plt.title("The progression of each of the four weights over the iterations")
plt.ylabel('Weights'), plt.xlabel('Iterations'), plt.legend(['w0', 'w1', 'w2', 'w3'])
plt.show()
print("-----------------------------------------------------------")
print("The final weight vector of GD: ", weights[:,-1])
train_loss = loss_function(weights[:,-1].reshape([-1,1]), x,y)
print("Achieved losses on Train sets: ", train_loss)
test_loss = loss_function(weights[:,-1].reshape([-1,1]), t_x,t_y)
print("Achieved losses on Test sets: ", test_loss)
print("-----------------------------------------------------------")

plot_differ_lr(x, y, w, tip='SGD')

lr_2= 0.4
Loss_2, weight_list_2 = SGD_Stochastic_Gradient_Descent(w,x,y,lr_2)
plt.plot(weight_list_2.T)
plt.ylabel('Weights'), plt.xlabel('Iterations'),plt.legend(['w0','w1','w2','w3'])
plt.show()
weight_last = weight_list_2[:,-1]
print("The final weight vector of SGD: ",weight_last)
train_loss = loss_function(weight_last.reshape([-1,1]), x,y)

print("Achieved losses on Train sets of SGD: ", train_loss)
test_loss = loss_function(weight_last.reshape([-1,1]), t_x,t_y)
print("Achieved losses on Test sets of SGD: ", test_loss)

print("-----------------------------------------------------------")
