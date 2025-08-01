ratings = [
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
]

# Matrix dimensions
num_users = len(ratings)
num_items = len(ratings[0])
k = 2  # number of latent features

# Initialize user and item matrices with small values
def initialize_matrix(rows, cols):
    return [[0.1 for _ in range(cols)] for _ in range(rows)]

user_features = initialize_matrix(num_users, k)
item_features = initialize_matrix(num_items, k)

# Training parameters
learning_rate = 0.01
epochs = 5000
regularization = 0.02

# Matrix Factorization using Stochastic Gradient Descent
for epoch in range(epochs):
    for i in range(num_users):
        for j in range(num_items):
            if ratings[i][j] > 0:
                # Predict rating
                predicted = sum(user_features[i][f] * item_features[j][f] for f in range(k))
                error = ratings[i][j] - predicted

                # Update features
                for f in range(k):
                    user_grad = error * item_features[j][f] - regularization * user_features[i][f]
                    item_grad = error * user_features[i][f] - regularization * item_features[j][f]
                    user_features[i][f] += learning_rate * user_grad
                    item_features[j][f] += learning_rate * item_grad

# Predict ratings
def predict_rating(user, item):
    return sum(user_features[user][f] * item_features[item][f] for f in range(k))

# Generate recommendation matrix
print("=== Predicted Ratings ===\n")
for i in range(num_users):
    row = []
    for j in range(num_items):
        if ratings[i][j] == 0:
            row.append(round(predict_rating(i, j), 2))
        else:
            row.append(ratings[i][j])  # keep original rating
    print(f"User {i+1}: {row}")
