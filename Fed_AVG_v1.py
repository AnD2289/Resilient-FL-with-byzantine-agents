import numpy as np
import matplotlib.pyplot as plt

# Define the objective function and gradient
def objective_function(x):
    return np.sum((x - optimal_model) ** 2)  # Example quadratic objective

def gradient(x):
    return 2 * (x - optimal_model)  # Gradient of the quadratic objective

def add_noise(gradient, noise_level=0.1):
    return gradient + noise_level * np.random.randn(gradient.shape[0])

# Define the number of clients and local iterations
num_clients = 10
T = 2

# Initialize server and client models
server_model = np.zeros(10)
client_models = [server_model.copy() for _ in range(num_clients)]

# Set the initial global learning rate and learning rate schedule
initial_learning_rate = 0.1
learning_rate = initial_learning_rate

# Store errors for plotting
errors = []

# Define the optimal model (for illustration purposes)
optimal_model = np.random.randn(10)

# Set the noise level for gradients
noise_level = 0.1

# Initialize the convergence counter
convergence_counter = 0

# Perform FedAvg with noisy gradients
for epoch in range(50):  # Perform a fixed number of iterations

    for i in range(num_clients):
        # Simulate local training with gradient descent and noisy gradients
        for t in range(T):
            local_gradient = gradient(client_models[i])
            noisy_local_gradient = add_noise(local_gradient, noise_level)
            client_models[i] -= learning_rate * noisy_local_gradient

    # Aggregate the client models using FedAvg
    server_model = np.mean(client_models, axis=0)

    # Compute the error between the server model and the optimal model
    error = np.linalg.norm(server_model - optimal_model)
    errors.append(error)

    # Update the learning rate schedule
    learning_rate *= 0.95  # Learning rate schedule

    # Check for convergence (error below a threshold)
    if error < 0.1:
        convergence_counter += 1
    else:
        convergence_counter = 0

    if convergence_counter >= 100:
        print(f'Converged after {epoch + 1} iterations.')
        break

    # Display the current training progress
    # print(f'Epoch [{epoch+1}/500]: Error: {error:.4f}')

# Plot the errors
plt.plot(errors)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Error vs. Epoch')
plt.show()

# The server_model now contains the optimized solution
