import numpy as np
import matplotlib.pyplot as plt

# Define the dimension of the model or feature vector
d = 10  # Replace with the appropriate dimension

# Generate the optimal model (for illustration purposes)
optimal_model = np.random.randn(d)

def generate_noisy_gradient(x, optimal_model, noise_level=0.1):
    # Generate a gradient with noise
    true_gradient = optimal_model - x
    noisy_gradient = true_gradient + noise_level * np.random.randn(d)
    
    return noisy_gradient

def federated_local_sgd_ce_filter(initial_model, N, T, max_iterations, initial_learning_rate, f, noise_level=0.1, l2_reg=1e-3):
    server_model = initial_model
    errors = []  # List to store the errors at each iteration
    learning_rate = initial_learning_rate
    
    for k in range(max_iterations):
        client_models = []  # List to store client models for aggregation
        
        for i in range(N):
            # Receive the current server model and initialize agent-specific values
            client_model = server_model.copy()
            optimizer = Adam(learning_rate=learning_rate)
            
            for t in range(T):
                # Update the client model using local SGD with noisy gradient
                noisy_gradient = generate_noisy_gradient(client_model, optimal_model, noise_level)
                
                # Apply L2 regularization to control parameter values
                noisy_gradient += l2_reg * client_model
                
                # Update client model using the Adam optimizer
                client_model = optimizer.minimize(client_model, noisy_gradient)
                
            # Append the client's model for aggregation
            client_models.append(client_model)
        
        # Calculate the error between the server model and the optimal model
        error = np.mean([np.linalg.norm(server_model - model) for model in client_models])
        errors.append(error)
        
        # Perform aggregation using a simple weighted average of selected clients
        selected_clients = np.argsort([np.linalg.norm(server_model - model) for model in client_models])[:min(N, N - f)]
        weights = [1.0 / (i + 1) for i in range(len(selected_clients))]
        aggregated_model = np.average([client_models[i] for i in selected_clients], axis=0, weights=weights)
        
        # Update the server's model for the next iteration
        server_model = aggregated_model
        
        # Adjust the learning rate using scheduling
        learning_rate *= 0.95  # You can experiment with different scheduling schemes

    return errors

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 0

    def minimize(self, model, gradient):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        update = -self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        model += update
        return model

# Example usage
initial_model = np.random.randn(d)  # Initialize with random values
N = 30  # Number of clients
T = 5  # Number of local iterations
max_iterations = 250  # Maximum number of iterations
initial_learning_rate = 0.01  # Initial learning rate
f = 12  # Number of clients to eliminate in each iteration

# Set a lower noise level for slower convergence
noise_level = 0.95

errors = federated_local_sgd_ce_filter(initial_model, N, T, max_iterations, initial_learning_rate, f, noise_level)

# Plot the errors
plt.plot(errors)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Error vs. Iteration')
plt.show()
