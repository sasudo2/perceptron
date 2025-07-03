import numpy as np
from PIL import Image
import os

np.seterr(all='ignore') 
original_dtype = np.float32

X_path = "./X/"
E_path = "./E/"
X_imgs = os.listdir(X_path)
E_imgs = os.listdir(E_path)

class perceptron:
    def __init__(self, S_units = 10*10, A_units=1000, R_units = 2, excitory=700, inhibitory=300, threshold=0.2):
        if excitory + inhibitory > A_units:
            raise ValueError("connections exceeded the number of s-unit")
        self.A_units = A_units
        self.S_units = S_units
        self.R_units = R_units
        self.excitory = excitory
        self.inhibitory = inhibitory
        self.threshold = np.float32(threshold)

        self.connection_matrix = self.__create_connection_matrix__()
        self.value_matrix = self.__create_zero_sum_matrix__()

    
    def __create_connection_matrix__(self):
        rows = self.S_units
        cols = self.A_units
        excitory_connections = np.ones((rows, self.excitory))
        no_connections = np.zeros((rows, cols - self.excitory - self.inhibitory))
        inhibitory_connections = np.full([rows, self.inhibitory], -1.0)
        base_matrix = np.concatenate([excitory_connections, no_connections, inhibitory_connections], axis = 1)

        for i in range(rows):
            np.random.shuffle(base_matrix[i])

        return base_matrix

    def __create_zero_sum_matrix__(self):
        rows = self.A_units
        cols = self.R_units
        random_matrix = np.random.normal(0, 0.1, (rows, cols)).astype(np.float32)

        sum = np.sum(random_matrix)
        total_elements = rows*cols
        avg = np.float32(sum)/np.float32(total_elements)

        zero_sum_matrix = random_matrix - avg
        return zero_sum_matrix
    
    def train(self, inputs, response):
        for input in inputs:
            #calculate input to each A_unit and determine the active a unit
            if input.ndim == 1:
                input = input.reshape(1, -1)
            
       
            A_unit_input_matrix = np.matmul(input, self.connection_matrix)
            active_A_unit_matrix = np.where(A_unit_input_matrix > self.threshold, 1.0, 0.0)

            output_response = np.matmul(active_A_unit_matrix, self.value_matrix)
            predicted_class = np.argmax(output_response)

            if predicted_class != response:
            #create a change matrix for gamma system
                total_active_units = np.sum(active_A_unit_matrix)
                if total_active_units > 0:
                    avg = np.float32(total_active_units)/np.float32(active_A_unit_matrix.size)
                    increment_decrement_matrix = np.where(active_A_unit_matrix == 1, 0.01, -avg*0.01)
                    reinforcement_matrix = np.zeros((self.A_units, self.R_units))
                    reinforcement_matrix[:, response] = increment_decrement_matrix.flatten()

                    #change the value of value matrix based on change matrix                                    
                    self.value_matrix = self.value_matrix + reinforcement_matrix
                    
    def predict(self, input):
         #calculate input to each A_unit and determine the active a unit
        if input.ndim == 1:
            input = input.reshape(1, -1)
        A_unit_input_matrix = np.matmul(input, self.connection_matrix)
        active_A_unit_matrix = np.where(A_unit_input_matrix > self.threshold, 1.0, 0.0)
        
        #response recieved to R_unit
        response = np.matmul(active_A_unit_matrix, self.value_matrix)
        return np.argmax(response)



def create_dataset(path, image_list):
        dataset = []
        for img in image_list:
            image = Image.open(path + img)
            
            image = image.convert('L')
            image = image.resize((10,10), Image.Resampling.LANCZOS)
            image_array = np.array(image)/255.0
            dataset.append(image_array.flatten())
        
        return np.array(dataset)

X_data = create_dataset(X_path, X_imgs)
E_data = create_dataset(E_path, E_imgs)

import matplotlib.pyplot as plt

def plot_training_size_vs_accuracy(X_path, E_path, X_imgs, E_imgs, sizes_to_test=None, test_size=50, random_seed=42):

    if sizes_to_test is None:
        sizes_to_test =  [(2**x)*10 for x in range(9)]
    
    np.random.seed(random_seed)
    
    # Load all data
    X_data = create_dataset(X_path, X_imgs)
    E_data = create_dataset(E_path, E_imgs)
    
    # Prepare test data (use last samples)
    X_test = X_data[-test_size:]
    E_test = E_data[-test_size:]
    X_test_labels = [0] * len(X_test)
    E_test_labels = [1] * len(E_test)
    
    test_data = np.concatenate([X_test, E_test])
    test_labels = X_test_labels + E_test_labels
    
    accuracies = []
    
    print("Training models with different data sizes...")
    
    for train_size in sizes_to_test:
        print(f"Training with {train_size} samples per class...")
        
        # Create model
        model = perceptron()
        
        # Get training data (exclude test samples)
        X_train = X_data[:-test_size][:train_size]
        E_train = E_data[:-test_size][:train_size]
        
        # Create labels
        X_labels = [0] * len(X_train)
        E_labels = [1] * len(E_train)
        
        # Combine and shuffle training data
        all_train_data = np.concatenate([X_train, E_train])
        all_train_labels = X_labels + E_labels
        
        indices = np.random.permutation(len(all_train_data))
        all_train_data = all_train_data[indices]
        all_train_labels = [all_train_labels[i] for i in indices]
        
        # Train the model
        for data, label in zip(all_train_data, all_train_labels):
            model.train([data], label)
        
        # Test the model
        correct_predictions = 0
        for test_sample, true_label in zip(test_data, test_labels):
            predicted_label = model.predict(test_sample)
            if predicted_label == true_label:
                correct_predictions += 1
        
        accuracy = correct_predictions / len(test_data)
        accuracies.append(accuracy)
        print(f"  Accuracy: {accuracy:.3f}")
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(sizes_to_test, accuracies, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Training Data Size (per class)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Perceptron Accuracy vs Training Data Size', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Add accuracy values on the plot
    for i, (size, acc) in enumerate(zip(sizes_to_test, accuracies)):
        plt.annotate(f'{acc:.3f}', (size, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\nSummary:")
    print("Training Size | Accuracy")
    print("-" * 22)
    for size, acc in zip(sizes_to_test, accuracies):
        print(f"{size:12d} | {acc:.3f}")
    
    return sizes_to_test, accuracies

# Example usage:
sizes, accuracies = plot_training_size_vs_accuracy(X_path, E_path, X_imgs, E_imgs)
