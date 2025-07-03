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
    def __init__(self, S_units = 10*10, A_units=1000, R_units = 2, excitory=500, inhibitory=500, threshold=0.2):
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

    
    #define a random matrix that represents connections from s-unit to r-unit
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

    #define a matrix whose sum of all element is zero, use for gamme system
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
            
            total_active_units = np.sum(active_A_unit_matrix)
            if total_active_units > 0:
                avg = np.float32(total_active_units)/np.float32(active_A_unit_matrix.size)
                increment_decrement_matrix = np.where(active_A_unit_matrix == 1, 0.01, -avg*0.01)
                reinforcement_matrix = np.zeros((self.A_units, self.R_units))
                reinforcement_matrix[:, response] = increment_decrement_matrix.flatten()

                #change the value of value matrix based on change matrix                                    
                self.value_matrix = self.value_matrix + reinforcement_matrix


    def predict(self, input, threshold = 0.4):
         #calculate input to each A_unit and determine the active a unit
        if input.ndim == 1:
            input = input.reshape(1, -1)
        A_unit_input_matrix = np.matmul(input, self.connection_matrix)
        active_A_unit_matrix = np.where(A_unit_input_matrix > self.threshold, 1.0, 0.0)
        
        #response recieved to R_unit and provide output based on threshold value
        response = np.matmul(active_A_unit_matrix, self.value_matrix)
        sum = np.sum(response)
        response = response/np.float32(sum)
        if response.any() > threshold:
            return np.argmax(response)
        
        return None
        

#load the data set in image format and convert it to numpy array.

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

def plot_training_size_vs_accuracy_detailed(X_path, E_path, X_imgs, E_imgs, sizes_to_test=None, test_size=50, random_seed=42):
    """
    Plot training data size vs accuracy for X-class, E-class, and overall accuracy.
    Shows three separate lines on the same graph for detailed analysis.
    """
    if sizes_to_test is None:
        sizes_to_test = [(x)*50 for x in range(5)]
    
    np.random.seed(random_seed)
    
    # Load all data
    X_data = create_dataset(X_path, X_imgs)
    E_data = create_dataset(E_path, E_imgs)
    
    # Prepare test data (use last samples)
    X_test = X_data[-test_size:]
    E_test = E_data[-test_size:]
    
    overall_accuracies = []
    x_class_accuracies = []
    e_class_accuracies = []
    
    print("Training models with different data sizes...")
    print("=" * 60)
    
    for train_size in sizes_to_test:
        print(f"\nTraining with {train_size} samples per class...")
        
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
        
        # Test the model on X-class (class 0)
        x_correct = 0
        for test_sample in X_test:
            predicted_label = model.predict(test_sample)
            if predicted_label == 0:  # Correct prediction for X-class
                x_correct += 1
        x_accuracy = x_correct / len(X_test)
        
        # Test the model on E-class (class 1)
        e_correct = 0
        for test_sample in E_test:
            predicted_label = model.predict(test_sample)
            if predicted_label == 1:  # Correct prediction for E-class
                e_correct += 1
        e_accuracy = e_correct / len(E_test)
        
        # Overall accuracy
        total_correct = x_correct + e_correct
        total_samples = len(X_test) + len(E_test)
        overall_accuracy = total_correct / total_samples
        
        # Store results
        x_class_accuracies.append(x_accuracy)
        e_class_accuracies.append(e_accuracy)
        overall_accuracies.append(overall_accuracy)
        
        print(f"  X-class accuracy: {x_accuracy:.3f}")
        print(f"  E-class accuracy: {e_accuracy:.3f}")
        print(f"  Overall accuracy: {overall_accuracy:.3f}")
    
    # Create the plot with three lines
    plt.figure(figsize=(12, 8))
    
    # Plot all three curves
    plt.plot(sizes_to_test, x_class_accuracies, 'r-o', linewidth=2, markersize=8, label='X-class Accuracy', alpha=0.8)
    plt.plot(sizes_to_test, e_class_accuracies, 'g-s', linewidth=2, markersize=8, label='E-class Accuracy', alpha=0.8)
    plt.plot(sizes_to_test, overall_accuracies, 'b-^', linewidth=2, markersize=8, label='Overall Accuracy', alpha=0.8)
    
    # Customize the plot
    plt.xlabel('Training Data Size (per class)', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Perceptron Learning Curves: Class-Specific vs Overall Accuracy', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=12, loc='lower right')
    
    # Add accuracy values on the plot for overall accuracy
    for i, (size, acc) in enumerate(zip(sizes_to_test, overall_accuracies)):
        plt.annotate(f'{acc:.2f}', (size, acc), textcoords="offset points", 
                    xytext=(0,15), ha='center', fontsize=9, color='blue', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed summary
    print("\n" + "=" * 60)
    print("DETAILED SUMMARY")
    print("=" * 60)
    print(f"{'Training Size':<12} | {'X-class':<8} | {'E-class':<8} | {'Overall':<8}")
    print("-" * 50)
    for size, x_acc, e_acc, overall_acc in zip(sizes_to_test, x_class_accuracies, e_class_accuracies, overall_accuracies):
        print(f"{size:<12d} | {x_acc:<8.3f} | {e_acc:<8.3f} | {overall_acc:<8.3f}")
    
    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    best_overall_idx = np.argmax(overall_accuracies)
    best_x_idx = np.argmax(x_class_accuracies)
    best_e_idx = np.argmax(e_class_accuracies)
    
    print(f"Best overall accuracy: {overall_accuracies[best_overall_idx]:.3f} at {sizes_to_test[best_overall_idx]} samples")
    print(f"Best X-class accuracy: {x_class_accuracies[best_x_idx]:.3f} at {sizes_to_test[best_x_idx]} samples")
    print(f"Best E-class accuracy: {e_class_accuracies[best_e_idx]:.3f} at {sizes_to_test[best_e_idx]} samples")
    
    # Check for class imbalance in learning
    final_x_acc = x_class_accuracies[-1]
    final_e_acc = e_class_accuracies[-1]
    class_diff = abs(final_x_acc - final_e_acc)
    
    if class_diff > 0.1:
        better_class = "X" if final_x_acc > final_e_acc else "E"
        print(f"\nNote: {better_class}-class is learned better (difference: {class_diff:.3f})")
        print("Consider adjusting training data balance or model parameters.")
    else:
        print(f"\nGood balance: Class accuracy difference is only {class_diff:.3f}")
    
    return {
        'sizes': sizes_to_test,
        'x_accuracies': x_class_accuracies,
        'e_accuracies': e_class_accuracies,
        'overall_accuracies': overall_accuracies
    }

# Updated example usage
results = plot_training_size_vs_accuracy_detailed(X_path, E_path, X_imgs, E_imgs)
