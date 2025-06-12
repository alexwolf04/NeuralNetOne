# NeuralNetOne - XOR Neural Network

A Python implementation of a multi-layer neural network that learns the XOR logic gate using backpropagation. Features a sleek GUI interface for training, prediction, and visualization.

## Features

- **Multi-layer Neural Network**: 3-layer architecture (2-6-4-1) with sigmoid activation
- **XOR Logic Learning**: Trained to solve the classic XOR problem
- **Interactive GUI**: User-friendly interface with cyber-themed design
- **Model Persistence**: Automatic saving/loading of trained weights
- **Loss Visualization**: Real-time training loss plotting
- **Backpropagation**: Complete implementation from scratch using NumPy

## Architecture

```
Input Layer (2 nodes) → Hidden Layer 1 (6 nodes) → Hidden Layer 2 (4 nodes) → Output Layer (1 node)
```

- **Activation Function**: Sigmoid
- **Learning Rate**: 0.1
- **Training Epochs**: 10,000
- **Loss Function**: Mean Squared Error

## Requirements

```
numpy
matplotlib
tkinter (usually included with Python)
pickle (standard library)
```

## Installation

1. Install required dependencies:
   ```bash
   pip install numpy matplotlib
   ```
2. Run the application:
   ```bash
   python neural_network.py
   ```

## Usage

### GUI Interface

The application launches with a dark-themed GUI featuring three main functions:

- **🔮 Predict**: Enter XOR inputs (format: `0,1`) to get predictions
- **🧠 Retrain**: Reinitialize and retrain the network from scratch
- **📈 Show Loss Graph**: Display training loss curve over epochs

### Input Format

Enter XOR inputs as comma-separated values:
- `0,0` → Expected output: ~0
- `0,1` → Expected output: ~1
- `1,0` → Expected output: ~1
- `1,1` → Expected output: ~0

### Training Data

The network is trained on the complete XOR truth table:

| Input A | Input B | Output |
|---------|---------|--------|
|    0    |    0    |   0    |
|    0    |    1    |   1    |
|    1    |    0    |   1    |
|    1    |    1    |   0    |

## Technical Details

### Network Architecture
- **Input Layer**: 2 neurons (for binary inputs A and B)
- **Hidden Layer 1**: 6 neurons with sigmoid activation
- **Hidden Layer 2**: 4 neurons with sigmoid activation  
- **Output Layer**: 1 neuron with sigmoid activation

### Training Process
1. Forward propagation through all layers
2. Error calculation using MSE loss
3. Backward propagation to compute gradients
4. Weight updates using gradient descent
5. Progress tracking every 1000 epochs

### Model Persistence
- Trained weights automatically saved to `neuralnet_weights.pkl`
- Model loads previous weights on startup if available
- Manual retraining reinitializes all weights randomly

## File Structure

```
project/
├── neural_network.py    # Main application file
├── neuralnet_weights.pkl # Saved model weights (generated)
└── README.md           # This file
```

## Example Output

After training, the network should produce outputs close to:
```
Input [0,0] → Output: ~0.000
Input [0,1] → Output: ~1.000  
Input [1,0] → Output: ~1.000
Input [1,1] → Output: ~0.000
```

## Key Functions

- `sigmoid()` & `sigmoid_deriv()`: Activation function and derivative
- `init_weights()`: Random weight initialization
- `forward_pass()`: Forward propagation through network
- `train()`: Complete training loop with backpropagation
- `save_weights()` & `load_weights()`: Model persistence

## Customization

You can easily modify the network by adjusting these parameters:

```python
input_size = 2      # Input layer size
hidden1_size = 6    # First hidden layer size  
hidden2_size = 4    # Second hidden layer size
output_size = 1     # Output layer size
epochs = 10000      # Training iterations
lr = 0.1           # Learning rate
```


## License

This project is open source and available under the MIT License.
