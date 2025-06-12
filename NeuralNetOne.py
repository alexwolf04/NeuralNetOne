import numpy as np
import matplotlib.pyplot as plt
import pickle
import tkinter as tk
from tkinter import messagebox


input_size = 2
hidden1_size = 6
hidden2_size = 4
output_size = 1
epochs = 10000
lr = 0.1
model_file = "neuralnet_weights.pkl"


def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x): return x * (1 - x)


X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])


def init_weights():
    return {
        'W1': np.random.uniform(-1,1,(input_size, hidden1_size)),
        'b1': np.zeros((1, hidden1_size)),
        'W2': np.random.uniform(-1,1,(hidden1_size, hidden2_size)),
        'b2': np.zeros((1, hidden2_size)),
        'W3': np.random.uniform(-1,1,(hidden2_size, output_size)),
        'b3': np.zeros((1, output_size))
    }


def forward_pass(x, weights):
    z1 = x.dot(weights['W1']) + weights['b1']
    a1 = sigmoid(z1)
    z2 = a1.dot(weights['W2']) + weights['b2']
    a2 = sigmoid(z2)
    z3 = a2.dot(weights['W3']) + weights['b3']
    a3 = sigmoid(z3)
    return a1, a2, a3


def train(weights):
    losses = []
    for epoch in range(epochs):
         
        a1, a2, output = forward_pass(X, weights)
        error = y - output
        loss = np.mean(np.square(error))
        losses.append(loss)

        
        d_output = error * sigmoid_deriv(output)
        d_a2 = d_output.dot(weights['W3'].T) * sigmoid_deriv(a2)
        d_a1 = d_a2.dot(weights['W2'].T) * sigmoid_deriv(a1)

        
        weights['W3'] += a2.T.dot(d_output) * lr
        weights['b3'] += np.sum(d_output, axis=0, keepdims=True) * lr
        weights['W2'] += a1.T.dot(d_a2) * lr
        weights['b2'] += np.sum(d_a2, axis=0, keepdims=True) * lr
        weights['W1'] += X.T.dot(d_a1) * lr
        weights['b1'] += np.sum(d_a1, axis=0, keepdims=True) * lr

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.5f}")
    return losses


def save_weights(weights):
    with open(model_file, 'wb') as f:
        pickle.dump(weights, f)

def load_weights():
    try:
        with open(model_file, 'rb') as f:
            return pickle.load(f)
    except:
        return init_weights()


class NeuralNetGUI:
    def __init__(self, master):
        self.master = master
        master.title("CyberSecureNet - XOR Neural Network")
        master.configure(bg="#1e1e2f")
        self.weights = load_weights()

        self.label = tk.Label(master, text="Enter XOR Input (0/1,0/1):", fg="#fff", bg="#1e1e2f", font=("Consolas", 12))
        self.label.pack(pady=10)

        self.entry = tk.Entry(master, width=10, font=("Consolas", 12))
        self.entry.pack()

        self.predict_btn = tk.Button(master, text="🔮 Predict", command=self.predict, bg="#282a36", fg="white", font=("Consolas", 12))
        self.predict_btn.pack(pady=10)

        self.train_btn = tk.Button(master, text="🧠 Retrain", command=self.retrain, bg="#6272a4", fg="white", font=("Consolas", 12))
        self.train_btn.pack(pady=5)

        self.graph_btn = tk.Button(master, text="📈 Show Loss Graph", command=self.plot_loss, bg="#50fa7b", fg="black", font=("Consolas", 12))
        self.graph_btn.pack(pady=5)

    def predict(self):
        try:
            vals = list(map(int, self.entry.get().split(',')))
            x_input = np.array([vals])
            _, _, output = forward_pass(x_input, self.weights)
            result = output[0][0]
            messagebox.showinfo("Prediction", f"Predicted Output: {round(result, 3)}")
        except:
            messagebox.showerror("Error", "Invalid input. Use format: 0,1")

    def retrain(self):
        self.weights = init_weights()
        losses = train(self.weights)
        save_weights(self.weights)
        messagebox.showinfo("Training", "Training complete.")
        self.losses = losses

    def plot_loss(self):
        if not hasattr(self, 'losses'):
            self.retrain()
        plt.plot(self.losses)
        plt.title("Training Loss Over Time")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    weights = load_weights()
    losses = train(weights)
    save_weights(weights)

    print("\nFinal Output After Training:")
    _, _, final = forward_pass(X, weights)
    print(np.round(final, 3))

    root = tk.Tk()
    app = NeuralNetGUI(root)
    root.mainloop()
