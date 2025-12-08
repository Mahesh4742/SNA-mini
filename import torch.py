from tensorflow.keras import layers, models, optimizers
import numpy as np


import matplotlib.pyplot as plt
def create_data():
    X = np.random.randn(1000, 10) 
    y = np.random.randn(1000, 1)  
    return X, y
def create_model():
    model = models.Sequential([
        layers.Dense(50, activation='relu', input_shape=(10,)),  
        layers.Dense(20, activation='relu'),                     
        layers.Dense(1)                                          
    ])
    return model
def train_model_with_history(model, optimizer, X, y, batch_size, epochs, optimizer_name):
    model.compile(optimizer=optimizer, loss='mean_squared_error') 
    history = [] 
    for epoch in range(epochs):
        hist = model.fit(X, y, batch_size=batch_size, epochs=1, verbose=0) 
        loss = hist.history['loss'][0]  
        history.append(loss) 
        print(f"Epoch {epoch + 1}/{epochs} - {optimizer_name} Loss: {loss:.4f}")
    return history
X, y = create_data()
model_sgd = create_model()
model_adam = create_model()
optimizer_sgd = optimizers.SGD(learning_rate=0.01) 
optimizer_adam = optimizers.Adam(learning_rate=0.001) 
epochs = 50
batch_size = 32
print("\nTraining with SGD Optimizer:")
sgd_loss = train_model_with_history(model_sgd, optimizer_sgd, X, y, batch_size, epochs, 'SGD')
print("\nTraining with Adam Optimizer:")
adam_loss = train_model_with_history(model_adam, optimizer_adam, X, y, batch_size, epochs, 'Adam')
plt.plot(range(1, epochs + 1), sgd_loss, label='SGD', color='blue')
plt.plot(range(1, epochs + 1), adam_loss, label='Adam', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('SGD vs Adam Optimizer: Loss Comparison')
plt.legend()
plt.grid(True)
plt.show()
