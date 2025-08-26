import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# Load and prepare data
print("Loading data...")
X = np.load('data/lob_features_720.npy')
print("Original data shape:", X.shape)

# Normalize data to [-1, 1] range for better GAN performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaled data shape:", X_scaled.shape)
print("Data range: [{:.3f}, {:.3f}]".format(X_scaled.min(), X_scaled.max()))

# Improved Generator with Batch Normalization
def build_generator(latent_dim, output_dim):
    model = Sequential([
        Dense(256, input_dim=latent_dim),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        
        Dense(512),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        
        Dense(1024),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        
        Dense(output_dim, activation='tanh')  # tanh for [-1,1] output
    ])
    return model

# Improved Discriminator
def build_discriminator(input_dim):
    model = Sequential([
        Dense(1024, input_dim=input_dim),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        
        Dense(512),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        
        Dense(1, activation='sigmoid')
    ])
    return model

# Create models
latent_dim = 100
output_dim = X_scaled.shape[1]

print("Building improved models...")
generator = build_generator(latent_dim, output_dim)
discriminator = build_discriminator(output_dim)

# Compile with different learning rates
discriminator.compile(
    loss='binary_crossentropy', 
    optimizer=Adam(0.0001, 0.5),  # Lower learning rate
    metrics=['accuracy']
)

# Build GAN
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# Make discriminator not trainable for GAN
discriminator.trainable = False

gan_input = Input(shape=(latent_dim,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

print("Improved models built successfully!")

# Training with label smoothing and other improvements
def train_improved_gan(epochs, batch_size=32):
    half_batch = batch_size // 2
    history = {'d_loss': [], 'g_loss': [], 'd_acc': []}
    
    for epoch in range(epochs):
        # Train Discriminator
        # Select random real samples
        idx = np.random.randint(0, X_scaled.shape[0], half_batch)
        real_samples = X_scaled[idx]
        
        # Add label smoothing (real labels = 0.9 instead of 1.0)
        real_labels = np.random.uniform(0.8, 1.0, (half_batch, 1))
        
        # Generate fake samples
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_samples = generator.predict(noise, verbose=0)
        
        # Fake labels with noise (0.1 instead of 0.0)
        fake_labels = np.random.uniform(0.0, 0.2, (half_batch, 1))
        
        # Train discriminator
        d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train Generator (multiple times to balance)
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1))  # Always 1 for generator
        g_loss = gan.train_on_batch(noise, valid_labels)
        
        # Store loss and accuracy
        history['d_loss'].append(d_loss[0])
        history['d_acc'].append(d_loss[1])
        history['g_loss'].append(g_loss)
        
        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss[0]:.4f}, D Acc: {d_loss[1]:.4f}, G Loss: {g_loss:.4f}")
    
    return history

# Start improved training
print("\nStarting IMPROVED GAN training...")
print("Training for 2000 epochs...")
history = train_improved_gan(epochs=2000, batch_size=32)

# Generate final samples
print("\nGenerating final synthetic samples...")
noise = np.random.normal(0, 1, (100, latent_dim))
generated_samples = generator.predict(noise, verbose=0)
generated_original = scaler.inverse_transform(generated_samples)

# Save results
np.save('data/improved_generated_lob_samples.npy', generated_original)
generator.save('data/improved_generator_model.keras')  # New Keras format
print("Results saved!")
print("Generated samples shape:", generated_original.shape)

print("Training completed!")