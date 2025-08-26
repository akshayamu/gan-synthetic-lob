import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load processed data
print("Loading data...")
X = np.load('data/lob_features_720.npy')
print("Original data shape:", X.shape)

# Normalize data (important for GANs)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaled data shape:", X_scaled.shape)
print("Data range: [{:.3f}, {:.3f}]".format(X_scaled.min(), X_scaled.max()))

# Build Generator
def build_generator(latent_dim, output_dim):
    model = Sequential([
        Dense(128, input_dim=latent_dim),
        LeakyReLU(alpha=0.2),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dense(512),
        LeakyReLU(alpha=0.2),
        Dense(output_dim, activation='tanh')
    ])
    return model

# Build Discriminator
def build_discriminator(input_dim):
    model = Sequential([
        Dense(512, input_dim=input_dim),
        LeakyReLU(alpha=0.2),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dense(128),
        LeakyReLU(alpha=0.2),
        Dense(1, activation='sigmoid')
    ])
    return model

# Create models
latent_dim = 100
output_dim = X_scaled.shape[1]

print("Building models...")
generator = build_generator(latent_dim, output_dim)
discriminator = build_discriminator(output_dim)

# Compile discriminator
discriminator.compile(
    loss='binary_crossentropy', 
    optimizer=Adam(0.0002, 0.5), 
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

print("Models built successfully!")
print("Generator summary:")
generator.summary()

# Training function
def train_gan(epochs, batch_size=32):
    half_batch = batch_size // 2
    history = {'d_loss': [], 'g_loss': [], 'd_acc': []}
    
    for epoch in range(epochs):
        # Train Discriminator
        # Select random real samples
        idx = np.random.randint(0, X_scaled.shape[0], half_batch)
        real_samples = X_scaled[idx]
        
        # Generate fake samples
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_samples = generator.predict(noise, verbose=0)
        
        # Labels for real and fake samples
        real_labels = np.ones((half_batch, 1))
        fake_labels = np.zeros((half_batch, 1))
        
        # Train discriminator
        d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_labels)
        
        # Store loss and accuracy
        history['d_loss'].append(d_loss[0])
        history['d_acc'].append(d_loss[1])
        history['g_loss'].append(g_loss)
        
        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss[0]:.4f}, D Acc: {d_loss[1]:.4f}, G Loss: {g_loss:.4f}")
            
        # Save generated samples every 500 epochs
        if epoch % 500 == 0 and epoch > 0:
            generate_and_save_samples(epoch)
    
    return history

# Function to generate and save samples
def generate_and_save_samples(epoch):
    noise = np.random.normal(0, 1, (5, latent_dim))
    generated = generator.predict(noise, verbose=0)
    generated_original = scaler.inverse_transform(generated)
    
    print(f"\nGenerated samples at epoch {epoch}:")
    print("Shape:", generated_original.shape)
    print("Sample features (first 10):", generated_original[0, :10])

# Start training
print("\nStarting GAN training...")
print("Training for 1000 epochs...")
history = train_gan(epochs=1000, batch_size=32)

# Generate final samples
print("\nGenerating final synthetic samples...")
noise = np.random.normal(0, 1, (100, latent_dim))
generated_samples = generator.predict(noise, verbose=0)
generated_original = scaler.inverse_transform(generated_samples)

# Save results
np.save('data/generated_lob_samples.npy', generated_original)
generator.save('data/generator_model.h5')
print("Results saved!")
print("Generated samples shape:", generated_original.shape)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['d_loss'], label='Discriminator Loss')
plt.plot(history['g_loss'], label='Generator Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['d_acc'])
plt.title('Discriminator Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.savefig('data/training_history.png')
plt.show()

print("Training completed!")