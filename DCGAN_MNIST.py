# ---------------------------------------
#               Imports
# ---------------------------------------
import os
import uuid
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
import torchvision.utils as vutils
from tqdm import tqdm

# ---------------------------------------
#        Configuration & Setup
# ---------------------------------------

# Create a unique identifier for this training run. This helps in organizing and tracking experiments.
UNIQUE_RUN_ID = str(uuid.uuid4())
print(f"Preparing Training Run {UNIQUE_RUN_ID}")

# Define directory paths for this specific run. All logs and generated images will be saved here.
run_dir = f"runs/{UNIQUE_RUN_ID}"
logs_dir = f"{run_dir}/logs"
images_dir = f"{run_dir}/generated_images"

# Create the directories if they don't already exist.
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

# ---------------------------------------
#              Data Loading
# ---------------------------------------

# Define the transformation pipeline for the input images.
# 1. transforms.ToTensor(): Converts images from PIL format to PyTorch Tensors.
# 2. transforms.Normalize((0.5,), (0.5,)): Normalizes the tensor values to a range of [-1, 1].
#    This is a common practice for training GANs as it helps with model stability.
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Load the MNIST training dataset.
# - "MNIST_data/": The directory where the data will be stored.
# - download=True: Downloads the dataset if it's not already in the specified directory.
# - train=True: Specifies that we are loading the training set.
# - transform=transform: Applies the defined transformation pipeline to each image.
trainset = datasets.MNIST("MNIST_data/", download=True, train=True, transform=transform)

# Create a DataLoader for the training set.
# - trainset: The dataset to load.
# - batch_size=128: The number of images to process in each batch.
# - shuffle=True: Randomly shuffles the data at the beginning of each epoch to prevent the model from learning the order of the data.
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

# Note: We are not using the test set in this GAN implementation, as the goal is to generate images, not to evaluate classification performance.
testset = datasets.MNIST("MNIST_data/", download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)

# ---------------------------------------
#           Model Configuration
# ---------------------------------------

# Set the device for training. It will use the GPU (cuda) if available, otherwise it will fall back to the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weights_init(m):
    """
    Custom weight initialization function.
    As recommended in the DCGAN paper, weights are initialized from a normal distribution
    with a mean of 0 and a standard deviation of 0.02. This helps prevent issues like
    vanishing or exploding gradients during the initial phases of training.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# ---------------------------------------
#          Discriminator Model
# ---------------------------------------

def disc_conv(in_c, out_c, ks=4, stride=2, padding=1, bn=True, out_layer=False):
    """
    Helper function to create a convolutional block for the Discriminator.
    Each block consists of a Conv2d layer, an optional BatchNorm2d layer, and an activation function.
    """
    layers = [nn.Conv2d(in_c, out_c, kernel_size=ks, stride=stride, padding=padding, bias=False)]
    if bn:
        layers.append(nn.BatchNorm2d(out_c))
    # The final layer uses a Sigmoid activation to output a probability, while intermediate layers use LeakyReLU.
    layers.append(nn.Sigmoid() if out_layer else nn.LeakyReLU(0.2))
    return nn.Sequential(*layers)

# The Discriminator is a fully convolutional network that takes an image and outputs a single value
# representing the probability that the image is real (as opposed to fake).
D = nn.Sequential(
    disc_conv(1, 32, bn=False),      # Input: 1x28x28 -> Output: 32x14x14
    disc_conv(32, 64),               # Input: 32x14x14 -> Output: 64x7x7
    disc_conv(64, 128, ks=3),        # Input: 64x7x7 -> Output: 128x3x3
    disc_conv(128, 1, out_layer=True, bn=False, padding=0) # Input: 128x3x3 -> Output: 1x1x1
)

# ---------------------------------------
#            Generator Model
# ---------------------------------------

class Generator(nn.Module):
    """
    The Generator model takes a random noise vector as input and generates an image.
    It uses a series of transposed convolutional layers to upsample the noise into a full-sized image.
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(*[
            self.conv_block(100, 128, padding=0),
            self.conv_block(128, 64, stride=2, ks=3),
            self.conv_block(64, 32, stride=2),
            self.conv_block(32, 1, stride=2, bn=False, out_layer=True)
        ])

    @staticmethod
    def conv_block(in_c, out_c, out_layer=False, ks=4, stride=1, padding=1, bias=False, bn=True):
        """
        Helper function to create a transposed convolutional block for the Generator.
        """
        l = [nn.ConvTranspose2d(in_c, out_c, ks, stride=stride, padding=padding, bias=bias)]
        if bn:
            l.append(nn.BatchNorm2d(out_c))
        # The final layer uses a Tanh activation to scale the output to [-1, 1], matching the normalized input images.
        # Intermediate layers use ReLU.
        if out_layer:
            l.append(nn.Tanh())
        else:
            l.append(nn.ReLU())
        return nn.Sequential(*l)

    def forward(self, x):
        return self.layers(x)

# ---------------------------------------
#        Initialization & Training
# ---------------------------------------

# Create an instance of the Generator.
G = Generator()

# Create a fixed noise vector. This is used to generate a consistent set of sample images
# throughout training, making it easier to visualize the Generator's progress.
fixed_noise = torch.randn((16, 100, 1, 1), device=device)

# Apply the custom weight initialization to both the Generator and Discriminator.
G.apply(weights_init)
D.apply(weights_init)

# Define the loss function. Binary Cross-Entropy (BCE) is used, which is suitable for
# the binary classification task of the Discriminator (real vs. fake).
criterion = nn.BCELoss()

# Setup the Adam optimizers for both the Generator and Discriminator.
# The learning rate (lr) and beta values are commonly used for GANs.
optim_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optim_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Move the models and loss function to the selected device (GPU or CPU).
D = D.to(device)
G = G.to(device)
criterion = criterion.to(device)

# --- Training Hyperparameters ---
EPOCHS = 1
FAKE_LABEL = 0.0
REAL_LABEL = 1.0

# Open a log file to record the training progress.
log_file = open(f"{logs_dir}/training.log", "w")

# --- Main Training Loop ---
for epoch in range(1, EPOCHS + 1):
  loss_d = 0.0
  loss_g = 0.0
  
  # Create a progress bar for the current epoch.
  pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch}/{EPOCHS}")
  for i, (images, labels) in pbar:
    # Move the batch of real images to the device.
    images = images.to(device)
    
    # Create labels for fake and real images.
    fake_labels = torch.full((images.size(0), ), FAKE_LABEL, device=device)
    real_labels = torch.full((images.size(0), ), REAL_LABEL, device=device)

    # -------------------------
    #  (1) Train Discriminator
    # -------------------------
    D.zero_grad()

    # --- Train with real images ---
    d_real = D(images).view(-1)
    d_loss_real = criterion(d_real, real_labels)
    d_loss_real.backward()

    # --- Train with fake images ---
    # Generate a batch of fake images using the Generator.
    noise = torch.randn((images.size(0), 100, 1, 1), device=device)
    fake_images = G(noise)

    # We use .detach() here because we don't want to backpropagate through the Generator while training the Discriminator.
    d_fake = D(fake_images.detach()).view(-1)
    d_loss_fake = criterion(d_fake, fake_labels)
    d_loss_fake.backward()

    # The total discriminator loss is the sum of the real and fake losses.
    d_loss = d_loss_real + d_loss_fake
    # Update the Discriminator's weights.
    optim_D.step()

    # ---------------------
    #  (2) Train Generator
    # ---------------------
    G.zero_grad()

    # Get the Discriminator's prediction on the fake images.
    d_fake = D(fake_images).view(-1)
    # The Generator's goal is to make the Discriminator classify its fake images as real.
    # So, we calculate the loss using the `real_labels`.
    g_loss = criterion(d_fake, real_labels)
    g_loss.backward()

    # Update the Generator's weights.
    optim_G.step()

    # ---------------------
    #  (3) Logging & Saving
    # ---------------------
    if i % 100 == 0:
      log_message = f"Epoch [{epoch}/{EPOCHS}], Batch [{i}], LOSS_D: {d_loss.item():.4f}, LOSS_G: {g_loss.item():.4f}"
      pbar.set_postfix_str(log_message)
      log_file.write(log_message + "\n")
      
      # Save a grid of sample images generated from the fixed noise vector.
      with torch.no_grad():
        sample_images = G(fixed_noise).cpu().detach()
        vutils.save_image(sample_images, f"{images_dir}/epoch_{epoch}batch_{i}.png", normalize=True, nrow=4)

# Close the log file and indicate that training is complete.
log_file.close()
print("Finished Training")

# ---------------------------------------
#          Final Image Generation
# ---------------------------------------
# Generate and save a final grid of images to visualize the Generator's output.
fig = plt.figure(figsize=(4, 4))
plt.axis("off")
out = vutils.make_grid(G(fixed_noise).cpu().detach(), padding=5, normalize=True, nrow=4)
plt.imshow(np.transpose(out.numpy(), (1, 2, 0)), cmap="gray")
plt.savefig(f"{images_dir}/final_output.png")
plt.show()
