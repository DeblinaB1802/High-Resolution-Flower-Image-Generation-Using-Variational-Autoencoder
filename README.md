# 🌸 Variational Autoencoder (VAE) for High-Resolution Flower Image Generation

This repository presents a deep learning project that utilizes a **Variational Autoencoder (VAE)** to generate high-resolution flower images. The model is trained on the **Oxford Flowers102** dataset, a widely used benchmark for fine-grained image recognition and generation tasks.

---

## 📌 Project Objective

The core aim of this project is to explore **generative modeling** using VAE on visually rich and high-resolution data. The model learns to encode images into a latent space and then decode them back, enabling us to generate new, realistic-looking flower images by sampling from this latent space.

---

## 📁 Project Structure

```
High-Resolution-Flower-Image-Generation-Using-Variational-Autoencoder
├── theory.md # VAE theory 
├── VAE.ipynb
├── requirements.txt # Dependencies
└── README.md # Project documentation
```

## 🌼 Dataset: Oxford Flowers102

The **Oxford Flowers102** dataset consists of:

- **8,189 images** of flowers.
- **102 categories**, each corresponding to a different flower species.
- Images with varying resolutions (typically between **250×250 to 500×500**).
- A challenging dataset due to the **fine-grained** distinctions between classes.

The flowers display a rich diversity in color, shape, and texture, making this dataset ideal for training and evaluating generative models like VAEs.

---

## 🧠 Model Architecture: Variational Autoencoder (VAE)

A **Variational Autoencoder** is a probabilistic generative model that maps input data into a **latent distribution**, instead of deterministic points. Key components of the architecture include:

### 1. **Encoder**
- Learns to map high-dimensional input images to a lower-dimensional latent space.
- Outputs the **mean (μ)** and **log-variance (log(σ²))** of a Gaussian distribution.

### 2. **Latent Space Sampling**
- Uses the **reparameterization trick** to sample latent vectors:  
  `z = μ + σ * ε`, where ε ~ N(0, I)

### 3. **Decoder**
- Transforms sampled latent vectors back into image space.
- Learns to reconstruct original images with minimal loss.

### 4. **Loss Function**
- **Reconstruction Loss**: Measures pixel-wise difference (e.g., Binary Cross Entropy).
- **KL Divergence Loss**: Regularizes the latent space to follow a unit Gaussian distribution.

Total loss = `Reconstruction Loss + β * KL Divergence`

---

## 🔍 Training Details

- **Input Image Size**: 256x256
- **Latent Dimension**: 256
- **Optimizer**: Adam
- **Learning Rate**: 0.0001
- **Batch Size**: 64
- **Epochs**: 300
- **Framework**: PyTorch with TorchVision

Extensive use of **image transforms** (resize, normalization) was employed to standardize input for efficient learning.

---

## 🎨 Results & Visualization

The trained VAE demonstrates an impressive ability to reconstruct and generate high-quality flower images. Key achievements:

- Clear retention of floral shapes and colors.
- Smooth interpolation in latent space—images transition naturally between classes.
- Latent space sampling yields visually plausible, high-resolution flower images.

### ✨ Sample Outputs:

- **Reconstructed Images**: The model successfully recreates original flower images from their latent representations.
- **Generated Images**: Sampling from the latent space produces new flower images not seen during training.

---

## 📈 Insights & Learnings

- VAE can generate **diverse and realistic images** even from a fine-grained dataset like Flowers102.
- **Latent space visualizations** highlight how the model organizes different flower types in a continuous space.
- Using a **larger latent dimension** improves detail preservation but may reduce generalization.
- Higher resolution inputs require **careful architecture tuning** to balance model capacity and overfitting.

---

## 🚀 Future Work

- Integrate **Conditional VAE (CVAE)** to control the output category.
- Improve decoder sharpness using **VAE-GAN hybrids**.
- Apply dimensionality reduction (e.g., PCA, t-SNE) for latent space exploration.
- Extend to multimodal tasks (e.g., image-to-text using captioning on Flowers102).

---

## 🤝 Acknowledgements

- **University of Oxford – VGG** for the Flowers102 dataset.
- TorchVision for dataset loading support.
- PyTorch community for tools, tutorials, and contributions.

---

## 📚 References

- "An Introduction to Variational Autoencoders" by Diederik P. Kingma, Max Welling(2019): [https://arxiv.org/abs/1906.02691](https://arxiv.org/abs/1906.02691)
- Oxford Flowers102 Dataset: [http://www.robots.ox.ac.uk/~vgg/data/flowers/102](http://www.robots.ox.ac.uk/~vgg/data/flowers/102)

---
