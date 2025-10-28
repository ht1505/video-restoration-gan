# 🎥 AI-Powered Video Restoration ✨

This project uses a pix2pix-style Generative Adversarial Network (GAN) to **automatically restore** degraded video frames. The model is trained to reverse a combination of severe degradations, including Gaussian blur, random "occlusion" boxes, and Gaussian noise.

---

## 🎯 The Problem vs. The Solution

The model is trained to reverse multiple, severe degradations simultaneously.

| The Problem: Degraded Input | The Solution: Our AI's Output |
| :---: | :---: |
| *(Your Degraded Image Here)* | *(Your Restored Image Here)* |
| **Blurry** (7x7 Gaussian Blur) | ✅ **De-blurred and Sharp** |
| **Noisy** (Gaussian Noise) | ✅ **De-noised and Clean** |
| **Missing Info** (Gray Boxes) | ✅ **"Inpainted" and Re-created** |

---

## 🧠 How it Works: A "Pix2Pix" Showdown

This project is a **Generative Adversarial Network (GAN)**, a "two-player game" between two models: an **Artist** and a **Critic**.

### 1. The Artist 🎨 (The `UNetGenerator`)

This is the model that does the actual work. Its job is to take a `Degraded Image` and paint a `Restored Image`.

* **Architecture:** It's a **U-Net**, an encoder-decoder with "skip connections."
* **Secret Weapon:** The skip connections pipe low-level details (like edges and textures) *directly* from the blurry input to the final output. This is crucial for preventing a smudged mess and ensuring a sharp, detailed image.

### 2. The Critic 🕵️ (The `PatchGAN` Discriminator)

This model's only job is to get *really good* at spotting fakes. It's the "Adversary" that makes the Artist get better.

* **Job:** It looks at a pair of images (`Degraded` + `Restored`) and decides if the `Restored` image is the *real* original or a *fake* one made by the Artist.
* **Secret Weapon:** It's a **`PatchGAN`**. It doesn't output one "real/fake" score. It outputs a **14x14 grid of scores**, one for each "patch" of the image. This forces the Artist to make *every single part* of the image look perfect.

---

## ⚖️ The "Rulebook" (How We Train)

The `Generator` (Artist) is trained to optimize a combined loss function:

1.  **`L1 Loss` (The "Manager"):** This is a pixel-by-pixel accuracy score, comparing the `Restored Image` to the `Original Image`. We make this score **100 times more important** (`lambda_l1 = 100`) to force the model to be **correct**, not just "pretty."

2.  **`GAN Loss` (The "Critic"):** This score measures how well the Artist *fooled* the Critic. This pushes the Artist to make the image look **sharp, realistic, and believable**.

**Total Generator Loss = `(L1_Loss * 100) + GAN_Loss`**

---

## 🚀 Get Started in 3 Steps

### 1. Clone & Install
Clone this repository and install the required libraries.

```bash
# Clone the repo
git clone [https://github.com/your-username/video-restoration-gan.git](https://github.com/your-username/video-restoration-gan.git)

# Enter the directory
cd video-restoration-gan

# Install all dependencies
pip install -r requirements.txt
```

### 2. Download Data
This model was trained on the [UCF-101 Dataset](https://www.crcv.ucf.edu/data/UCF101.php). You'll need to download the videos (`UCF-101.rar`) and unzip them into a folder.

### 3. Let's Go!
You're all set to train or test the model.

---

## 🚂 How to Train Your Own Model

Run `train.py` and point it to your `UCF-101` video folder.

```bash
# Run a quick test-train on the first 100 videos (recommended first)
python train.py --data_dir /path/to/your/UCF-101/UCF-101 --use_first_100

# Run a full-scale train on the ENTIRE dataset for 50 epochs
python train.py --data_dir /path/to/your/UCF-101/UCF-101 --epochs 50 --batch_size 16
```
Trained models (`generator.pth` and `discriminator.pth`) will be saved in the `saved_models/` folder.

---

## 🎬 How to Test (Inference)

Use `inference.py` to restore a new video with your trained model.

```bash
# Basic inference: Just restore a video
python inference.py \
    --degraded_video /path/to/your_test_video.avi \
    --weights_path saved_models/generator.pth \
    --output_dir restored_videos

# Advanced: Restore AND calculate the PSNR quality score against the original
python inference.py \
    --degraded_video /path/to/your_test_video.avi \
    --weights_path saved_models/generator.pth \
    --original_video /path/to/your_clean_original.avi \
    --output_dir restored_videos
```
Your new, restored `.avi` file will appear in the `restored_videos` folder.

---

## 📈 Results & Future Work

* **Quantitative Score:** After 50 epochs on the 100-video subset, this model achieved an average **PSNR of [XX.XX] dB** on test videos!
* **Future Work:**
    * Train on the *full* 13,000+ videos in UCF-101 for even better results.
    * Experiment with more advanced Generator architectures.
    * Try on real-world (not artificially degraded) old videos.
