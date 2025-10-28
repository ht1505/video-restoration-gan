# train.py
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import from our other .py files
from models import UNetGenerator, Discriminator
from dataset import UCF101Dataset, transform
from utils import weights_init

def main(args):
    # --- 1. Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 2. Load Data ---
    # (Note: transform is imported from dataset.py)
    dataset = UCF101Dataset(data_dir=args.data_dir, transform=transform, use_first_100=args.use_first_100)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # --- 3. Initialize Models ---
    generator = UNetGenerator().to(device)
    discriminator = Discriminator().to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # --- 4. Define Loss and Optimizers ---
    criterion_gan = nn.BCEWithLogitsLoss() # Your original loss
    criterion_l1 = nn.L1Loss()
    lambda_l1 = 100

    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # --- 5. The Training Loop ---
    print("--- Starting Training ---")
    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        total_g_loss = 0
        total_d_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for degraded_imgs, original_imgs in progress_bar:
            degraded_imgs = degraded_imgs.to(device)
            original_imgs = original_imgs.to(device)

            # Create labels (Discriminator's "Answer Key")
            # We must calculate the output size of the PatchGAN
            # Input 128 -> 64 -> 32 -> 16 -> 15 -> 14
            # So the output grid is 14x14
            batch_size = degraded_imgs.size(0)
            real_labels = torch.ones(batch_size, 1, 14, 14, device=device)
            fake_labels = torch.zeros(batch_size, 1, 14, 14, device=device)

            # --- Train Discriminator ---
            optimizer_d.zero_grad()
            
            # Real Pair loss
            real_output = discriminator(degraded_imgs, original_imgs)
            d_loss_real = criterion_gan(real_output, real_labels)

            # Fake Pair loss
            restored_imgs = generator(degraded_imgs)
            fake_output = discriminator(degraded_imgs, restored_imgs.detach())
            d_loss_fake = criterion_gan(fake_output, fake_labels)

            d_loss = (d_loss_real + d_loss_fake) * 0.5
            d_loss.backward()
            optimizer_d.step()

            # --- Train Generator ---
            optimizer_g.zero_grad()
            
            # 1. GAN Loss (How well it fooled the D)
            fake_output = discriminator(degraded_imgs, restored_imgs)
            g_loss_gan = criterion_gan(fake_output, real_labels)
            
            # 2. L1 Loss (Pixel-perfect accuracy)
            g_loss_l1 = criterion_l1(restored_imgs, original_imgs) * lambda_l1
            
            g_loss = g_loss_gan + g_loss_l1
            g_loss.backward()
            optimizer_g.step()

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            progress_bar.set_postfix(G_loss=g_loss.item(), D_loss=d_loss.item())

        avg_g_loss = total_g_loss / len(dataloader)
        avg_d_loss = total_d_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs} - Avg G_Loss: {avg_g_loss:.4f}, Avg D_Loss: {avg_d_loss:.4f}")

    # --- 6. Save The Trained Model ---
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(generator.state_dict(), os.path.join(args.output_dir, 'generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(args.output_dir, 'discriminator.pth'))
    print(f"Training finished. Models saved to {args.output_dir}")


if __name__ == "__main__":
    # This block allows you to run the script from the command line
    parser = argparse.ArgumentParser(description="Train a pix2pix GAN for video restoration.")
    
    # Add command-line arguments
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the UCF-101 video directory')
    parser.add_argument('--output_dir', type=str, default='saved_models', help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--use_first_100', action='store_true', help='Use only the first 100 videos for quick testing')

    args = parser.parse_args()
    main(args)
