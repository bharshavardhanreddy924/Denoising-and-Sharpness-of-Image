import cv2
import numpy as np
import torch
import torch.nn as nn
import os
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from pathlib import Path

# DnCNN Model Definition
class DnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        
        # First convolutional layer
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, 
                                kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        # Middle convolutional layers with BatchNorm and ReLU
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, 
                                    kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        
        # Final convolutional layer
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, 
                                kernel_size=kernel_size, padding=padding, bias=False))
        
        # Create sequential model
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        # Predict noise and subtract from input
        noise = self.dncnn(x)
        return x - noise

class ISPPipeline:
    """Basic Image Signal Processing Pipeline"""
    
    def __init__(self):
        self.console = Console()
        
    def read_raw(self, filepath, width=1920, height=1280):
        """
        Read 12-bit RAW image in GRBG Bayer pattern
        Args:
            filepath: Path to RAW file
            width: Image width (default 1920)
            height: Image height (default 1280)
        Returns:
            numpy array of shape (height, width) with 16-bit values
        """
        try:
            with open(filepath, 'rb') as f:
                raw_data = np.fromfile(f, dtype=np.uint16)
                
            if len(raw_data) != width * height:
                raise ValueError(f"Expected {width*height} pixels, got {len(raw_data)}")
                
            # Reshape and mask to 12 bits
            raw_image = raw_data.reshape((height, width))
            raw_image = raw_image & 0x0FFF
            raw_image = raw_image.astype(np.uint16)
            
            # Scale to full 16-bit range
            raw_image = (raw_image << 4).astype(np.uint16)
            return raw_image
            
        except Exception as e:
            self.console.print(f"[red]Error loading RAW image: {e}[/red]")
            return None
            
    def demosaic(self, bayer_image):
        """
        Edge-aware demosaicing for GRBG Bayer pattern
        """
        try:
            return cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_GR2RGB)
        except Exception as e:
            self.console.print(f"[red]Demosaicing error: {e}[/red]")
            return None
            
    def white_balance(self, image):
        """
        Apply gray world white balance
        """
        try:
            result = np.zeros_like(image)
            for i in range(3):
                avg = np.mean(image[:, :, i])
                result[:, :, i] = np.clip(image[:, :, i] * (128 / avg), 0, 65535)
            return result
        except Exception as e:
            self.console.print(f"[red]White balance error: {e}[/red]")
            return None
            
    def apply_gamma(self, image, gamma=2.2):
        """
        Apply gamma correction and convert to 8-bit
        """
        try:
            normalized = image.astype(np.float32) / 65535.0
            gamma_corrected = np.power(normalized, 1/gamma)
            return (gamma_corrected * 255).astype(np.uint8)
        except Exception as e:
            self.console.print(f"[red]Gamma correction error: {e}[/red]")
            return None

class DenoiseSharpenPipeline:
    """Advanced denoising and sharpening pipeline"""
    
    def __init__(self):
        self.console = Console()
        self.isp = ISPPipeline()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dncnn = self._load_dncnn()
        
    def _load_dncnn(self):
        """Load pretrained DnCNN model"""
        try:
            # Check if weights file exists
            if not os.path.exists('sidd_denoiser.pth'):
                self.console.print("[red]DnCNN weights file not found![/red]")
                return None

            # Ensure correct device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.console.print(f"[yellow]Using device: {self.device}[/yellow]")

            # Load model
            model = DnCNN(channels=3, num_of_layers=17)
            
            try:
                # Load the state dict
                state_dict = torch.load('sidd_denoiser.pth', map_location=self.device)
                
                # Create a new state dict with the correct prefix
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_key = k.replace('denoiser', 'dncnn')
                    new_state_dict[new_key] = v
                
                # Load the modified state dict
                model.load_state_dict(new_state_dict)
            except Exception as load_error:
                self.console.print(f"[red]Error loading weights: {load_error}[/red]")
                return None

            model = model.to(self.device)
            model.eval()
            self.console.print("[green]Loaded pretrained DnCNN weights[/green]")
            return model

        except Exception as e:
            self.console.print(f"[red]Fatal error loading DnCNN: {e}[/red]")
            return None
        
    def apply_denoise_methods(self, image):
        """Apply various denoising methods"""
        results = {}
        results['original'] = image.copy()
        
        try:
            # Traditional methods
            results['gaussian'] = cv2.GaussianBlur(image, (5, 5), 1.0)
            results['median'] = cv2.medianBlur(image, 5)
            results['bilateral'] = cv2.bilateralFilter(image, 9, 75, 75)
            
            # DnCNN denoising
            if self.dncnn is not None:
                # Preprocess image for DnCNN
                img_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    denoised_tensor = self.dncnn(img_tensor)
                
                # Convert back to numpy
                denoised = denoised_tensor.squeeze().cpu().numpy().transpose((1, 2, 0))
                results['dncnn'] = (np.clip(denoised, 0, 1) * 255).astype(np.uint8)
            
        except Exception as e:
            self.console.print(f"[red]Error in denoising: {e}[/red]")
            
        return results
        
    def apply_sharpen_methods(self, image):
        """Apply various sharpening methods"""
        results = {}
        
        try:
            # Unsharp mask
            gaussian = cv2.GaussianBlur(image, (5, 5), 1.0)
            results['unsharp_mask'] = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
            
            # Laplacian
            kernel = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ])
            results['laplacian'] = cv2.filter2D(image, -1, kernel)
            
        except Exception as e:
            self.console.print(f"[red]Error in sharpening: {e}[/red]")
            
        return results
        
    def compute_metrics(self, image, roi):
        """Compute SNR and edge strength metrics"""
        try:
            x, y, w, h = roi
            roi_img = image[y:y+h, x:x+w]
            
            # Convert to grayscale if needed
            if len(roi_img.shape) == 3:
                roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)
            else:
                roi_gray = roi_img
                
            # Compute SNR
            signal = np.mean(roi_gray)
            noise = np.std(roi_gray)
            snr = signal / noise if noise > 0 else float('inf')
            
            # Compute edge strength using Sobel operator
            sobelx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_strength = np.mean(np.sqrt(sobelx**2 + sobely**2))
            
            return snr, edge_strength
            
        except Exception as e:
            self.console.print(f"[red]Error computing metrics: {e}[/red]")
            return 0, 0
            
    def process_image(self, input_path, output_dir):
        """Main processing pipeline"""
        with Progress() as progress:
            task = progress.add_task("Processing image...", total=100)
            
            try:
                # Create output directory
                os.makedirs(output_dir, exist_ok=True)
                
                # Load and preprocess image
                raw_image = self.isp.read_raw(input_path)
                if raw_image is None:
                    return
                    
                progress.update(task, advance=20)
                
                # Apply ISP pipeline
                demosaiced = self.isp.demosaic(raw_image)
                wb_image = self.isp.white_balance(demosaiced)
                base_image = self.isp.apply_gamma(wb_image)
                
                if base_image is None:
                    return
                    
                progress.update(task, advance=20)
                
                # Define ROIs
                rois = [
                    (200, 200, 400, 400),    # Dark region
                    (800, 600, 400, 400),    # Mid-tone region
                    (1400, 800, 400, 400)    # Bright region
                ]
                
                # Apply denoising and sharpening
                denoised_results = self.apply_denoise_methods(base_image)
                progress.update(task, advance=20)
                
                sharpened_results = self.apply_sharpen_methods(base_image)
                progress.update(task, advance=20)
                
                # Compute metrics
                metrics = {}
                for name, img in {**denoised_results, **sharpened_results}.items():
                    metrics[name] = []
                    for i, roi in enumerate(rois):
                        snr, edge_strength = self.compute_metrics(img, roi)
                        metrics[name].append({
                            'region': 'dark' if i == 0 else 'mid' if i == 1 else 'bright',
                            'snr': snr,
                            'edge_strength': edge_strength
                        })
                        
                progress.update(task, advance=10)
                
                # Save results
                self._save_results(denoised_results, sharpened_results, metrics, output_dir, rois)
                progress.update(task, advance=10)
                
                # Display results
                self._display_results(metrics)
                
                return denoised_results, sharpened_results, metrics
                
            except Exception as e:
                self.console.print(f"[red]Fatal error: {e}[/red]")
                return None
                
    def _save_results(self, denoised_results, sharpened_results, metrics, output_dir, rois):
        """Save processed images and metrics"""
        try:
            # Save images
            for name, img in {**denoised_results, **sharpened_results}.items():
                # Save original image
                cv2.imwrite(
                    os.path.join(output_dir, f"{name}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                )
                
                # Save image with ROIs marked
                img_with_roi = img.copy()
                for i, (x, y, w, h) in enumerate(rois):
                    color = [(0,255,0), (255,0,0), (0,0,255)][i]
                    cv2.rectangle(img_with_roi, (x,y), (x+w,y+h), color, 2)
                
                cv2.imwrite(
                    os.path.join(output_dir, f"{name}_roi.png"),
                    cv2.cvtColor(img_with_roi, cv2.COLOR_RGB2BGR)
                )
            
            # Save metrics to CSV
            with open(os.path.join(output_dir, 'metrics.csv'), 'w') as f:
                f.write('Method,Region,SNR,Edge_Strength\n')
                for method, regions in metrics.items():
                    for region in regions:
                        f.write(f"{method},{region['region']},"
                               f"{region['snr']:.2f},{region['edge_strength']:.2f}\n")
                               
        except Exception as e:
            self.console.print(f"[red]Error saving results: {e}[/red]")
            
    def _display_results(self, metrics):
        """Display metrics in a formatted table"""
        table = Table(title="Image Quality Metrics")
        
        table.add_column("Method", justify="left", style="cyan")
        table.add_column("Region", justify="center")
        table.add_column("SNR", justify="right")
        table.add_column("Edge Strength", justify="right")
        
        for method, regions in metrics.items():
            for region in regions:
                table.add_row(
                    method,
                    region['region'],
                    f"{region['snr']:.2f}",
                    f"{region['edge_strength']:.2f}"
                )
                
        self.console.print(table)

def main():
    # Set paths
    input_file = "eSFR_1920x1280_12b_GRGB_6500K_60Lux.raw"
    output_dir = "assignment2_results"
    
    # Initialize pipeline and run
    pipeline = DenoiseSharpenPipeline()
    pipeline.console.print("[yellow]Starting Advanced ISP Pipeline[/yellow]")
    
    results = pipeline.process_image(input_file, output_dir)
    
    pipeline.console.print(f"\n[green]Processing complete! "
                         f"Results saved in: {output_dir}[/green]")

if __name__ == "__main__":
    main()
