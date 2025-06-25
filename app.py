import streamlit as st
import numpy as np
import cv2
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

from pipe import ISPPipeline, DenoiseSharpenPipeline

def _safe_dncnn_denoise(pipeline, image):
    """
    Safely apply DnCNN denoising with fallback
    """
    try:
        # Check if DnCNN model is loaded
        if pipeline.dncnn is None:
            print("DnCNN model not loaded. Falling back to original image.")
            return image
        
        # Ensure image is in correct format for DnCNN
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Preprocess image for DnCNN
        img_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(pipeline.device)
        
        # Apply denoising
        with torch.no_grad():
            denoised_tensor = pipeline.dncnn(img_tensor)
        
        # Convert back to numpy
        denoised = denoised_tensor.squeeze().cpu().numpy().transpose((1, 2, 0))
        denoised = (np.clip(denoised, 0, 1) * 255).astype(np.uint8)
        
        return denoised
    
    except Exception as e:
        print(f"Error in DnCNN denoising: {e}")
        return image

def convert_image_for_display(image, roi=None):
    """Convert image for Streamlit display and optionally add ROI rectangle"""
    if image.dtype == np.uint16:
        image = (image / 256).astype(np.uint8)
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2RGB)
    
    # Add red ROI rectangle if ROI is provided
    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red rectangle
    
    return image

def generate_comprehensive_pdf(processed_images, metrics_results):
    """
    Generate a comprehensive PDF report with advanced visualizations, high-resolution images, and detailed analysis
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    import tempfile
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'Title', 
        parent=styles['Title'], 
        alignment=TA_CENTER, 
        fontSize=18, 
        textColor=HexColor('#2C3E50')
    )
    
    section_style = ParagraphStyle(
        'Section', 
        parent=styles['Heading2'], 
        alignment=TA_LEFT, 
        fontSize=14, 
        textColor=HexColor('#34495E')
    )
    
    description_style = ParagraphStyle(
        'Description',
        parent=styles['Normal'],
        alignment=TA_JUSTIFY,
        fontSize=10,
        textColor=HexColor('#2C3E50')
    )
    
    content = []
    
    # Title
    content.append(Paragraph("Advanced Image Signal Processing Analysis Report", title_style))
    content.append(Spacer(1, 12))
    
    # Create temporary directory for high-resolution images
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Processed Images Comparison Section
        content.append(Paragraph("Comparative Image Processing Results", section_style))
        
        # High-resolution side-by-side image comparison
        plt.figure(figsize=(16, 10), dpi=300)
        plt.suptitle("Image Processing Methods Comparison", fontsize=16)
        
        num_methods = len(processed_images)
        grid_rows = (num_methods + 1) // 2  # Calculate rows needed
        
        for i, (name, img) in enumerate(processed_images.items(), 1):
            plt.subplot(grid_rows, 2, i)
            plt.title(name, fontsize=12)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            plt.axis('off')
        
        plt.tight_layout(pad=3.0)
        images_comparison_path = os.path.join(tmpdirname, 'high_res_comparison.png')
        plt.savefig(images_comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Add high-resolution comparison image
        img = Image(images_comparison_path, width=7*inch, height=5*inch)
        content.append(img)
        content.append(Spacer(1, 12))
        
        # Detailed Method Descriptions
        content.append(Paragraph("Method Descriptions", section_style))
        
        method_descriptions = {
            "Gaussian Denoising": 
                "A linear filtering technique that smooths images by replacing each pixel's value "
                "with a weighted average of neighboring pixels. Effective for reducing Gaussian noise, "
                "but may blur image details.",
            
            "Median Denoising": 
                "A non-linear filtering method that replaces each pixel's value with the median of "
                "neighboring pixel intensities. Particularly effective at removing salt-and-pepper noise "
                "while preserving edge details.",
            
            "Bilateral Denoising": 
                "An advanced edge-preserving smoothing filter that considers both spatial and intensity "
                "differences. Reduces noise while maintaining sharp edges and local image structures.",
            
            "DnCNN Denoising": 
                "A deep learning-based denoising approach using convolutional neural networks. "
                "Adaptively learns noise patterns and removes them with high precision, "
                "preserving image details and textures.",
            
            "Unsharp Mask": 
                "A sharpening technique that enhances image details by subtracting a blurred version "
                "of the image from the original. Increases local contrast and brings out fine details.",
            
            "Laplacian Sharpening": 
                "An edge enhancement method that uses the Laplacian operator to detect and amplify "
                "high-frequency image details. Increases image sharpness by emphasizing rapid "
                "intensity changes."
        }
        
        for method, description in method_descriptions.items():
            if method in processed_images:
                content.append(Paragraph(f"<b>{method}</b>", section_style))
                content.append(Paragraph(description, description_style))
                content.append(Spacer(1, 6))
        
        # Metrics Visualization and Table
        content.append(Paragraph("Performance Metrics", section_style))
        
        metrics_df = pd.DataFrame.from_dict(metrics_results, orient='index')
        
        plt.figure(figsize=(12, 6), dpi=300)
        metrics_df.plot(kind='bar', rot=45)
        plt.title("Comparative Image Quality Metrics", fontsize=14)
        plt.xlabel("Processing Method", fontsize=12)
        plt.ylabel("Metric Value", fontsize=12)
        plt.tight_layout()
        
        metrics_path = os.path.join(tmpdirname, 'metrics_comparison.png')
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        metrics_img = Image(metrics_path, width=7*inch, height=4*inch)
        content.append(metrics_img)
        content.append(Spacer(1, 12))
        
        # Detailed Metrics Table
        metrics_table_data = [['Method', 'SNR', 'Edge Strength']]
        for method, metrics in metrics_results.items():
            metrics_table_data.append([
                method, 
                f"{metrics['SNR']:.2f}", 
                f"{metrics['Edge Strength']:.2f}"
            ])
        
        metrics_table = Table(metrics_table_data, colWidths=[200, 100, 100])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), HexColor('#3498DB')),
            ('TEXTCOLOR', (0,0), (-1,0), HexColor('#FFFFFF')),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), HexColor('#F2F3F4')),
            ('GRID', (0,0), (-1,-1), 1, HexColor('#95A5A6'))
        ]))
        content.append(metrics_table)
        
        # Final Insights
        content.append(Spacer(1, 12))
        content.append(Paragraph("Key Insights", section_style))
        insights_text = """
        This comprehensive analysis reveals nuanced performance characteristics of various image processing techniques. 
        Each method demonstrates unique strengths in noise reduction, edge preservation, and detail enhancement. 
        The choice of processing technique depends on specific image characteristics and desired output quality.
        """
        content.append(Paragraph(insights_text, description_style))
        
        # Build PDF
        doc.build(content)
    
    buffer.seek(0)
    return buffer


def main():
    st.set_page_config(page_title="Advanced Image Signal Processing", layout="wide")
    
    st.title("🖼️ Advanced Image Signal Processing")
    
    # Sidebar for configuration
    st.sidebar.header("Image Processing Options")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload RAW Image", 
        type=['raw'], 
        help="Upload a 12-bit RAW image in GRBG Bayer pattern"
    )
    
    # Processing methods selection
    st.sidebar.subheader("Processing Methods")
    denoising_methods = st.sidebar.multiselect(
        "Select Denoising Methods",
        [
            "Gaussian Denoising", 
            "Median Denoising", 
            "Bilateral Denoising", 
            "DnCNN Denoising"
        ],
        default=["Gaussian Denoising", "Median Denoising"]
    )
    
    # Sharpening methods
    sharpening_methods = st.sidebar.multiselect(
        "Select Sharpening Methods",
        [
            "Unsharp Mask", 
            "Laplacian Sharpening"
        ],
        default=["Laplacian Sharpening"]
    )
    
    # ROI selection
    st.sidebar.subheader("Region of Interest")
    roi_x = st.sidebar.slider("ROI X Position", 0, 1920, 200)
    roi_y = st.sidebar.slider("ROI Y Position", 0, 1280, 200)
    roi_width = st.sidebar.slider("ROI Width", 100, 800, 400)
    roi_height = st.sidebar.slider("ROI Height", 100, 800, 400)
    roi = (roi_x, roi_y, roi_width, roi_height)
    
    if uploaded_file is not None:
        # Initialize pipelines
        isp_pipeline = ISPPipeline()
        denoiser_pipeline = DenoiseSharpenPipeline()
        
        # Save uploaded file temporarily
        with open("temp_upload.raw", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Read RAW image
        raw_image = isp_pipeline.read_raw("temp_upload.raw")
        
        # Basic image processing steps
        demosaiced = isp_pipeline.demosaic(raw_image)
        wb_image = isp_pipeline.white_balance(demosaiced)
        gamma_image = isp_pipeline.apply_gamma(wb_image)
        
        # Denoising methods
        denoising_methods_dict = {
            "Gaussian Denoising": lambda img: cv2.GaussianBlur(img, (5, 5), 1.0),
            "Median Denoising": lambda img: cv2.medianBlur(img, 5),
            "Bilateral Denoising": lambda img: cv2.bilateralFilter(img, 9, 75, 75),
            "DnCNN Denoising": lambda img: _safe_dncnn_denoise(denoiser_pipeline, img)
        }
        
        # Sharpening methods
        sharpening_methods_dict = {
            "Unsharp Mask": lambda img: cv2.addWeighted(img, 1.5, cv2.GaussianBlur(img, (5, 5), 1.0), -0.5, 0),
            "Laplacian Sharpening": lambda img: cv2.filter2D(img, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
        }
        
        # Process images
        processed_images = {}
        metrics_results = {}
        
        # Apply denoising methods
        for method_name in denoising_methods:
            denoised_img = denoising_methods_dict[method_name](gamma_image)
            processed_images[method_name] = convert_image_for_display(denoised_img, roi)
            
            # Compute metrics
            snr, edge_strength = denoiser_pipeline.compute_metrics(processed_images[method_name], roi)
            metrics_results[method_name] = {
                'SNR': snr,
                'Edge Strength': edge_strength
            }
        
        # Apply sharpening methods
        for method_name in sharpening_methods:
            sharpened_img = sharpening_methods_dict[method_name](gamma_image)
            processed_images[method_name] = convert_image_for_display(sharpened_img, roi)
            
            # Compute metrics
            snr, edge_strength = denoiser_pipeline.compute_metrics(processed_images[method_name], roi)
            metrics_results[method_name] = {
                'SNR': snr,
                'Edge Strength': edge_strength
            }
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Processed Images")
            for name, img in processed_images.items():
                st.subheader(name)
                st.image(img, channels="RGB")
        
        with col2:
            st.header("Image Quality Metrics")
            metrics_df = pd.DataFrame.from_dict(metrics_results, orient='index')
            st.dataframe(metrics_df)
        
        # Generate detailed visualization of metrics
        st.header("Metrics Visualization")
        plt.figure(figsize=(10, 6))
        metrics_df.plot(kind='bar', rot=45)
        plt.title("Comparison of Image Processing Methods")
        plt.tight_layout()
        st.pyplot(plt)
        
        # Comparative Analysis
        st.header("Comparative Analysis")
        analysis_text = """
        ### Denoising Method Comparison
        - **Gaussian Filtering**: Provides smooth noise reduction but may blur image details.
        - **Median Filtering**: Effective at removing salt-and-pepper noise while preserving edges.
        - **Bilateral Filtering**: Reduces noise while preserving edge information.
        - **DnCNN Denoising**: AI-based approach that can adaptively remove noise.
        
        ### Sharpening Method Comparison
        - **Unsharp Mask**: Enhances image details by subtracting a blurred version of the image.
        - **Laplacian Sharpening**: Edge enhancement through high-frequency detail amplification.
        """
        st.markdown(analysis_text)
        
        # Download options
        st.header("Download Results")
        
        # Metrics CSV
        results_csv = metrics_df.to_csv(index=True)
        st.download_button(
            label="Download Metrics CSV",
            data=results_csv,
            file_name="image_processing_metrics.csv",
            mime="text/csv"
        )
        
        # PDF Report
        pdf_buffer = generate_comprehensive_pdf(processed_images, metrics_results)
        st.download_button(
            label="Download Detailed PDF Report",
            data=pdf_buffer,
            file_name="image_processing_comparison.pdf",
            mime="application/pdf"
        )

if __name__ == "__main__":
    main()
