import streamlit as st
import numpy as np
import cv2
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# Attempt to import custom pipelines
try:
    from pipe import ISPPipeline, DenoiseSharpenPipeline
except ImportError:
    st.error("Could not import 'pipe'. Please ensure 'pipe.py' is in the same directory.")
    # Dummy fallbacks for offline testing
    class ISPPipeline:
        def read_raw(self, path): return np.random.randint(0, 4095, (1280, 1920), dtype=np.uint16)
        def demosaic(self, img): return cv2.cvtColor((img // 16).astype(np.uint8), cv2.COLOR_BAYER_GR2RGB)
        def white_balance(self, img): return img
        def apply_gamma(self, img): return ((img / img.max()) ** (1/2.2) * 255).astype(np.uint8)
    class DenoiseSharpenPipeline:
        def __init__(self): self.dncnn = None; self.device = 'cpu'
        def compute_metrics(self, img, roi): return np.random.rand() * 50, np.random.rand() * 100

# Utility: convert to 8-bit RGB
def convert_image_for_display(image):
    if image.dtype == np.uint16:
        image = (image / 256).astype(np.uint8)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2RGB)
    return image

# Convert to grayscale for display to avoid color distortion
def to_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

# Safe DnCNN wrapper
def _safe_dncnn_denoise(pipeline, image):
    try:
        if pipeline.dncnn is None:
            st.warning("DnCNN model not loaded. Skipping DnCNN denoising.")
            return image
        img = image
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        tensor = torch.from_numpy(img.transpose((2,0,1))).float() / 255.0
        tensor = tensor.unsqueeze(0).to(pipeline.device)
        with torch.no_grad():
            out = pipeline.dncnn(tensor)
        den = out.squeeze().cpu().numpy().transpose((1,2,0))
        return (np.clip(den,0,1)*255).astype(np.uint8)
    except Exception as e:
        st.error(f"DnCNN error: {e}")
        return image

# PDF generation (unchanged)
def generate_comprehensive_pdf(processed_images, metrics_results):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    import tempfile
    import matplotlib.pyplot as plt

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Title'], alignment=TA_CENTER, fontSize=18, textColor=HexColor('#2C3E50'))
    section_style = ParagraphStyle('Heading2', parent=styles['Heading2'], alignment=TA_LEFT, fontSize=14, textColor=HexColor('#34495E'))
    desc_style = ParagraphStyle('Normal', parent=styles['Normal'], alignment=TA_JUSTIFY, fontSize=10, textColor=HexColor('#2C3E50'))

    content = [Paragraph("Advanced Image Signal Processing Analysis Report", title_style), Spacer(1,12)]
    with tempfile.TemporaryDirectory() as tmpdir:
        plt.figure(figsize=(16,10), dpi=300)
        plt.suptitle("Image Processing Comparison", fontsize=16)
        rows = (len(processed_images)+1)//2
        for idx, (name,img) in enumerate(processed_images.items(),1):
            plt.subplot(rows,2,idx)
            plt.title(name)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap='gray')
            plt.axis('off')
        comp_path = os.path.join(tmpdir,'comp.png')
        plt.tight_layout(pad=3.0)
        plt.savefig(comp_path, dpi=300, bbox_inches='tight')
        plt.close()
        content.append(Image(comp_path, width=7*inch, height=5*inch)); content.append(Spacer(1,12))

        df = pd.DataFrame.from_dict(metrics_results, orient='index')
        plt.figure(figsize=(12,6), dpi=300)
        df.plot(kind='bar', rot=45)
        plt.title("Metrics")
        plt.tight_layout()
        mpath = os.path.join(tmpdir,'metrics.png')
        plt.savefig(mpath, dpi=300, bbox_inches='tight')
        plt.close()
        content.append(Image(mpath, width=7*inch, height=4*inch)); content.append(Spacer(1,12))

        table_data = [['Method','SNR','Edge Strength']] + [[m, f"{v['SNR']:.2f}", f"{v['Edge Strength']:.2f}"] for m,v in metrics_results.items()]
        tbl = Table(table_data, colWidths=[200,100,100])
        tbl.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),HexColor('#3498DB')),
            ('TEXTCOLOR',(0,0),(-1,0),HexColor('#FFFFFF')),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('GRID',(0,0),(-1,-1),1,HexColor('#95A5A6'))
        ]))
        content.append(tbl); content.append(Spacer(1,12))
        content.append(Paragraph("Key Insights", section_style))
        content.append(Paragraph("By rendering all results in grayscale, we eliminate color bias and focus on texture and luminance differences.", desc_style))
        doc.build(content)
    buffer.seek(0)
    return buffer

# Main Streamlit App
def main():
    st.set_page_config(page_title="Denoising and Edge Enhancement ", layout="wide")
    st.title("🖼️ Advanced Image Signal Processing (Grayscale)")

    uploaded = st.sidebar.file_uploader("Upload RAW Image", type=['raw'], help="12-bit RAW GRBG pattern")
    den_methods = st.sidebar.multiselect("Denoising Methods",["Gaussian","Median","Bilateral","DnCNN"], default=["Gaussian","Median"])
    sharp_methods = st.sidebar.multiselect("Sharpening Methods",["Unsharp","Laplacian"], default=["Laplacian"])
    x = st.sidebar.slider("ROI X", 0, 1920, 200);
    y = st.sidebar.slider("ROI Y", 0, 1280, 200)
    w = st.sidebar.slider("ROI Width", 100, 800, 400);
    h = st.sidebar.slider("ROI Height", 100, 800, 400)
    roi = (x, y, w, h)

    if uploaded:
        isp = ISPPipeline(); dsp = DenoiseSharpenPipeline()
        with open("temp.raw","wb") as f: f.write(uploaded.getvalue())
        raw = isp.read_raw("temp.raw")
        dem = isp.demosaic(raw)
        gam = isp.apply_gamma(dem)

        denoise_ops = {
            "Gaussian": lambda img: cv2.GaussianBlur(img,(5,5),1.0),
            "Median": lambda img: cv2.medianBlur(img,5),
            "Bilateral": lambda img: cv2.bilateralFilter(img,9,75,75),
            "DnCNN": lambda img: _safe_dncnn_denoise(dsp, img)
        }
        sharpen_ops = {
            "Unsharp": lambda img: cv2.addWeighted(img,1.5,cv2.GaussianBlur(img,(5,5),1.0),-0.5,0),
            "Laplacian": lambda img: cv2.filter2D(img,-1,np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))
        }
        all_ops = {**denoise_ops, **sharpen_ops}
        selected = den_methods + sharp_methods

        processed = {}
        roi_crops = {}
        metrics = {}

        for name in selected:
            if name not in all_ops: continue
            out_raw = all_ops[name](gam)
            disp8 = convert_image_for_display(out_raw)
            snr, es = dsp.compute_metrics(disp8, roi)
            metrics[name] = {'SNR': snr, 'Edge Strength': es}
            gray_disp = to_grayscale(disp8)
            x0,y0,w0,h0 = roi
            roi_crops[name] = gray_disp[y0:y0+h0, x0:x0+w0]
            boxed = gray_disp.copy()
            cv2.rectangle(boxed, (x0,y0), (x0+w0, y0+h0), (255,255,255), 2)
            processed[name] = boxed

        st.header("Processed Images with ROI (Grayscale)")
        cols = st.columns(min(len(processed), 2))
        for i, (n, img) in enumerate(processed.items()):
            with cols[i % 2]:
                st.subheader(n)
                st.image(img, channels="RGB", use_container_width=True)

        st.header("🔬 ROI Comparison (Grayscale)")
        cols = st.columns(len(roi_crops))
        for i, (n, img) in enumerate(roi_crops.items()):
            with cols[i]:
                st.subheader(n)
                st.image(img, use_container_width=True)

        st.header("Image Quality Metrics")
        df = pd.DataFrame.from_dict(metrics, orient='index')
        st.dataframe(df)

        st.header("Metrics Visualization")
        fig, ax = plt.subplots(figsize=(10,6))
        df.plot(kind='bar', rot=45, ax=ax)
        ax.set_title("Comparison of Methods")
        plt.tight_layout()
        st.pyplot(fig)

        st.header("Comparative Analysis Summary")
        st.markdown("""
### Denoising
- Gaussian: smooth noise reduction, may blur.
- Median: removes impulse noise, preserves edges.
- Bilateral: edge-preserving smoothing.
- DnCNN: learned denoising, detail-preserving.

### Sharpening
- Unsharp: enhances contrast, subtracts blur.
- Laplacian: edge amplification, high-frequency boost.
""")

        st.header("Download Results")
        csv = df.to_csv(index=True)
        st.download_button("Download Metrics CSV", data=csv, file_name="metrics.csv", mime="text/csv")
        pdf_buf = generate_comprehensive_pdf(processed, metrics)
        st.download_button("Download PDF Report", data=pdf_buf, file_name="report.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()