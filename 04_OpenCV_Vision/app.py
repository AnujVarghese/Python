import io
import cv2
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
from vision_processor import VisionProcessor

st.set_page_config(
    page_title="SmartVision OpenCV Computer Vision Studio",
    page_icon="👁️",
    layout="wide"
)

st.markdown("""
<style>
    .cv-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f12711, #f5af19);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="cv-header">SmartVision: OpenCV Computer Vision Studio</div>', unsafe_allow_html=True)
st.write("Real-time Document Scanning, Color & Object HSV Segmentation, Facial Privacy Filter, and Advanced Computer Vision Operations.")

def generate_sample_document_image():
    """Generates a synthetic document image for testing scanner."""
    img = Image.new('RGB', (400, 300), color=(180, 180, 180))
    draw = ImageDraw.Draw(img)
    # Draw angled white rectangle representing paper
    paper_pts = [(50, 40), (330, 20), (360, 260), (30, 250)]
    draw.polygon(paper_pts, fill=(255, 255, 255), outline=(0, 0, 0))
    draw.text((70, 70), "INVOICE #10492", fill=(0, 0, 0))
    draw.text((70, 100), "Date: 2026-07-24", fill=(0, 0, 0))
    draw.text((70, 130), "Total Amount: $1,450.00", fill=(0, 0, 0))
    draw.line([(70, 160), (300, 160)], fill=(100, 100, 100), width=2)
    return img

option_source = st.sidebar.radio("Select Image Input", ["Sample Test Image", "Upload Image File"])

img_pil = None
if option_source == "Sample Test Image":
    img_pil = generate_sample_document_image()
else:
    file = st.sidebar.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "png", "jpeg"])
    if file is not None:
        img_pil = Image.open(file).convert("RGB")

if img_pil is not None:
    img_np = np.array(img_pil)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Document Scanner & Warp",
        "🎨 HSV Color Segmentation",
        "👤 Face Detection & Privacy Blur",
        "🛠️ Digital Filter Studio"
    ])
    
    with tab1:
        st.subheader("Document Boundary Detection & Perspective Warping")
        c_low = st.slider("Canny Low Threshold", 10, 150, 50)
        c_high = st.slider("Canny High Threshold", 50, 300, 150)
        
        edged, contour_img, warped = VisionProcessor.document_scanner(img_np, c_low, c_high)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(edged, caption="1. Canny Edges", use_container_width=True)
        with col2:
            st.image(contour_img, caption="2. Detected Document Boundary", use_container_width=True)
        with col3:
            st.image(warped, caption="3. Flattened Perspective Transform", use_container_width=True)
            
    with tab2:
        st.subheader("HSV Color Space Object Segmentation")
        st.write("Tune HSV sliders to isolate specific color regions and extract bounding boxes.")
        
        col_h, col_s, col_v = st.columns(3)
        with col_h:
            h_range = st.slider("Hue Range (0-179)", 0, 179, (0, 179))
        with col_s:
            s_range = st.slider("Saturation Range (0-255)", 0, 255, (50, 255))
        with col_v:
            v_range = st.slider("Value Range (0-255)", 0, 255, (50, 255))
            
        mask, seg_res, boxed = VisionProcessor.hsv_color_segmentation(
            img_np, h_range[0], h_range[1], s_range[0], s_range[1], v_range[0], v_range[1]
        )
        
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.image(mask, caption="Binary Color Mask", use_container_width=True)
        with col_s2:
            st.image(seg_res, caption="Segmented Color Channel", use_container_width=True)
        with col_s3:
            st.image(boxed, caption="Detected Bounding Rectangles", use_container_width=True)
            
    with tab3:
        st.subheader("Facial Detection & Privacy Anonymization Blur")
        blur_val = st.slider("Blur Intensity Kernel Size", 5, 51, 25, step=2)
        
        face_cnt, blurred_faces = VisionProcessor.face_privacy_blur(img_np, blur_strength=blur_val)
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.image(img_np, caption="Original Input Image", use_container_width=True)
        with col_f2:
            st.image(blurred_faces, caption=f"Anonymized Output ({face_cnt} faces detected)", use_container_width=True)
            
    with tab4:
        st.subheader("Interactive OpenCV Filter Studio")
        filter_choice = st.selectbox("Select Filter Algorithm", [
            "Gaussian Blur",
            "Canny Edge",
            "Sobel Gradient",
            "Laplacian",
            "Histogram Equalization (CLAHE)"
        ])
        kernel_sz = st.slider("Kernel / Window Size", 3, 21, 5, step=2)
        
        filtered_res = VisionProcessor.apply_filter(img_np, filter_choice, ksize=kernel_sz)
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.image(img_np, caption="Original Image", use_container_width=True)
        with col_p2:
            st.image(filtered_res, caption=f"Filtered Output ({filter_choice})", use_container_width=True)
