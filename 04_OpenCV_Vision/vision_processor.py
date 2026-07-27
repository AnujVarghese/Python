"""
SmartVision OpenCV Processing Engine
Includes:
- Document Edge Detection & Perspective Scanner
- Color & HSV Object Segmentation
- Face Detection & Privacy Blur Filter
- Advanced Image Filters & Morphological Operations
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path


FACE_CASCADE_FILE = "haarcascade_frontalface_default.xml"


class VisionProcessorError(RuntimeError):
    """Raised when an optional OpenCV feature is unavailable."""

class VisionProcessor:
    """Core OpenCV Image Processing Utilities."""

    @staticmethod
    def document_scanner(image_np: np.ndarray, low_thresh: int = 50, high_thresh: int = 150):
        """Detects paper document boundaries and performs 4-point perspective transform."""
        orig = image_np.copy()
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, low_thresh, high_thresh)

        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        doc_contour = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_contour = approx
                break

        contour_img = orig.copy()
        warped = orig.copy()

        if doc_contour is not None:
            cv2.drawContours(contour_img, [doc_contour], -1, (0, 255, 0), 3)
            pts = doc_contour.reshape(4, 2)
            warped = VisionProcessor._four_point_transform(orig, pts)

        return edged, contour_img, warped

    @staticmethod
    def _four_point_transform(image, pts):
        """Performs 4-point perspective transformation."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # Top-left
        rect[2] = pts[np.argmax(s)] # Bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # Top-right
        rect[3] = pts[np.argmax(diff)] # Bottom-left

        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    @staticmethod
    def hsv_color_segmentation(image_np: np.ndarray, h_min: int, h_max: int, s_min: int, s_max: int, v_min: int, v_max: int):
        """Segments objects by HSV color bounds."""
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(image_np, image_np, mask=mask)

        # Draw bounding boxes around segmented objects
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxed_img = image_np.copy()
        for c in contours:
            if cv2.contourArea(c) > 300:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(boxed_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return mask, result, boxed_img

    @staticmethod
    def face_privacy_blur(image_np: np.ndarray, blur_strength: int = 25):
        """Detects faces using Haar Cascade and applies privacy blur."""
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        face_cascade = VisionProcessor._load_face_cascade()

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        blurred_img = image_np.copy()

        blur_val = blur_strength if blur_strength % 2 != 0 else blur_strength + 1
        for (x, y, w, h) in faces:
            roi = blurred_img[y:y+h, x:x+w]
            roi_blurred = cv2.GaussianBlur(roi, (blur_val, blur_val), 30)
            blurred_img[y:y+h, x:x+w] = roi_blurred
            cv2.rectangle(blurred_img, (x, y), (x+w, y+h), (0, 255, 255), 2)

        return len(faces), blurred_img

    @staticmethod
    def _load_face_cascade():
        """Load OpenCV's bundled frontal-face Haar cascade with clear errors."""
        if not hasattr(cv2, "CascadeClassifier"):
            raise VisionProcessorError(
                "This OpenCV build does not include CascadeClassifier. "
                "Install opencv-python-headless 4.x for the face privacy blur."
            )

        candidates = VisionProcessor._face_cascade_candidates()
        for cascade_path in candidates:
            if not cascade_path.exists():
                continue

            classifier = cv2.CascadeClassifier(str(cascade_path))
            if not classifier.empty():
                return classifier

        searched = ", ".join(str(path) for path in candidates) or "no cascade paths"
        raise VisionProcessorError(
            f"OpenCV could not load {FACE_CASCADE_FILE}. "
            "Install opencv-python-headless 4.x, then redeploy the app. "
            f"Searched: {searched}"
        )

    @staticmethod
    def _face_cascade_candidates():
        """Return likely locations for the bundled OpenCV Haar cascade."""
        candidates = []

        data_module = getattr(cv2, "data", None)
        haarcascades = getattr(data_module, "haarcascades", None)
        if haarcascades:
            candidates.append(Path(haarcascades) / FACE_CASCADE_FILE)

        cv2_file = getattr(cv2, "__file__", None)
        if cv2_file:
            candidates.append(Path(cv2_file).resolve().parent / "data" / FACE_CASCADE_FILE)

        candidates.append(Path(__file__).resolve().parent / "data" / FACE_CASCADE_FILE)

        unique_candidates = []
        seen = set()
        for path in candidates:
            path_key = str(path)
            if path_key not in seen:
                seen.add(path_key)
                unique_candidates.append(path)

        return unique_candidates

    @staticmethod
    def apply_filter(image_np: np.ndarray, filter_name: str, ksize: int = 5):
        """Applies various OpenCV filters."""
        ksize = ksize if ksize % 2 != 0 else ksize + 1
        
        if filter_name == "Canny Edge":
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            return cv2.Canny(gray, 100, 200)
        elif filter_name == "Sobel Gradient":
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            grad = np.sqrt(sobelx**2 + sobely**2)
            return np.uint8(np.clip(grad, 0, 255))
        elif filter_name == "Laplacian":
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
            return np.uint8(np.clip(np.abs(lap), 0, 255))
        elif filter_name == "Gaussian Blur":
            return cv2.GaussianBlur(image_np, (ksize, ksize), 0)
        elif filter_name == "Histogram Equalization (CLAHE)":
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            return clahe.apply(gray)
        else:
            return image_np
