import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from skimage import measure, morphology, filters
import warnings
warnings.filterwarnings('ignore')
class CellViabilityAnalyzer:
    def __init__(self):
        self.original_image = None
        self.background_map = None
        self.background_corrected = None
        self.processed_image = None
        self.live_cells = 0
        self.dead_cells = 0
        self.total_cells = 0
        self.background_stats = {}
        
    def create_synthetic_cell_image(self, width=400, height=400):
        """Create a synthetic grayscale image with live and dead cells and varying background"""
        image = np.zeros((height, width), dtype=np.uint8)
        
        # Create non-uniform background with gradient and local variations
        # Add gradient background (simulates uneven illumination)
        y_gradient = np.linspace(15, 35, height)
        x_gradient = np.linspace(10, 25, width)
        Y, X = np.meshgrid(y_gradient, x_gradient, indexing='ij')
        background_base = Y + X
        
        # Add local background variations (simulates debris, uneven staining)
        np.random.seed(123)
        for i in range(8):
            center_y = np.random.randint(50, height-50)
            center_x = np.random.randint(50, width-50)
            sigma_y = np.random.randint(30, 80)
            sigma_x = np.random.randint(30, 80)
            intensity = np.random.randint(10, 30)
            
            y_coords, x_coords = np.ogrid[:height, :width]
            gaussian_blob = intensity * np.exp(
                -((x_coords - center_x)**2 / (2*sigma_x**2) + 
                  (y_coords - center_y)**2 / (2*sigma_y**2))
            )
            background_base += gaussian_blob
        
        # Add fine-grained noise
        noise = np.random.normal(0, 3, (height, width))
        background = background_base + noise
        background = np.clip(background, 0, 255)
        
        image = background.astype(np.uint8)
        
        # Create live cells (brighter, more circular)
        np.random.seed(42)
        num_live_cells = 30
        for _ in range(num_live_cells):
            x = np.random.randint(30, width-30)
            y = np.random.randint(30, height-30)
            radius = np.random.randint(8, 15)
            
            # Get local background intensity
            local_bg = image[y-5:y+5, x-5:x+5].mean()
            
            # Create circular cell with intensity relative to background
            Y_coords, X_coords = np.ogrid[:height, :width]
            dist_from_center = np.sqrt((X_coords - x)**2 + (Y_coords - y)**2)
            
            # Live cells are significantly brighter than local background
            cell_mask = dist_from_center <= radius
            cell_intensity = min(255, local_bg + 120 + np.random.randint(-15, 15))
            image[cell_mask] = cell_intensity
            
            # Add bright center
            inner_mask = dist_from_center <= radius * 0.6
            image[inner_mask] = min(255, cell_intensity + 20)
        
        # Create dead cells (moderately brighter than background)
        num_dead_cells = 20
        for _ in range(num_dead_cells):
            x = np.random.randint(30, width-30)
            y = np.random.randint(30, height-30)
            radius = np.random.randint(6, 12)
            
            # Get local background intensity
            local_bg = image[y-5:y+5, x-5:x+5].mean()
            
            # Create more irregular shape for dead cells
            Y_coords, X_coords = np.ogrid[:height, :width]
            dist_from_center = np.sqrt((X_coords - x)**2 + (Y_coords - y)**2)
            
            # Add irregularity
            angle = np.arctan2(Y_coords - y, X_coords - x)
            irregularity = 0.3 * np.sin(4 * angle)
            effective_radius = radius * (1 + irregularity)
            
            cell_mask = dist_from_center <= effective_radius
            # Dead cells are only moderately brighter than background
            cell_intensity = min(255, local_bg + 60 + np.random.randint(-10, 10))
            image[cell_mask] = cell_intensity
            
            # Dead cells often have fragmented appearance
            if np.random.random() > 0.5:
                fragment_mask = dist_from_center <= radius * 0.4
                image[fragment_mask] = max(local_bg, cell_intensity - 15)
        
        return image
    
    def estimate_background(self, image, method='morphological', kernel_size=50):
        """Estimate background intensity using different methods"""
        if method == 'morphological':
            # Morphological opening to remove cell-like structures
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            
        elif method == 'gaussian':
            # Large Gaussian blur to estimate smooth background
            background = ndimage.gaussian_filter(image, sigma=kernel_size/3)
            
        elif method == 'median':
            # Large median filter
            background = ndimage.median_filter(image, size=kernel_size)
            
        elif method == 'rolling_ball':
            # Simplified rolling ball algorithm
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            background = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
            background = image - background
            
        else:  # 'none'
            background = np.full_like(image, image.mean())
        
        return background.astype(np.uint8)
    
    def correct_background(self, image, background, method='subtract'):
        """Apply background correction"""
        if method == 'subtract':
            # Simple subtraction
            corrected = cv2.subtract(image, background)
            # Add small offset to prevent complete darkness
            corrected = cv2.add(corrected, 20)
            
        elif method == 'divide':
            # Division normalization
            # Avoid division by zero
            background_safe = np.where(background < 10, 10, background)
            corrected = (image.astype(np.float32) / background_safe.astype(np.float32) * 100).astype(np.uint8)
            corrected = np.clip(corrected, 0, 255)
            
        elif method == 'ratio':
            # Ratio method with global mean
            global_mean = background.mean()
            background_safe = np.where(background < 5, 5, background)
            ratio = global_mean / background_safe.astype(np.float32)
            corrected = (image.astype(np.float32) * ratio).astype(np.uint8)
            corrected = np.clip(corrected, 0, 255)
            
        else:  # 'none'
            corrected = image
            
        return corrected
    
    def calculate_background_stats(self, original, background, corrected):
        """Calculate background-related statistics"""
        stats = {
            'original_mean': float(original.mean()),
            'original_std': float(original.std()),
            'background_mean': float(background.mean()),
            'background_std': float(background.std()),
            'corrected_mean': float(corrected.mean()),
            'corrected_std': float(corrected.std()),
            'background_variation': float(background.std() / background.mean() * 100),
            'correction_improvement': float(original.std() / max(corrected.std(), 1))
        }
        return stats
    
    def preprocess_image(self, image, gaussian_sigma=1.0, median_size=3):
        """Apply preprocessing filters to reduce noise"""
        if gaussian_sigma > 0:
            processed = ndimage.gaussian_filter(image, sigma=gaussian_sigma)
        else:
            processed = image.copy()
        
        if median_size > 1:
            processed = ndimage.median_filter(processed, size=int(median_size))
        
        return processed
    
    def adaptive_threshold_cells(self, image, background_method='morphological', 
                               live_threshold_offset=80, dead_threshold_offset=40,
                               use_adaptive=True):
        """Segment cells using adaptive thresholding based on local background"""
        if use_adaptive:
            # Use local background for adaptive thresholding
            if background_method != 'none':
                background = self.estimate_background(image, background_method, kernel_size=50)
                live_threshold_map = background + live_threshold_offset
                dead_threshold_map = background + dead_threshold_offset
            else:
                # Global thresholding
                global_bg = image.mean()
                live_threshold_map = np.full_like(image, global_bg + live_threshold_offset)
                dead_threshold_map = np.full_like(image, global_bg + dead_threshold_offset)
            
            # Apply adaptive thresholds
            live_mask = image >= live_threshold_map
            dead_mask = (image >= dead_threshold_map) & (image < live_threshold_map)
        else:
            # Fixed global thresholds
            global_bg = image.mean()
            live_threshold = global_bg + live_threshold_offset
            dead_threshold = global_bg + dead_threshold_offset
            
            live_mask = image >= live_threshold
            dead_mask = (image >= dead_threshold) & (image < live_threshold)
        
        return live_mask, dead_mask
    
    def segment_cells(self, image, threshold_low=60, threshold_high=200, 
                     min_area=50, max_area=800, use_background_correction=True,
                     background_method='morphological', correction_method='subtract',
                     adaptive_threshold=True):
        """Segment cells with background correction"""
        
        # Estimate and correct background if enabled
        if use_background_correction and background_method != 'none':
            self.background_map = self.estimate_background(image, background_method)
            self.background_corrected = self.correct_background(image, self.background_map, correction_method)
            working_image = self.background_corrected
        else:
            self.background_map = np.full_like(image, image.mean())
            self.background_corrected = image
            working_image = image
        
        # Calculate statistics
        self.background_stats = self.calculate_background_stats(
            image, self.background_map, self.background_corrected)
        
        # Apply thresholding
        if adaptive_threshold and use_background_correction:
            live_mask, potential_dead_mask = self.adaptive_threshold_cells(
                working_image, background_method)
        else:
            # Traditional fixed thresholding
            live_mask = (working_image >= threshold_high)
            potential_dead_mask = (working_image >= threshold_low) & (working_image < threshold_high)
        
        # Clean up masks
        live_mask = morphology.remove_small_objects(live_mask, min_size=min_area//2)
        live_mask = morphology.remove_small_holes(live_mask, area_threshold=min_area//4)
        
        potential_dead_mask = morphology.remove_small_objects(potential_dead_mask, min_size=min_area//3)
        
        # Label connected components
        live_labels = measure.label(live_mask)
        dead_labels = measure.label(potential_dead_mask)
        
        return live_labels, dead_labels
    
    def analyze_cells(self, live_labels, dead_labels, min_area=50, max_area=800,
                     min_circularity=0.4):
        """Analyze segmented regions and classify as live or dead cells"""
        live_cells = []
        dead_cells = []
        
        # Analyze live cell candidates
        live_props = measure.regionprops(live_labels)
        for prop in live_props:
            area = prop.area
            if min_area <= area <= max_area:
                perimeter = prop.perimeter
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity >= min_circularity:
                        live_cells.append({
                            'centroid': prop.centroid,
                            'area': area,
                            'circularity': circularity,
                            'bbox': prop.bbox
                        })
        
        # Analyze dead cell candidates
        dead_props = measure.regionprops(dead_labels)
        for prop in dead_props:
            area = prop.area
            if min_area <= area <= max_area:
                perimeter = prop.perimeter
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity >= min_circularity * 0.7:
                        dead_cells.append({
                            'centroid': prop.centroid,
                            'area': area,
                            'circularity': circularity,
                            'bbox': prop.bbox
                        })
        
        return live_cells, dead_cells
    
    def process_image(self, threshold_low=60, threshold_high=200, gaussian_sigma=1.0,
                     median_size=3, min_area=50, max_area=800, min_circularity=0.4,
                     use_background_correction=True, background_method='morphological',
                     correction_method='subtract', adaptive_threshold=True):
        """Complete image processing pipeline with background correction"""
        if self.original_image is None:
            return
        
        # Preprocessing
        processed = self.preprocess_image(self.original_image, gaussian_sigma, median_size)
        
        # Segmentation with background correction
        live_labels, dead_labels = self.segment_cells(
            processed, threshold_low, threshold_high, min_area, max_area,
            use_background_correction, background_method, correction_method, adaptive_threshold)
        
        # Analysis
        live_cells, dead_cells = self.analyze_cells(
            live_labels, dead_labels, min_area, max_area, min_circularity)
        
        # Update results
        self.live_cells = len(live_cells)
        self.dead_cells = len(dead_cells)
        self.total_cells = self.live_cells + self.dead_cells
        
        # Create visualization image using background-corrected image
        vis_image = cv2.cvtColor(self.background_corrected, cv2.COLOR_GRAY2RGB)
        
        # Draw live cells in green
        for cell in live_cells:
            y, x = [int(c) for c in cell['centroid']]
            cv2.circle(vis_image, (x, y), 10, (0, 255, 0), 2)
            cv2.putText(vis_image, 'L', (x-5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw dead cells in red
        for cell in dead_cells:
            y, x = [int(c) for c in cell['centroid']]
            cv2.circle(vis_image, (x, y), 10, (255, 0, 0), 2)
            cv2.putText(vis_image, 'D', (x-5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        self.processed_image = vis_image
        return vis_image


if __name__ == "__main__":
    st.title("Interactive Cell Viability Analyzer")
    analyzer = CellViabilityAnalyzer()
    
    # Step 1: Create synthetic image
    analyzer.original_image = analyzer.create_synthetic_cell_image()
    if analyzer.original_image is None:
        st.error("Failed to generate synthetic image.")
        st.stop()

    st.image(analyzer.original_image, caption="Original Image", channels="GRAY")

    # Step 2: Streamlit Controls
    thresh_low = st.slider("Low Threshold", 10, 150, 60)
    thresh_high = st.slider("High Threshold", 100, 250, 200)
    gaussian_sigma = st.slider("Gaussian Sigma", 0.0, 3.0, 1.0)
    min_area = st.slider("Min Cell Area", 20, 200, 50)
    bg_kernel = st.slider("Background Kernel Size", 20, 100, 50)
    live_offset = st.slider("Live Cell Offset", 40, 150, 80)
    dead_offset = st.slider("Dead Cell Offset", 20, 100, 40)
    use_bg_correction = st.checkbox("Use Background Correction", value=True)
    adaptive_thresh = st.checkbox("Use Adaptive Thresholding", value=True)

    # Step 3: Process image
    processed = analyzer.process_image(
        threshold_low=thresh_low,
        threshold_high=thresh_high,
        gaussian_sigma=gaussian_sigma,
        median_size=3,
        min_area=min_area,
        use_background_correction=use_bg_correction,
        background_method='morphological',
        correction_method='subtract',
        adaptive_threshold=adaptive_thresh
    )

    # Show results horizontally
    cols = st.columns(3)

    with cols[0]:
        if analyzer.background_map is not None:
            st.image(analyzer.background_map, caption="Background Map", channels="GRAY")
        else:
            st.warning("No background map")

    with cols[1]:
        if analyzer.background_corrected is not None:
            st.image(analyzer.background_corrected, caption="Corrected", channels="GRAY")
        else:
            st.warning("No corrected image")

    with cols[2]:
        if analyzer.processed_image is not None:
            st.image(analyzer.processed_image, caption="Detected Cells")
        else:
            st.warning("No processed image")

    # Step 4: Stats
    viability = (analyzer.live_cells / max(analyzer.total_cells, 1)) * 100
    bg_stats = analyzer.background_stats or {}

    st.markdown("### Cell Statistics")
    st.write(f"Live Cells: {analyzer.live_cells}")
    st.write(f"Dead Cells: {analyzer.dead_cells}")
    st.write(f"Total Cells: {analyzer.total_cells}")
    st.write(f"Viability: {viability:.1f}%")

    st.markdown("### Background Stats")
    st.write(bg_stats)