import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

def praktikum_9_2():
    """
    Implementasi edge detection dan region-based segmentation
    """
    print("\nPRAKTIKUM 9.2: EDGE DETECTION DAN REGION-BASED SEGMENTATION")
    print("=" * 70)
    
    # Buat citra test dengan berbagai edge types
    def create_edge_test_image():
        """Create test image with different edge types"""
        img = np.zeros((300, 400), dtype=np.uint8)
        
        # Step edge (sharp transition)
        cv2.rectangle(img, (50, 50), (150, 150), 100, -1)
        cv2.rectangle(img, (151, 50), (250, 150), 200, -1)
        
        # Ramp edge (gradual transition)
        for i in range(50, 150):
            img[160:240, i] = 50 + (i - 50) * 2
        
        # Roof edge (triangular)
        triangle_cnt = np.array([(300, 160), (350, 240), (250, 240)])
        cv2.drawContours(img, [triangle_cnt], 0, 150, -1)
        
        # Line edge (thin line)
        cv2.line(img, (50, 260), (350, 260), 200, 3)
        
        # Add noise untuk testing robustness
        noise = np.random.normal(0, 15, img.shape)
        img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
        
        return img
    
    # Implementasi berbagai edge detectors
    def apply_sobel_edge_detection(image, ksize=3):
        """Sobel edge detection"""
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
        
        # Compute magnitude
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        # Compute direction
        direction = np.arctan2(sobely, sobelx) * 180 / np.pi
        
        return magnitude.astype(np.uint8), direction
    
    def apply_prewitt_edge_detection(image):
        """Prewitt edge detection"""
        # Define Prewitt kernels
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        
        prewittx = cv2.filter2D(image.astype(np.float64), -1, kernelx)
        prewitty = cv2.filter2D(image.astype(np.float64), -1, kernely)
        
        # Compute magnitude
        magnitude = np.sqrt(prewittx**2 + prewitty**2)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        return magnitude.astype(np.uint8)
    
    def apply_canny_edge_detection(image, low_threshold=50, high_threshold=150):
        """Canny edge detection"""
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (5, 5), 1.4)
        
        # Apply Canny
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        
        return edges
    
    def apply_laplacian_edge_detection(image):
        """Laplacian edge detection"""
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        laplacian = np.abs(laplacian)
        laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)
        
        return laplacian.astype(np.uint8)
    
    # Region-based segmentation methods
    def region_growing_segmentation(image, seeds, threshold=20):
        """Region growing segmentation"""
        segmented = np.zeros_like(image)
        visited = np.zeros_like(image, dtype=bool)
        
        # Convert seeds to list if single seed
        if isinstance(seeds, tuple):
            seeds = [seeds]
        
        for seed in seeds:
            if visited[seed]:
                continue
                
            stack = [seed]
            region_pixels = []
            
            while stack:
                x, y = stack.pop()
                
                if visited[x, y]:
                    continue
                    
                visited[x, y] = True
                current_value = image[x, y]
                region_pixels.append(current_value)
                segmented[x, y] = 255
                
                # Check 4-connected neighbors
                neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
                region_mean = np.mean(region_pixels)
                
                for nx, ny in neighbors:
                    if (0 <= nx < image.shape[0] and 0 <= ny < image.shape[1] 
                        and not visited[nx, ny]):
                        neighbor_value = image[nx, ny]
                        if abs(neighbor_value - region_mean) < threshold:
                            stack.append((nx, ny))
        
        return segmented
    
    def region_splitting_merging(image, min_size=32, threshold=20):
        """Region splitting and merging algorithm"""
        def should_split(region):
            """Check if region should be split"""
            return np.std(region) > threshold and region.shape[0] > min_size
        
        def split_region(region):
            """Split region into 4 sub-regions"""
            h, w = region.shape
            return [
                region[:h//2, :w//2],  # Top-left
                region[:h//2, w//2:],  # Top-right
                region[h//2:, :w//2],  # Bottom-left
                region[h//2:, w//2:]   # Bottom-right
            ]
        
        def merge_regions(regions):
            """Merge similar adjacent regions"""
            # This is a simplified implementation
            merged = []
            used = [False] * len(regions)
            
            for i, region in enumerate(regions):
                if used[i]:
                    continue
                    
                current_region = region
                used[i] = True
                
                # Try to merge with similar regions
                for j, other_region in enumerate(regions[i+1:], i+1):
                    if used[j]:
                        continue
                        
                    # Check similarity (simplified)
                    if (abs(np.mean(current_region) - np.mean(other_region)) < threshold and
                        abs(np.std(current_region) - np.std(other_region)) < threshold/2):
                        
                        # Merge regions (simplified - just combine)
                        h1, w1 = current_region.shape
                        h2, w2 = other_region.shape
                        h_max = max(h1, h2)
                        w_max = max(w1, w2)
                        
                        merged_region = np.zeros((h_max, w_max))
                        merged_region[:h1, :w1] = current_region
                        merged_region[:h2, :w2] = np.maximum(
                            merged_region[:h2, :w2], other_region
                        )
                        
                        current_region = merged_region
                        used[j] = True
                
                merged.append(current_region)
            
            return merged
        
        # Start with whole image
        regions = [image]
        
        # Split phase
        splitting = True
        while splitting:
            splitting = False
            new_regions = []
            
            for region in regions:
                if should_split(region):
                    new_regions.extend(split_region(region))
                    splitting = True
                else:
                    new_regions.append(region)
            
            regions = new_regions
        
        # Merge phase
        regions = merge_regions(regions)
        
        # Create visualization
        result = np.zeros_like(image)
        current_value = 50
        
        for region in regions:
            h, w = region.shape
            result[:h, :w] = current_value
            current_value = min(current_value + 50, 255)
        
        return result
    
    # Buat citra test
    test_image = create_edge_test_image()
    
    # Terapkan berbagai edge detectors
    edge_results = {}
    
    # Sobel
    sobel_magnitude, sobel_direction = apply_sobel_edge_detection(test_image)
    edge_results['Sobel'] = sobel_magnitude
    
    # Prewitt
    prewitt_magnitude = apply_prewitt_edge_detection(test_image)
    edge_results['Prewitt'] = prewitt_magnitude
    
    # Canny
    canny_edges = apply_canny_edge_detection(test_image)
    edge_results['Canny'] = canny_edges
    
    # Laplacian
    laplacian_edges = apply_laplacian_edge_detection(test_image)
    edge_results['Laplacian'] = laplacian_edges
    
    # Region-based segmentation
    region_results = {}
    
    # Region growing dengan berbagai seeds
    # FIXED: Changed (300, 200) to (200, 300) to stay within image bounds
    seeds = [(75, 100),   # Step edge region
             (200, 100),  # Ramp edge region  
             (200, 300)]  # Roof edge region 
    
    region_grown = region_growing_segmentation(test_image, seeds, threshold=25)
    region_results['Region Growing'] = region_grown
    
    # Region splitting and merging
    split_merge_result = region_splitting_merging(test_image, min_size=32, threshold=20)
    region_results['Split & Merge'] = split_merge_result
    
    # Visualisasi hasil edge detection
    print("\nEDGE DETECTION RESULTS")
    print("-" * 40)
    
    # PERUBAHAN: Ubah grid dari (2, 3) menjadi (2, 4) dan sesuaikan ukuran figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: Original and edge detectors
    axes[0, 0].imshow(test_image, cmap='gray')
    axes[0, 0].set_title('Test Image\n(Various Edge Types)')
    axes[0, 0].axis('off')
    
    methods = ['Sobel', 'Prewitt', 'Canny']
    for idx, method in enumerate(methods, 1):
        axes[0, idx].imshow(edge_results[method], cmap='gray')
        axes[0, idx].set_title(f'{method} Edge Detection')
        axes[0, idx].axis('off')
    
    # Row 2: More edge detectors and edge direction
    axes[1, 0].imshow(edge_results['Laplacian'], cmap='gray')
    axes[1, 0].set_title('Laplacian Edge Detection')
    axes[1, 0].axis('off')
    
    # Sobel direction (as HSV image)
    hsv = np.zeros((test_image.shape[0], test_image.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (sobel_direction + 180) / 2  # Hue based on direction
    hsv[..., 1] = 255  # Full saturation
    hsv[..., 2] = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    axes[1, 1].imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    axes[1, 1].set_title('Sobel Edge Direction\n(Hue=Direction, Value=Magnitude)')
    axes[1, 1].axis('off')
    
    # Combined edges visualization
    combined_edges = np.zeros_like(test_image)
    for method in ['Sobel', 'Canny']:
        edges_norm = edge_results[method] / 255.0
        combined_edges = np.maximum(combined_edges, edges_norm * 255)
    
    axes[1, 2].imshow(combined_edges, cmap='gray')
    axes[1, 2].set_title('Combined Edges\n(Sobel + Canny)')
    axes[1, 2].axis('off')
    
    # PERUBAHAN: Matikan/Sembunyikan axis yang kosong di baris ke-2 kolom ke-4
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    plt.tight_layout()
    plt.show()
    
    # Analisis komparatif edge detectors
    print("\nEDGE DETECTOR COMPARISON ANALYSIS")
    print("-" * 50)
    
    # Hitung edge density dan continuity
    def analyze_edge_image(edge_image, threshold=128):
        """Analyze edge image characteristics"""
        binary_edges = (edge_image > threshold).astype(np.uint8)
        
        # Edge density
        edge_density = np.sum(binary_edges) / binary_edges.size
        
        # Edge continuity (using connected components)
        num_labels, labels = cv2.connectedComponents(binary_edges)
        avg_component_size = np.sum(binary_edges) / max(num_labels - 1, 1)
        
        # Thinness ratio (perimeter^2 / area)
        contours, _ = cv2.findContours(binary_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thinness_ratios = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if area > 0:
                thinness = perimeter**2 / (4 * np.pi * area)
                thinness_ratios.append(thinness)
        
        avg_thinness = np.mean(thinness_ratios) if thinness_ratios else 0
        
        return {
            'density': edge_density,
            'num_components': num_labels - 1,
            'avg_component_size': avg_component_size,
            'avg_thinness': avg_thinness
        }
    
    print(f"{'Method':<15} {'Density':<10} {'Components':<12} {'Avg Size':<12} {'Thinness':<10}")
    print("-" * 60)
    
    for method_name, edge_image in edge_results.items():
        analysis = analyze_edge_image(edge_image)
        print(f"{method_name:<15} {analysis['density']:<10.4f} {analysis['num_components']:<12} "
              f"{analysis['avg_component_size']:<12.2f} {analysis['avg_thinness']:<10.2f}")
    
    # Visualisasi region-based segmentation
    print("\nREGION-BASED SEGMENTATION RESULTS")
    print("-" * 50)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Original and region growing
    axes[0, 0].imshow(test_image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(region_results['Region Growing'], cmap='gray')
    axes[0, 1].set_title('Region Growing\n(with seed points)')
    axes[0, 1].axis('off')
    
    # Tampilkan seed points
    seed_display = test_image.copy()
    for seed in seeds:
        cv2.circle(seed_display, (seed[1], seed[0]), 5, 255, -1)
    axes[0, 2].imshow(seed_display, cmap='gray')
    axes[0, 2].set_title('Seed Points\n(For Region Growing)')
    axes[0, 2].axis('off')
    
    # Row 2: Split & merge dan hybrid approach
    axes[1, 0].imshow(region_results['Split & Merge'], cmap='gray')
    axes[1, 0].set_title('Region Splitting & Merging')
    axes[1, 0].axis('off')
    
    # Hybrid approach: Edge-based + Region-based
    # Gunakan edges sebagai constraint untuk region growing
    edge_constrained = canny_edges.copy()
    edge_constrained = cv2.dilate(edge_constrained, np.ones((3,3), np.uint8), iterations=1)
    
    # Region growing dengan edge constraints
    constrained_result = np.zeros_like(test_image)
    for seed in seeds:
        # Simple constrained region growing
        mask = np.zeros_like(test_image, dtype=np.uint8)
        mask[seed] = 1
        
        # Dilate mask but stop at edges
        for _ in range(20):
            dilated = cv2.dilate(mask, np.ones((3,3), np.uint8))
            # Stop dilation at edges
            dilated[edge_constrained > 0] = 0
            if np.array_equal(dilated, mask):
                break
            mask = dilated
        
        constrained_result[mask > 0] = 255
    
    axes[1, 1].imshow(constrained_result, cmap='gray')
    axes[1, 1].set_title('Edge-constrained\nRegion Growing')
    axes[1, 1].axis('off')
    
    # Region boundaries overlay
    boundaries = cv2.Canny(region_results['Region Growing'].astype(np.uint8), 50, 150)
    overlay = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR)
    overlay[boundaries > 0] = [0, 255, 0]
    
    axes[1, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Region Boundaries\nOverlay on Original')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Analisis region characteristics
    print("\nREGION CHARACTERISTICS ANALYSIS")
    print("-" * 50)
    
    def analyze_regions(binary_image):
        """Analyze segmented regions"""
        # Label connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image, connectivity=8
        )
        
        region_info = []
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            left = stats[i, cv2.CC_STAT_LEFT]
            top = stats[i, cv2.CC_STAT_TOP]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Extract region pixels
            region_mask = (labels == i)
            region_pixels = test_image[region_mask]
            
            region_info.append({
                'label': i,
                'area': area,
                'centroid': centroids[i],
                'mean_intensity': np.mean(region_pixels),
                'std_intensity': np.std(region_pixels),
                'bbox': (left, top, width, height)
            })
        
        return region_info
    
    # Analisis region growing results
    region_grown_binary = (region_results['Region Growing'] > 0).astype(np.uint8)
    region_info = analyze_regions(region_grown_binary)
    
    print(f"Number of regions detected: {len(region_info)}")
    print("\nRegion Statistics:")
    print(f"{'Region':<8} {'Area':<10} {'Mean':<10} {'Std':<10} {'Centroid':<20}")
    print("-" * 60)
    
    for info in region_info[:5]:  # Show first 5 regions
        centroid_str = f"({info['centroid'][0]:.1f}, {info['centroid'][1]:.1f})"
        print(f"{info['label']:<8} {info['area']:<10} {info['mean_intensity']:<10.2f} "
              f"{info['std_intensity']:<10.2f} {centroid_str:<20}")
    
    # Visualisasi region properties
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Region area distribution
    areas = [info['area'] for info in region_info]
    axes[0].hist(areas, bins=20, alpha=0.7, edgecolor='black')
    axes[0].set_title('Region Area Distribution')
    axes[0].set_xlabel('Area (pixels)')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # Region intensity distribution
    intensities = [info['mean_intensity'] for info in region_info]
    axes[1].hist(intensities, bins=20, alpha=0.7, edgecolor='black', color='orange')
    axes[1].set_title('Region Mean Intensity Distribution')
    axes[1].set_xlabel('Mean Intensity')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)
    
    # Scatter plot: Area vs Intensity
    axes[2].scatter(areas, intensities, alpha=0.6)
    axes[2].set_title('Region Area vs Mean Intensity')
    axes[2].set_xlabel('Area (pixels)')
    axes[2].set_ylabel('Mean Intensity')
    axes[2].grid(True, alpha=0.3)
    
    # Add regression line
    if len(areas) > 1:
        z = np.polyfit(areas, intensities, 1)
        p = np.poly1d(z)
        axes[2].plot(areas, p(areas), "r--", alpha=0.8, 
                    label=f'Correlation: {np.corrcoef(areas, intensities)[0,1]:.2f}')
        axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Kesimpulan dan aplikasi praktis
    print("\nKESIMPULAN DAN APLIKASI PRAKTIS")
    print("-" * 40)
    print("""
1. Edge Detection:
   - Sobel/Prewitt: Baik untuk gradient magnitude, sensitif noise
   - Canny: Robust dengan noise reduction, menghasilkan thin edges
   - Laplacian: Deteksi zero-crossing, sensitif terhadap noise

2. Region-based Methods:
   - Region Growing: Intuitif butuh seed points, sensitive to parameters
   - Split & Merge: Fully automatic, cocok untuk regular structures
   - Hybrid approaches: Kombinasi edge dan region information

3. Aplikasi Praktis:
   - Medical imaging: Organ segmentation
   - Remote sensing: Land cover classification
   - Industrial inspection: Defect detection
   - Document analysis: Text region extraction
    """)
    
    return test_image, edge_results, region_results

edge_test_image, edge_results, region_results = praktikum_9_2()