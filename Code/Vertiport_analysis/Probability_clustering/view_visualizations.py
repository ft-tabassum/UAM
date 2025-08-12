import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob

def view_visualizations():
    """View all generated visualization plots"""
    
    # Path to the centroid directory
    centroid_dir = 'D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/Centroid'
    
    # Check if directory exists
    if not os.path.exists(centroid_dir):
        print(f"Directory not found: {centroid_dir}")
        print("Please run the optimization script first.")
        return
    
    # Find all PNG files
    png_files = glob.glob(os.path.join(centroid_dir, '*.png'))
    
    if not png_files:
        print("No visualization files found.")
        return
    
    print(f"Found {len(png_files)} visualization files:")
    for i, file in enumerate(png_files):
        print(f"{i+1}. {os.path.basename(file)}")
    
    # Display each plot
    for file in png_files:
        print(f"\nDisplaying: {os.path.basename(file)}")
        img = mpimg.imread(file)
        plt.figure(figsize=(12, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title(os.path.basename(file))
        plt.show()

if __name__ == "__main__":
    view_visualizations()
