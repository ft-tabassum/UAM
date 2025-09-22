"""
UAM PTAL Accessibility Visualization
==================================

This script creates visualizations for the UAM accessibility calculation results:
- PTAL band maps showing accessibility levels across the city
- Accessibility distribution charts
- Travel time vs accessibility analysis
- Vertiport coverage analysis

Requirements:
- Results from Accessibility.py
- matplotlib, seaborn, folium for interactive maps
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Try to import folium, but don't fail if it's not available
try:
    import folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    print("Note: folium not available. Interactive maps will be skipped.")

class UAMAccessibilityVisualizer:
    def __init__(self, base_path="D:/Thesis/UAM/"):
        self.base_path = base_path
        self.results_dir = os.path.join(base_path, "Result/Vertiport_analysis/Probability_clustering/Accessibility")
        self.output_dir = os.path.join(base_path, "Result/Vertiport_analysis/Probability_clustering/PTAL_Visualizations")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("viridis")
        
    def load_results(self):
        """Load accessibility calculation results"""
        results_file = os.path.join(self.results_dir, "uam_accessibility_results.csv")
        
        if not os.path.exists(results_file):
            print(f"Results file not found: {results_file}")
            print("Please run Accessibility.py first")
            return None
            
        self.results_df = pd.read_csv(results_file)
        print(f"Loaded {len(self.results_df)} accessibility results")
        
        return self.results_df
        
    def plot_ptal_distribution(self):
        """Create PTAL band distribution chart"""
        print("Creating PTAL distribution chart...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart
        ptal_counts = self.results_df['ptal_band'].value_counts().sort_index()
        bars = ax1.bar(ptal_counts.index, ptal_counts.values, color=plt.cm.viridis(np.linspace(0, 1, len(ptal_counts))))
        ax1.set_xlabel('PTAL Band')
        ax1.set_ylabel('Number of Jobs')
        ax1.set_title('Distribution of Jobs by PTAL Band')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, ptal_counts.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count:,}', ha='center', va='bottom')
        
        # Pie chart
        ptal_percentages = (ptal_counts / ptal_counts.sum()) * 100
        colors = plt.cm.viridis(np.linspace(0, 1, len(ptal_percentages)))
        wedges, texts, autotexts = ax2.pie(ptal_percentages.values, 
                                          labels=[f'PTAL {i}' for i in ptal_percentages.index],
                                          autopct='%1.1f%%',
                                          colors=colors)
        ax2.set_title('PTAL Band Distribution (%)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'ptal_distribution.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_accessibility_scatter(self):
        """Create scatter plot of accessibility vs travel time"""
        print("Creating accessibility scatter plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # EDF vs Travel Time
        scatter = ax1.scatter(self.results_df['wt_minutes'], self.results_df['edf'], 
                             c=self.results_df['ptal_band'], cmap='viridis', alpha=0.6, s=10)
        ax1.set_xlabel('Walk/Drive Time (minutes)')
        ax1.set_ylabel('Equivalent Doorstep Frequency (EDF)')
        ax1.set_title('EDF vs Travel Time (colored by PTAL)')
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('PTAL Band')
        
        # TAT vs PTAL
        ax2.scatter(self.results_df['tat_minutes'], self.results_df['ptal_band'], 
                   alpha=0.6, s=10, color='steelblue')
        ax2.set_xlabel('Total Access Time (minutes)')
        ax2.set_ylabel('PTAL Band')
        ax2.set_title('PTAL vs Total Access Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'accessibility_scatter.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_travel_time_analysis(self):
        """Analyze travel time components"""
        print("Creating travel time analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Walk/Drive Time Distribution
        axes[0,0].hist(self.results_df['wt_minutes'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_xlabel('Walk/Drive Time (minutes)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Distribution of Walk/Drive Times')
        axes[0,0].grid(True, alpha=0.3)
        
        # Total Access Time Distribution
        axes[0,1].hist(self.results_df['tat_minutes'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0,1].set_xlabel('Total Access Time (minutes)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Distribution of Total Access Times')
        axes[0,1].grid(True, alpha=0.3)
        
        # EDF Distribution
        axes[1,0].hist(self.results_df['edf'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1,0].set_xlabel('Equivalent Doorstep Frequency (EDF)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Distribution of EDF Values')
        axes[1,0].grid(True, alpha=0.3)
        
        # Box plot of PTAL by travel time ranges
        self.results_df['travel_time_range'] = pd.cut(self.results_df['wt_minutes'], 
                                                      bins=[0, 5, 10, 15, 20, 30, 100], 
                                                      labels=['0-5min', '5-10min', '10-15min', '15-20min', '20-30min', '30+min'])
        
        ptal_by_range = []
        range_labels = []
        for range_name in self.results_df['travel_time_range'].cat.categories:
            range_data = self.results_df[self.results_df['travel_time_range'] == range_name]['ptal_band']
            if len(range_data) > 0:
                ptal_by_range.append(range_data.values)
                range_labels.append(f'{range_name}\n(n={len(range_data)})')
        
        if ptal_by_range:
            axes[1,1].boxplot(ptal_by_range, labels=range_labels)
            axes[1,1].set_ylabel('PTAL Band')
            axes[1,1].set_title('PTAL Distribution by Travel Time Range')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'travel_time_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_interactive_map(self):
        """Create interactive map showing PTAL bands"""
        if not FOLIUM_AVAILABLE:
            print("Skipping interactive map - folium not available")
            return
            
        print("Creating interactive map...")
        
        # Calculate center point
        center_lat = self.results_df['job_y'].mean()
        center_lon = self.results_df['job_x'].mean()
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Define colors for PTAL bands
        ptal_colors = {
            1: '#d73027',  # Red - worst accessibility
            2: '#f46d43',
            3: '#fdae61',
            4: '#fee08b',
            5: '#ffffbf',  # Yellow - medium accessibility
            6: '#d9ef8b',
            7: '#a6d96a',
            8: '#66c2a5',
            9: '#3288bd',
            10: '#5e4fa2'  # Purple - best accessibility
        }
        
        # Sample data for visualization (too many points for full dataset)
        sample_size = min(5000, len(self.results_df))
        sample_df = self.results_df.sample(n=sample_size, random_state=42)
        
        # Add job locations colored by PTAL
        for idx, row in sample_df.iterrows():
            folium.CircleMarker(
                location=[row['job_y'], row['job_x']],
                radius=3,
                popup=f"PTAL: {row['ptal_band']}<br>EDF: {row['edf']:.2f}<br>Travel Time: {row['wt_minutes']:.1f} min",
                color=ptal_colors.get(row['ptal_band'], 'gray'),
                fill=True,
                fillOpacity=0.7
            ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 200px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>PTAL Legend</b></p>
        '''
        for ptal, color in ptal_colors.items():
            legend_html += f'<p><i class="fa fa-circle" style="color:{color}"></i> PTAL {ptal}</p>'
        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save map
        map_file = os.path.join(self.output_dir, 'ptal_interactive_map.html')
        m.save(map_file)
        print(f"Interactive map saved to: {map_file}")
        
    def create_static_map(self):
        """Create static scatter plot map showing PTAL bands"""
        print("Creating static PTAL map...")
        
        # Define colors for PTAL bands
        ptal_colors = {
            1: '#d73027',  # Red - worst accessibility
            2: '#f46d43',
            3: '#fdae61',
            4: '#fee08b',
            5: '#ffffbf',  # Yellow - medium accessibility
            6: '#d9ef8b',
            7: '#a6d96a',
            8: '#66c2a5',
            9: '#3288bd',
            10: '#5e4fa2'  # Purple - best accessibility
        }
        
        # Sample data for visualization (too many points for full dataset)
        sample_size = min(10000, len(self.results_df))
        sample_df = self.results_df.sample(n=sample_size, random_state=42)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create scatter plot colored by PTAL
        for ptal in sorted(sample_df['ptal_band'].unique()):
            ptal_data = sample_df[sample_df['ptal_band'] == ptal]
            ax.scatter(ptal_data['job_x'], ptal_data['job_y'], 
                      c=ptal_colors[ptal], label=f'PTAL {ptal}', 
                      alpha=0.6, s=8)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('UAM Accessibility Map (PTAL Bands)', fontsize=16, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'ptal_static_map.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_summary_dashboard(self):
        """Create a comprehensive summary dashboard"""
        print("Creating summary dashboard...")
        
        fig = plt.figure(figsize=(20, 16))
        
        # Create a grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. PTAL Distribution (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        ptal_counts = self.results_df['ptal_band'].value_counts().sort_index()
        ax1.bar(ptal_counts.index, ptal_counts.values, color=plt.cm.viridis(np.linspace(0, 1, len(ptal_counts))))
        ax1.set_title('PTAL Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('PTAL Band')
        ax1.set_ylabel('Count')
        
        # 2. EDF Distribution (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(self.results_df['edf'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax2.set_title('EDF Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('EDF')
        ax2.set_ylabel('Frequency')
        
        # 3. Travel Time Distribution (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(self.results_df['wt_minutes'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        ax3.set_title('Travel Time Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Minutes')
        ax3.set_ylabel('Frequency')
        
        # 4. PTAL vs Travel Time (top far right)
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.scatter(self.results_df['wt_minutes'], self.results_df['ptal_band'], alpha=0.5, s=5)
        ax4.set_title('PTAL vs Travel Time', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Travel Time (min)')
        ax4.set_ylabel('PTAL')
        
        # 5. Summary Statistics (bottom left, spans 2 columns)
        ax5 = fig.add_subplot(gs[1, :2])
        ax5.axis('off')
        
        # Calculate summary statistics
        stats_text = f"""
        UAM Accessibility Summary Statistics
        ===================================
        
        Total Jobs Analyzed: {len(self.results_df):,}
        
        PTAL Band Statistics:
        • Mean PTAL: {self.results_df['ptal_band'].mean():.2f}
        • Median PTAL: {self.results_df['ptal_band'].median():.2f}
        • Best Accessibility (PTAL 10): {len(self.results_df[self.results_df['ptal_band'] == 10]):,} jobs
        • Worst Accessibility (PTAL 1): {len(self.results_df[self.results_df['ptal_band'] == 1]):,} jobs
        
        Travel Time Statistics:
        • Mean Travel Time: {self.results_df['wt_minutes'].mean():.2f} minutes
        • Median Travel Time: {self.results_df['wt_minutes'].median():.2f} minutes
        • Max Travel Time: {self.results_df['wt_minutes'].max():.2f} minutes
        
        EDF Statistics:
        • Mean EDF: {self.results_df['edf'].mean():.3f}
        • Median EDF: {self.results_df['edf'].median():.3f}
        • Max EDF: {self.results_df['edf'].max():.3f}
        
        UAM Service Parameters:
        • Average Waiting Time (AWT): 3.5 minutes
        • Service Headway: 4 minutes
        • Reliability Factor: 1.5 minutes
        """
        
        ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # 6. PTAL Heatmap (bottom right, spans 2 columns)
        ax6 = fig.add_subplot(gs[1, 2:])
        
        # Create a 2D histogram of PTAL vs Travel Time
        hist, xedges, yedges = np.histogram2d(self.results_df['wt_minutes'], 
                                            self.results_df['ptal_band'], 
                                            bins=[20, 10])
        im = ax6.imshow(hist.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                       origin='lower', aspect='auto', cmap='YlOrRd')
        ax6.set_xlabel('Travel Time (minutes)')
        ax6.set_ylabel('PTAL Band')
        ax6.set_title('PTAL vs Travel Time Heatmap', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax6)
        cbar.set_label('Number of Jobs')
        
        # 7. Accessibility Coverage Analysis (bottom row)
        ax7 = fig.add_subplot(gs[2, :])
        
        # Calculate accessibility coverage by distance
        distance_bins = np.arange(0, self.results_df['wt_minutes'].max() + 1, 1)
        coverage_by_distance = []
        
        for i in range(len(distance_bins) - 1):
            mask = (self.results_df['wt_minutes'] >= distance_bins[i]) & (self.results_df['wt_minutes'] < distance_bins[i+1])
            if mask.sum() > 0:
                avg_ptal = self.results_df[mask]['ptal_band'].mean()
                coverage_by_distance.append(avg_ptal)
            else:
                coverage_by_distance.append(0)
        
        distance_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
        ax7.plot(distance_centers, coverage_by_distance, 'o-', linewidth=2, markersize=4)
        ax7.set_xlabel('Travel Time (minutes)')
        ax7.set_ylabel('Average PTAL')
        ax7.set_title('Average PTAL by Travel Time Distance', fontsize=14, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        # 8. PTAL Band Details (bottom row)
        ax8 = fig.add_subplot(gs[3, :])
        
        # Create detailed PTAL analysis
        ptal_analysis = self.results_df.groupby('ptal_band').agg({
            'wt_minutes': ['count', 'mean', 'std'],
            'edf': ['mean', 'std'],
            'tat_minutes': ['mean', 'std']
        }).round(2)
        
        # Flatten column names
        ptal_analysis.columns = ['_'.join(col).strip() for col in ptal_analysis.columns]
        
        # Create table
        table_data = []
        for ptal in sorted(self.results_df['ptal_band'].unique()):
            if ptal in ptal_analysis.index:
                row_data = [
                    f"PTAL {ptal}",
                    f"{ptal_analysis.loc[ptal, 'wt_minutes_count']:,}",
                    f"{ptal_analysis.loc[ptal, 'wt_minutes_mean']:.1f}",
                    f"{ptal_analysis.loc[ptal, 'edf_mean']:.3f}",
                    f"{ptal_analysis.loc[ptal, 'tat_minutes_mean']:.1f}"
                ]
                table_data.append(row_data)
        
        table = ax8.table(cellText=table_data,
                         colLabels=['PTAL Band', 'Count', 'Avg Travel Time (min)', 'Avg EDF', 'Avg TAT (min)'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax8.axis('off')
        ax8.set_title('Detailed PTAL Analysis', fontsize=14, fontweight='bold', pad=20)
        
        # Add main title
        fig.suptitle('UAM Accessibility Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)
        
        plt.savefig(os.path.join(self.output_dir, 'accessibility_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def run_all_visualizations(self):
        """Run all visualization functions"""
        print("Starting UAM Accessibility Visualizations")
        print("=" * 50)
        
        # Load results
        if self.load_results() is None:
            return
            
        # Create visualizations
        self.plot_ptal_distribution()
        self.plot_accessibility_scatter()
        self.plot_travel_time_analysis()
        self.create_static_map()  # Always create static map
        self.create_interactive_map()  # Only if folium is available
        self.create_summary_dashboard()
        
        print("\n" + "=" * 50)
        print("All visualizations complete!")
        print(f"Results saved to: {self.output_dir}")

def main():
    """Main function to run all visualizations"""
    visualizer = UAMAccessibilityVisualizer()
    visualizer.run_all_visualizations()

if __name__ == "__main__":
    main()
