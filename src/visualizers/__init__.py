"""
Visualizers Package - Charts and Data Visualization

This package provides professional charting and visualization capabilities
for macroeconomic event impact analysis. All visualizations are designed
for publication-quality output suitable for reports, presentations, and dashboards.

Key Components:
- Plotter: Main visualization class with comprehensive charting methods
- Multi-asset return comparison charts
- Sector impact heatmaps
- Volatility analysis plots
- Time-series event impact visualizations
- Statistical significance indicators

Supported Chart Types:
- Line charts for time-series data
- Bar charts for comparative analysis
- Heatmaps for correlation matrices
- Scatter plots for statistical relationships
- Candlestick charts for price action (when applicable)

Features:
- Interactive plots with hover tooltips and zoom
- Professional styling with consistent color schemes
- Automatic chart sizing and layout optimization
- Export capabilities for multiple formats (PNG, SVG, PDF)
- Responsive design for web dashboards
- Dark/light theme support

Libraries Used:
- Matplotlib: Foundation plotting library
- Seaborn: Statistical visualization enhancements
- Plotly: Interactive web-based charts
- Pandas: Data manipulation for plotting

Classes:
    ImpactPlotter: Comprehensive plotting engine for all analysis visualizations

Usage:
    from src.visualizers.plotter import ImpactPlotter

    plotter = ImpactPlotter()
    fig = plotter.plot_multi_asset_returns(market_reactions, time_horizon='1hr')
    fig.show()  # Interactive plot

    # Export for reports
    fig.write_image("market_impact_analysis.png")
"""
