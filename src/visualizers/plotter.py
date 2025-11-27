"""
Impact Plotter - Comprehensive Visualization Tools for Macro Event Analysis

This module provides a complete visualization suite for macroeconomic event impact analysis.
The ImpactPlotter creates both static and interactive visualizations to help understand
market reactions to economic data releases.

The ImpactPlotter supports:
- Price movement charts with event markers
- Multi-asset return comparisons
- Sector performance heatmaps
- Volatility before/after analysis
- Interactive correlation matrices
- Comprehensive dashboards
- Historical trend analysis
- Professional export capabilities
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import datetime
import logging
from pathlib import Path
from typing import Optional, Dict, List
from config.settings import config

# Set plotting defaults for consistent professional appearance
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10


class ImpactPlotter:
    """
    Comprehensive visualization tool for macroeconomic event impact analysis.

    This class creates professional-quality static and interactive visualizations
    to analyze and communicate market reactions to economic data releases.
    Supports both matplotlib/seaborn for static plots and plotly for interactive dashboards.

    Features:
    - Price movement charts with event markers
    - Multi-asset return comparisons
    - Sector performance heatmaps
    - Volatility analysis (pre/post event)
    - Interactive correlation matrices
    - Comprehensive dashboards
    - Historical trend visualization
    - Professional export capabilities

    Attributes:
        export_dir: Directory for saving plots
        logger: Logger instance for operation tracking
    """

    def __init__(self, export_dir: Optional[Path] = None) -> None:
        """
        Initialize the impact plotter with export directory.

        Args:
            export_dir: Directory for saving plots (defaults to config.EXPORT_DIR / 'plots')
        """
        self.export_dir = export_dir or (config.EXPORT_DIR / 'plots')
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger('plotter')

        self.logger.info(f"✅ ImpactPlotter initialized with export directory: {self.export_dir}")

    def plot_price_movement(self, ticker: str, price_data: pd.DataFrame,
                           event_time: datetime.datetime, save_path: Optional[str] = None):
        """
        Create price movement chart with event marker and volume subplot.

        Args:
            ticker: Stock ticker symbol
            price_data: DataFrame with price data (must have 'timestamp', 'close', 'volume' columns)
            event_time: Event timestamp to mark on chart
            save_path: Optional path to save figure

        Returns:
            matplotlib.figure.Figure: The created figure object
        """
        try:
            # Create figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})

            # Top subplot: Price movement
            ax1.plot(price_data['timestamp'], price_data['close'],
                    linewidth=2, color='#2E86AB', label='Close Price')

            # Add vertical line for event time
            ax1.axvline(x=event_time, color='red', linewidth=2, linestyle='--',
                       label='Event Release', alpha=0.8)

            ax1.set_title(f'{ticker} Price Movement Around Macro Event', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Price ($)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')

            # Format x-axis dates
            if hasattr(ax1, 'tick_params'):
                ax1.tick_params(axis='x', rotation=45)

            # Bottom subplot: Volume
            ax2.bar(price_data['timestamp'], price_data['volume'],
                   color='#A23B72', alpha=0.7, width=0.01)
            ax2.set_ylabel('Volume', fontsize=12)
            ax2.set_xlabel('Time', fontsize=12)
            ax2.grid(True, alpha=0.3)

            # Format dates on bottom subplot
            if hasattr(ax2, 'tick_params'):
                ax2.tick_params(axis='x', rotation=45)

            # Overall title
            event_date_str = event_time.strftime('%Y-%m-%d %H:%M')
            fig.suptitle(f'Market Impact Analysis: {ticker} - Event at {event_date_str}',
                        fontsize=16, fontweight='bold', y=0.98)

            plt.tight_layout()

            # Save or show
            if save_path:
                self._save_figure(fig, save_path)
            else:
                plt.show()

            self.logger.info(f"✅ Created price movement plot for {ticker}")
            return fig

        except Exception as e:
            self.logger.error(f"Failed to create price movement plot for {ticker}: {e}")
            plt.close('all')  # Clean up any partial figures
            raise

        finally:
            plt.close(fig)  # Always close to free memory

    def plot_multi_asset_returns(self, market_reactions: pd.DataFrame,
                               time_horizon: str = '1hr', save_path: Optional[str] = None):
        """
        Create horizontal bar chart of asset returns for a specific time horizon.

        Args:
            market_reactions: DataFrame with market reactions
            time_horizon: Time window to plot (e.g., '15 Minutes', '1hr', '1 Day')
            save_path: Optional path to save figure

        Returns:
            matplotlib.figure.Figure: The created figure object
        """
        try:
            if market_reactions.empty or time_horizon not in market_reactions.columns:
                raise ValueError(f"Time horizon '{time_horizon}' not found in market reactions data")

            # Prepare data
            data = market_reactions.dropna(subset=[time_horizon]).copy()
            data = data.sort_values(time_horizon, ascending=True)  # Sort for horizontal bars

            # Create colors based on return sign
            colors = ['#4CAF50' if x >= 0 else '#F44336' for x in data[time_horizon]]

            # Create figure
            fig, ax = plt.subplots(figsize=(12, max(8, len(data) * 0.4)))

            # Create horizontal bars
            bars = ax.barh(data['ticker'], data[time_horizon], color=colors, alpha=0.8)

            # Add value labels on bars
            for bar, value in zip(bars, data[time_horizon]):
                width = bar.get_width()
                label_x = width + (0.01 * abs(width)) if width >= 0 else width - (0.01 * abs(width))
                ax.text(label_x, bar.get_y() + bar.get_height()/2,
                       self._format_percentage(value),
                       ha='left' if width >= 0 else 'right',
                       va='center', fontweight='bold', fontsize=9)

            # Formatting
            ax.set_title(f'Asset Returns {time_horizon} After Release',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Return (%)', fontsize=12)
            ax.set_ylabel('Asset', fontsize=12)
            ax.grid(True, alpha=0.3, axis='x')
            ax.axvline(x=0, color='black', linewidth=1, alpha=0.5)

            # Add statistics annotation
            positive_pct = (data[time_horizon] > 0).mean() * 100
            avg_return = data[time_horizon].mean()

            stats_text = f'Assets: {len(data)} | Positive: {positive_pct:.1f}% | Avg Return: {self._format_percentage(avg_return)}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plt.tight_layout()

            # Save or show
            if save_path:
                self._save_figure(fig, save_path)
            else:
                plt.show()

            self.logger.info(f"✅ Created multi-asset returns plot for {time_horizon}")
            return fig

        except Exception as e:
            self.logger.error(f"Failed to create multi-asset returns plot: {e}")
            plt.close('all')
            raise

        finally:
            plt.close(fig)

    def plot_sector_heatmap(self, sector_analysis: pd.DataFrame, save_path: Optional[str] = None):
        """
        Create sector performance heatmap across time horizons.

        Args:
            sector_analysis: DataFrame from ImpactAnalyzer.analyze_by_sector()
            save_path: Optional path to save figure

        Returns:
            matplotlib.figure.Figure: The created figure object
        """
        try:
            if sector_analysis.empty:
                raise ValueError("Empty sector analysis data provided")

            # Extract return columns (columns containing 'mean')
            return_cols = [col for col in sector_analysis.columns if 'mean' in col]
            if not return_cols:
                raise ValueError("No return columns found in sector analysis")

            # Prepare data for heatmap
            heatmap_data = sector_analysis.set_index('sector')[return_cols]

            # Clean column names for display
            display_cols = [col.replace('_mean', '').replace(' Minutes', 'm').replace(' Hour', 'h')
                          for col in return_cols]

            # Create figure
            fig, ax = plt.subplots(figsize=(12, max(6, len(heatmap_data) * 0.8)))

            # Create heatmap
            sns.heatmap(heatmap_data,
                       annot=True,
                       fmt='.2f',
                       cmap='RdYlGn',
                       center=0,
                       cbar_kws={'label': 'Return (%)', 'shrink': 0.8},
                       linewidths=0.5,
                       ax=ax)

            # Formatting
            ax.set_title('Sector Performance Across Time Horizons',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Time After Release', fontsize=12)
            ax.set_ylabel('Sector', fontsize=12)

            # Update column labels
            ax.set_xticklabels(display_cols, rotation=45, ha='right')

            plt.tight_layout()

            # Save or show
            if save_path:
                self._save_figure(fig, save_path)
            else:
                plt.show()

            self.logger.info("✅ Created sector performance heatmap")
            return fig

        except Exception as e:
            self.logger.error(f"Failed to create sector heatmap: {e}")
            plt.close('all')
            raise

        finally:
            plt.close(fig)

    def plot_volatility_comparison(self, volatility_analysis: pd.DataFrame,
                                 save_path: Optional[str] = None):
        """
        Create grouped bar chart comparing pre-event and post-event volatility.

        Args:
            volatility_analysis: DataFrame from ImpactAnalyzer with volatility data
            save_path: Optional path to save figure

        Returns:
            matplotlib.figure.Figure: The created figure object
        """
        try:
            if volatility_analysis.empty:
                raise ValueError("Empty volatility analysis data provided")

            required_cols = ['ticker', 'pre_event_vol', 'post_event_vol']
            missing_cols = [col for col in required_cols if col not in volatility_analysis.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Prepare data
            data = volatility_analysis.dropna(subset=['pre_event_vol', 'post_event_vol']).copy()
            data = data.sort_values('post_event_vol', ascending=False)  # Sort by post-event vol

            # Set up positions for grouped bars
            x = np.arange(len(data))
            width = 0.35

            # Create figure
            fig, ax = plt.subplots(figsize=(14, max(8, len(data) * 0.5)))

            # Create grouped bars
            bars1 = ax.bar(x - width/2, data['pre_event_vol'],
                          width, label='Pre-Event', color='#2196F3', alpha=0.8)
            bars2 = ax.bar(x + width/2, data['post_event_vol'],
                          width, label='Post-Event', color='#FF9800', alpha=0.8)

            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.1f}%', ha='center', va='bottom',
                           fontweight='bold', fontsize=8)

            # Formatting
            ax.set_title('Volatility Before vs After Macro Release',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Asset', fontsize=12)
            ax.set_ylabel('Annualized Volatility (%)', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(data['ticker'], rotation=45, ha='right')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')

            # Add summary statistics
            avg_pre = data['pre_event_vol'].mean()
            avg_post = data['post_event_vol'].mean()
            avg_change = data['vol_change_pct'].mean()

            stats_text = f'Avg Pre: {avg_pre:.1f}% | Avg Post: {avg_post:.1f}% | Avg Change: {self._format_percentage(avg_change)}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

            plt.tight_layout()

            # Save or show
            if save_path:
                self._save_figure(fig, save_path)
            else:
                plt.show()

            self.logger.info("✅ Created volatility comparison plot")
            return fig

        except Exception as e:
            self.logger.error(f"Failed to create volatility comparison plot: {e}")
            plt.close('all')
            raise

        finally:
            plt.close(fig)

    def plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame,
                               save_path: Optional[str] = None):
        """
        Create interactive correlation heatmap using Plotly.

        Args:
            correlation_matrix: Correlation matrix DataFrame
            save_path: Optional path to save as HTML (shows in browser if None)

        Returns:
            plotly.graph_objects.Figure: The created figure object
        """
        try:
            if correlation_matrix.empty:
                raise ValueError("Empty correlation matrix provided")

            # Create hover text with formatted correlation values
            hover_text = []
            for i in range(len(correlation_matrix.index)):
                row_text = []
                for j in range(len(correlation_matrix.columns)):
                    value = correlation_matrix.iloc[i, j]
                    row_text.append(f"{correlation_matrix.columns[j]}: {value:.3f}")
                hover_text.append(row_text)

            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdBu',
                zmid=0,
                hoverongaps=False,
                hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>',
                colorbar=dict(title="Correlation", titleside="right")
            ))

            # Update layout
            fig.update_layout(
                title={
                    'text': 'Cross-Asset Correlation During Event Window',
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=16, family='Arial, sans-serif')
                },
                xaxis_title="Asset",
                yaxis_title="Asset",
                width=800,
                height=700,
                font=dict(size=10, family='Arial, sans-serif')
            )

            # Save or show
            if save_path:
                self._save_figure(fig, save_path)
            else:
                fig.show()

            self.logger.info("✅ Created interactive correlation heatmap")
            return fig

        except Exception as e:
            self.logger.error(f"Failed to create correlation heatmap: {e}")
            raise

    def plot_returns_timeline(self, analysis_results: Dict, save_path: Optional[str] = None):
        """
        Create line plot showing return evolution across time horizons.

        Args:
            analysis_results: Complete analysis results from ImpactAnalyzer
            save_path: Optional path to save figure

        Returns:
            matplotlib.figure.Figure: The created figure object
        """
        try:
            market_reactions = analysis_results.get('market_reactions', pd.DataFrame())
            if market_reactions.empty:
                raise ValueError("No market reactions data in analysis results")

            # Get time windows from config
            time_windows = [w['label'] for w in config.ANALYSIS_WINDOWS]
            available_windows = [w for w in time_windows if w in market_reactions.columns]

            if len(available_windows) < 2:
                raise ValueError("Need at least 2 time windows for timeline plot")

            # Get color palette
            colors = self._get_color_palette(len(market_reactions))

            # Create figure
            fig, ax = plt.subplots(figsize=(14, 8))

            # Plot each asset's return evolution
            for i, (_, asset_data) in enumerate(market_reactions.iterrows()):
                ticker = asset_data['ticker']
                returns = [asset_data.get(window, 0) for window in available_windows]

                ax.plot(available_windows, returns,
                       marker='o', linewidth=2, markersize=6,
                       color=colors[i % len(colors)],
                       label=ticker, alpha=0.8)

            # Formatting
            ax.set_title('Return Evolution Across Time Horizons',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Time After Release', fontsize=12)
            ax.set_ylabel('Return (%)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linewidth=1, alpha=0.5, linestyle='--')

            # Clean up x-axis labels
            display_labels = [label.replace(' Minutes', 'm').replace(' Hour', 'h').replace(' Day', 'd')
                            for label in available_windows]
            ax.set_xticks(range(len(available_windows)))
            ax.set_xticklabels(display_labels, rotation=45, ha='right')

            # Add legend with multiple columns if many assets
            ncol = min(3, max(1, len(market_reactions) // 8))
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                     fontsize=9, ncol=ncol)

            # Add summary statistics
            final_returns = market_reactions[available_windows[-1]].dropna()
            if len(final_returns) > 0:
                pos_pct = (final_returns > 0).mean() * 100
                avg_final = final_returns.mean()

                stats_text = f'Final Period: {pos_pct:.1f}% positive | Avg Return: {self._format_percentage(avg_final)}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

            plt.tight_layout()

            # Save or show
            if save_path:
                self._save_figure(fig, save_path)
            else:
                plt.show()

            self.logger.info("✅ Created returns timeline plot")
            return fig

        except Exception as e:
            self.logger.error(f"Failed to create returns timeline plot: {e}")
            plt.close('all')
            raise

        finally:
            plt.close(fig)

    def create_dashboard(self, analysis_results: Dict, save_path: Optional[str] = None):
        """
        Create comprehensive interactive dashboard with multiple panels.

        Args:
            analysis_results: Complete analysis results from ImpactAnalyzer
            save_path: Optional path to save as HTML (shows in browser if None)

        Returns:
            plotly.graph_objects.Figure: The created dashboard figure
        """
        try:
            # Create subplot layout
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Top Asset Returns (15min)',
                    'Sector Performance Heatmap',
                    'Volatility Change',
                    'Key Statistics'
                ),
                specs=[
                    [{"type": "bar"}, {"type": "heatmap"}],
                    [{"type": "bar"}, {"type": "table"}]
                ]
            )

            # Panel 1: Top asset returns (horizontal bar chart)
            market_reactions = analysis_results.get('market_reactions', pd.DataFrame())
            if not market_reactions.empty and '15 Minutes' in market_reactions.columns:
                data_15min = market_reactions.dropna(subset=['15 Minutes']).copy()
                data_15min = data_15min.nlargest(10, '15 Minutes')  # Top 10

                colors = ['#4CAF50' if x >= 0 else '#F44336' for x in data_15min['15 Minutes']]

                fig.add_trace(
                    go.Bar(
                        x=data_15min['15 Minutes'],
                        y=data_15min['ticker'],
                        orientation='h',
                        marker_color=colors,
                        name='15min Returns'
                    ),
                    row=1, col=1
                )

            # Panel 2: Sector heatmap
            sector_summary = analysis_results.get('sector_summary', pd.DataFrame())
            if not sector_summary.empty:
                # Get return columns
                return_cols = [col for col in sector_summary.columns if '_mean' in col]
                if return_cols:
                    heatmap_data = sector_summary[return_cols].values
                    sectors = sector_summary['sector'].tolist()
                    time_labels = [col.replace('_mean', '').replace(' Minutes', 'm').replace(' Hour', 'h')
                                 for col in return_cols]

                    fig.add_trace(
                        go.Heatmap(
                            z=heatmap_data,
                            x=time_labels,
                            y=sectors,
                            colorscale='RdYlGn',
                            zmid=0,
                            name='Sector Returns'
                        ),
                        row=1, col=2
                    )

            # Panel 3: Volatility change
            volatility_df = analysis_results.get('volatility_analysis', pd.DataFrame())
            if not volatility_df.empty and len(volatility_df) > 0:
                vol_data = volatility_df.head(10)  # Top 10 by post-event volatility

                fig.add_trace(
                    go.Bar(
                        name='Pre-Event',
                        x=vol_data['ticker'],
                        y=vol_data['pre_event_vol'],
                        marker_color='#2196F3'
                    ),
                    row=2, col=1
                )

                fig.add_trace(
                    go.Bar(
                        name='Post-Event',
                        x=vol_data['ticker'],
                        y=vol_data['post_event_vol'],
                        marker_color='#FF9800'
                    ),
                    row=2, col=1
                )

            # Panel 4: Key statistics table
            event_info = analysis_results.get('event_info', {})
            statistical_tests = analysis_results.get('statistical_tests', {})

            table_data = [
                ['Indicator', event_info.get('indicator', 'N/A')],
                ['Release Date', str(event_info.get('date', 'N/A'))],
                ['Value', str(event_info.get('value', 'N/A'))],
                ['Change', self._format_percentage(event_info.get('change', 0))],
                ['Data Type', event_info.get('data_type', 'N/A')],
                ['Assets Analyzed', str(analysis_results.get('assets_analyzed', 0))],
            ]

            # Add statistical significance for 15min if available
            if '15 Minutes' in statistical_tests:
                test_15min = statistical_tests['15 Minutes']
                table_data.extend([
                    ['15min Significant', 'Yes' if test_15min.get('significant', False) else 'No'],
                    ['15min P-Value', f"{test_15min.get('p_value', 'N/A'):.4f}"],
                    ['15min Sample Size', str(test_15min.get('sample_size', 'N/A'))]
                ])

            fig.add_trace(
                go.Table(
                    columnwidth=[150, 100],
                    header=dict(
                        values=['<b>Metric</b>', '<b>Value</b>'],
                        fill_color='#4472C4',
                        font=dict(color='white', size=12),
                        align='left'
                    ),
                    cells=dict(
                        values=list(zip(*table_data)),
                        fill_color=[['#E6F3FF', '#FFFFFF'] * len(table_data)],
                        align='left',
                        font_size=11,
                        height=25
                    )
                ),
                row=2, col=2
            )

            # Update layout
            event_title = f"Macro Event Impact Dashboard: {event_info.get('indicator', 'Unknown')} on {event_info.get('date', 'Unknown')}"
            fig.update_layout(
                title={
                    'text': event_title,
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=16, family='Arial, sans-serif')
                },
                showlegend=True,
                width=1200,
                height=900,
                font=dict(family='Arial, sans-serif')
            )

            # Save or show
            if save_path:
                self._save_figure(fig, save_path)
            else:
                fig.show()

            self.logger.info("✅ Created comprehensive interactive dashboard with 4 panels")
            return fig

        except Exception as e:
            self.logger.error(f"Failed to create dashboard: {e}")
            raise

    def plot_historical_trends(self, historical_df: pd.DataFrame, indicator_code: str,
                             save_path: Optional[str] = None):
        """
        Plot historical market response trends for an indicator.

        Args:
            historical_df: DataFrame from ImpactAnalyzer.batch_analyze_historical()
            indicator_code: Indicator code for plot title
            save_path: Optional path to save figure

        Returns:
            matplotlib.figure.Figure: The created figure object
        """
        try:
            if historical_df.empty:
                raise ValueError("Empty historical data provided")

            # Find return columns
            return_cols = [col for col in historical_df.columns
                          if 'avg_return' in col and not col.startswith('avg_15min')]

            if not return_cols:
                raise ValueError("No return columns found in historical data")

            # Use the first available return column
            return_col = return_cols[0]
            time_horizon = return_col.replace('avg_', '').replace('_avg_return', '').replace('_', ' ')

            # Prepare data
            plot_data = historical_df.dropna(subset=['release_date', return_col]).copy()
            plot_data = plot_data.sort_values('release_date')

            # Create figure
            fig, ax = plt.subplots(figsize=(14, 8))

            # Plot individual events
            ax.scatter(plot_data['release_date'], plot_data[return_col],
                      color='#2196F3', alpha=0.6, s=50, label='Individual Events')

            # Add trend line if enough data points
            if len(plot_data) >= 3:
                z = np.polyfit(range(len(plot_data)), plot_data[return_col], 1)
                p = np.poly1d(z)
                ax.plot(plot_data['release_date'], p(range(len(plot_data))),
                       color='#FF5722', linewidth=2, label='Trend Line')

            # Add rolling average if enough data
            if len(plot_data) >= 10:
                rolling_avg = plot_data[return_col].rolling(window=5, center=True).mean()
                ax.plot(plot_data['release_date'], rolling_avg,
                       color='#4CAF50', linewidth=2, linestyle='--', label='5-Event Rolling Avg')

            # Formatting
            ax.set_title(f'Historical Market Response to {indicator_code} Releases\n({time_horizon})',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Release Date', fontsize=12)
            ax.set_ylabel('Average Return (%)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linewidth=1, alpha=0.5, linestyle='--')
            ax.legend(loc='upper right')

            # Format dates
            if hasattr(ax, 'tick_params'):
                ax.tick_params(axis='x', rotation=45)

            # Add summary statistics
            if len(plot_data) > 0:
                avg_return = plot_data[return_col].mean()
                pos_pct = (plot_data[return_col] > 0).mean() * 100

                stats_text = f'Events: {len(plot_data)} | Avg Return: {self._format_percentage(avg_return)} | Positive: {pos_pct:.1f}%'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

            plt.tight_layout()

            # Save or show
            if save_path:
                self._save_figure(fig, save_path)
            else:
                plt.show()

            self.logger.info(f"✅ Created historical trends plot for {indicator_code}")
            return fig

        except Exception as e:
            self.logger.error(f"Failed to create historical trends plot: {e}")
            plt.close('all')
            raise

        finally:
            plt.close(fig)

    def generate_report_plots(self, analysis_results: Dict, output_dir: Optional[Path] = None) -> List[Path]:
        """
        Generate all standard plots for a single event analysis.

        Args:
            analysis_results: Complete analysis results from ImpactAnalyzer
            output_dir: Directory to save plots (defaults to self.export_dir)

        Returns:
            List[Path]: List of paths to saved plot files
        """
        output_dir = output_dir or self.export_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_plots = []
        event_info = analysis_results.get('event_info', {})
        indicator = event_info.get('indicator', 'unknown')
        date_str = event_info.get('date', datetime.datetime.now()).strftime('%Y%m%d_%H%M')

        try:
            # 1. Multi-asset returns plot
            market_reactions = analysis_results.get('market_reactions', pd.DataFrame())
            if not market_reactions.empty and '15 Minutes' in market_reactions.columns:
                plot_path = output_dir / f"{indicator}_{date_str}_asset_returns.png"
                self.plot_multi_asset_returns(market_reactions, '15 Minutes', str(plot_path))
                saved_plots.append(plot_path)

            # 2. Sector heatmap
            sector_summary = analysis_results.get('sector_summary', pd.DataFrame())
            if not sector_summary.empty:
                plot_path = output_dir / f"{indicator}_{date_str}_sector_heatmap.png"
                self.plot_sector_heatmap(sector_summary, str(plot_path))
                saved_plots.append(plot_path)

            # 3. Volatility comparison
            volatility_df = analysis_results.get('volatility_analysis', pd.DataFrame())
            if not volatility_df.empty:
                plot_path = output_dir / f"{indicator}_{date_str}_volatility_comparison.png"
                self.plot_volatility_comparison(volatility_df, str(plot_path))
                saved_plots.append(plot_path)

            # 4. Interactive dashboard
            plot_path = output_dir / f"{indicator}_{date_str}_dashboard.html"
            self.create_dashboard(analysis_results, str(plot_path))
            saved_plots.append(plot_path)

            self.logger.info(f"✅ Generated {len(saved_plots)} plots in {output_dir}")
            return saved_plots

        except Exception as e:
            self.logger.error(f"Failed to generate report plots: {e}")
            return saved_plots  # Return any successfully created plots

    def _format_percentage(self, value: float) -> str:
        """
        Format a decimal value as a percentage string with sign.

        Args:
            value: Decimal value to format

        Returns:
            str: Formatted percentage string
        """
        if value is None:
            return "0.00%"

        try:
            formatted = ".2f"
            return f"+{formatted}%" if value >= 0 else f"{formatted}%"
        except (ValueError, TypeError):
            return "0.00%"

    def _save_figure(self, fig, save_path: str) -> None:
        """
        Save figure to file with appropriate format.

        Args:
            fig: Figure object (matplotlib or plotly)
            save_path: Path to save figure
        """
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            if hasattr(fig, 'write_html'):  # Plotly figure
                fig.write_html(str(save_path))
            else:  # Matplotlib figure
                fig.savefig(str(save_path), bbox_inches='tight', dpi=300, facecolor='white')

            self.logger.debug(f"Saved plot to: {save_path}")

        except Exception as e:
            self.logger.error(f"Failed to save figure to {save_path}: {e}")
            raise

    def _get_color_palette(self, n_colors: int) -> List[str]:
        """
        Get a professional color palette for plotting.

        Args:
            n_colors: Number of colors needed

        Returns:
            List[str]: List of hex color codes
        """
        # Use seaborn's professional color palette
        try:
            palette = sns.color_palette("husl", n_colors)
            return [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for r, g, b in palette]
        except Exception:
            # Fallback to basic colors
            basic_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            return (basic_colors * ((n_colors // len(basic_colors)) + 1))[:n_colors]
