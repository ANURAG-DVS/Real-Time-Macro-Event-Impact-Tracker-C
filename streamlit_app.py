"""
Streamlit Web Dashboard - Macro Event Impact Tracker

A comprehensive web-based dashboard for analyzing how macroeconomic events
impact financial markets. 100% FREE to deploy on Streamlit Cloud.

Features:
- Interactive analysis of economic releases
- Multi-asset market impact visualization
- Sector-level performance analysis
- Volatility tracking before/after events
- Historical trend analysis
- Professional UI with real-time updates

Data Sources (100% FREE):
- FRED API: Economic indicators (requires free API key)
- Yahoo Finance: Market data (no API key needed)
"""

import streamlit as st
import pandas as pd
import datetime
import logging
from src.data_fetchers.fred_fetcher import FREDDataFetcher
from src.data_fetchers.yahoo_fetcher import YahooDataFetcher
from src.analyzers.impact_analyzer import ImpactAnalyzer
from src.visualizers.plotter import ImpactPlotter
from config.settings import config

# Page configuration
st.set_page_config(
    page_title="Macro Event Impact Tracker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': '100% FREE tool for tracking macro event impacts on markets'
    }
)

# Custom CSS for professional styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    color: #1f77b4;
    margin-bottom: 1rem;
}
.sub-header {
    text-align: center;
    color: #666;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.success-text {
    color: #28a745;
    font-weight: bold;
}
.error-text {
    color: #dc3545;
    font-weight: bold;
}
.warning-text {
    color: #ffc107;
    font-weight: bold;
}
.info-box {
    background-color: #e3f2fd;
    padding: 1rem;
    border-left: 4px solid #2196f3;
    border-radius: 0.25rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource
def initialize_components():
    """Initialize data fetchers and analyzer (cached for performance)"""
    try:
        fred_fetcher = FREDDataFetcher()
        yahoo_fetcher = YahooDataFetcher()
        analyzer = ImpactAnalyzer(fred_fetcher, yahoo_fetcher)
        plotter = ImpactPlotter()
        return fred_fetcher, yahoo_fetcher, analyzer, plotter
    except Exception as e:
        st.error(f"❌ Failed to initialize components: {e}")
        st.stop()


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_analysis(indicator_code, release_date, hours_after):
    """Run analysis and cache results"""
    try:
        fred_fetcher, yahoo_fetcher, analyzer, _ = initialize_components()
        results = analyzer.analyze_single_event(indicator_code, release_date, hours_after)
        return results
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise e


def main():
    """Main Streamlit application"""

    # Check configuration first
    try:
        config.validate_config()
    except ValueError as e:
        st.error(f"❌ Configuration Error: {str(e)}")
        st.info("""
        **Setup Instructions:**
        1. Get a FREE FRED API key at: https://fred.stlouisfed.org/docs/api/api_key.html
        2. Create a `.env` file in the project root
        3. Add: `FRED_API_KEY=your_key_here`
        4. Restart the Streamlit app
        """)
        st.stop()

    # Header
    st.markdown('<h1 class="main-header">📊 Macro Event Impact Tracker</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze how economic releases impact financial markets • 100% FREE</p>', unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Indicator selection
        indicator_options = list(config.MACRO_INDICATORS.keys())
        indicator_labels = [f"{k} - {v['name']}" for k, v in config.MACRO_INDICATORS.items()]
        selected_indicator = st.selectbox(
            "Economic Indicator",
            indicator_options,
            format_func=lambda x: f"{x} - {config.MACRO_INDICATORS[x]['name']}",
            help="Choose the economic indicator to analyze"
        )

        # Date selection
        use_latest = st.checkbox("Use Latest Release", value=True,
                                 help="Automatically use the most recent data release")

        release_date = None
        if not use_latest:
            release_date_input = st.date_input(
                "Release Date",
                value=datetime.datetime.now().date(),
                help="Specific date of the economic release"
            )
            release_date = datetime.datetime.combine(release_date_input, datetime.datetime.min.time())

        # Analysis window
        hours_after = st.slider(
            "Hours After Release",
            min_value=1,
            max_value=24,
            value=4,
            help="How many hours after the release to analyze"
        )

        # Asset selection
        st.subheader("Assets to Analyze")
        asset_options = list(config.MARKET_ASSETS.keys())
        selected_assets = st.multiselect(
            "Select Assets",
            asset_options,
            default=asset_options[:5],  # Default to first 5
            help="Choose which assets to include in analysis"
        )

        # Action button
        st.markdown("---")
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

        # Data age warning for older dates
        if not use_latest and release_date:
            days_ago = (datetime.datetime.now() - release_date).days
            if days_ago > 7:
                st.warning("⚠️ Note: Events older than 7 days will use daily data (not intraday) due to Yahoo Finance free tier limitations.")

        # About section
        with st.expander("ℹ️ About This Tool"):
            st.markdown("""
            **Data Sources (100% FREE):**
            - 📈 FRED API: Economic indicators
            - 💹 Yahoo Finance: Market data

            **Perfect for:**
            - Finance students building portfolios
            - Job seekers showcasing skills
            - Analysts exploring macro relationships

            **No cost, no credit card required!**
            """)

    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Event Analysis",
        "📊 Multi-Asset Comparison",
        "🎯 Sector Analysis",
        "📈 Volatility Analysis",
        "🕐 Historical Trends"
    ])

    # Tab 1 - Event Analysis
    with tab1:
        if run_analysis:
            with st.spinner("🔍 Fetching data and analyzing... This may take 30-60 seconds."):
                try:
                    # Run analysis
                    results = load_analysis(selected_indicator, release_date, hours_after)

                    # Display event info
                    st.subheader("📅 Event Information")
                    col1, col2, col3, col4 = st.columns(4)

                    event_info = results['event_info']
                    with col1:
                        st.metric("Indicator", event_info['indicator'])
                    with col2:
                        st.metric("Release Date", event_info['date'].strftime('%Y-%m-%d'))
                    with col3:
                        st.metric("Value", f"{event_info['value']:.2f}")
                    with col4:
                        change = event_info['change']
                        st.metric("Change", f"{change:+.2f}%", delta=f"{change:+.2f}%")

                    st.markdown("---")

                    # Display market reactions
                    st.subheader("💹 Market Reactions")

                    # Filter for selected assets
                    market_reactions = results['market_reactions']
                    filtered_reactions = market_reactions[
                        market_reactions['ticker'].isin(selected_assets)
                    ]

                    if not filtered_reactions.empty:
                        # Style the dataframe
                        def color_returns(val):
                            if isinstance(val, (int, float)) and not pd.isna(val):
                                color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                                return f'color: {color}'
                            return ''

                        # Apply styling to return columns
                        return_cols = [col for col in filtered_reactions.columns
                                     if any(x in col.lower() for x in ['15', '30', '1hr', '2hr', '4hr', '1day'])]
                        styled_df = filtered_reactions.style.applymap(
                            color_returns, subset=return_cols
                        ).format("{:.2f}", subset=return_cols)

                        st.dataframe(styled_df, use_container_width=True)

                        # Download button
                        csv = filtered_reactions.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"market_reactions_{selected_indicator}_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No market reaction data available for selected assets")

                    # Summary text
                    with st.expander("📝 Detailed Impact Report"):
                        summary = results.get('summary_text')
                        if summary:
                            st.text(summary)
                        else:
                            # Generate summary if not available
                            try:
                                fred_fetcher, yahoo_fetcher, analyzer, _ = initialize_components()
                                summary = analyzer.generate_impact_summary(results)
                                st.text(summary)
                            except Exception as e:
                                st.text(f"Summary generation failed: {e}")

                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.info("""
                    💡 Troubleshooting tips:
                    - Check your internet connection
                    - Verify FRED API key is set correctly
                    - Try a different date or indicator
                    - For events >7 days old, only daily data is available
                    """)
        else:
            st.info("👈 Configure your analysis in the sidebar and click 'Run Analysis'")

    # Tab 2 - Multi-Asset Comparison
    with tab2:
        if run_analysis and 'results' in locals() and 'results' in globals():
            try:
                st.subheader("📊 Return Comparison Across Time Horizons")

                market_reactions = results['market_reactions']
                filtered_data = market_reactions[market_reactions['ticker'].isin(selected_assets)]

                if not filtered_data.empty:
                    # Time horizon selector
                    time_columns = [col for col in filtered_data.columns
                                  if any(x in col.lower() for x in ['15', '30', '1hr', '2hr', '4hr', '1day'])]
                    selected_horizon = st.selectbox("Select Time Horizon", time_columns)

                    # Create bar chart
                    chart_data = filtered_data[['ticker', selected_horizon]].sort_values(selected_horizon)

                    import plotly.graph_objects as go
                    fig = go.Figure(data=[
                        go.Bar(
                            x=chart_data[selected_horizon],
                            y=chart_data['ticker'],
                            orientation='h',
                            marker_color=['#4CAF50' if x > 0 else '#F44336' for x in chart_data[selected_horizon]],
                            hovertemplate='%{y}: %{x:.2f}%<extra></extra>'
                        )
                    ])
                    fig.update_layout(
                        title=f"Asset Returns: {selected_horizon}",
                        xaxis_title="Return (%)",
                        yaxis_title="Asset",
                        height=max(400, len(chart_data) * 30)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Top movers
                    st.subheader("🏆 Top Movers")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Top Gainers**")
                        top_gainers = chart_data.nlargest(3, selected_horizon)
                        for _, row in top_gainers.iterrows():
                            st.markdown(f"<span class='success-text'>• {row['ticker']}: +{row[selected_horizon]:.2f}%</span>", unsafe_allow_html=True)

                    with col2:
                        st.markdown("**Top Losers**")
                        top_losers = chart_data.nsmallest(3, selected_horizon)
                        for _, row in top_losers.iterrows():
                            st.markdown(f"<span class='error-text'>• {row['ticker']}: {row[selected_horizon]:.2f}%</span>", unsafe_allow_html=True)

                else:
                    st.warning("No data available for selected assets")
            except Exception as e:
                st.error(f"Error in multi-asset comparison: {e}")
        else:
            st.info("Run analysis first to see multi-asset comparison")

    # Tab 3 - Sector Analysis
    with tab3:
        if run_analysis and 'results' in locals() and 'results' in globals():
            try:
                st.subheader("🎯 Sector-Level Performance")

                sector_data = results.get('sector_summary')
                if sector_data is not None and not sector_data.empty:
                    # Create heatmap
                    import plotly.graph_objects as go

                    # Get return columns
                    return_cols = [col for col in sector_data.columns if '_mean' in col]
                    if return_cols:
                        heatmap_data = sector_data[return_cols].values
                        sectors = sector_data.index.tolist()
                        time_labels = [col.replace('_mean', '').replace(' Minutes', 'm').replace(' Hour', 'h')
                                     for col in return_cols]

                        fig = go.Figure(data=go.Heatmap(
                            z=heatmap_data,
                            x=time_labels,
                            y=sectors,
                            colorscale='RdYlGn',
                            zmid=0,
                            text=[[f'{val:.2f}%' for val in row] for row in heatmap_data],
                            texttemplate='%{text}',
                            textfont={"size": 10},
                            hoverongaps=False,
                            hovertemplate='%{y} at %{x}: %{z:.2f}%<extra></extra>'
                        ))
                        fig.update_layout(
                            title="Sector Returns Across Time Horizons",
                            xaxis_title="Time After Release",
                            yaxis_title="Sector",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Sector insights
                        st.subheader("📌 Key Insights")
                        insights = []
                        for sector in sector_data.index:
                            avg_return = sector_data.loc[sector, return_cols].mean()
                            if abs(avg_return) > 0.5:  # Only show significant movements
                                direction = "gained" if avg_return > 0 else "lost"
                                insights.append(f"• **{sector.capitalize()}** {direction} an average of **{abs(avg_return):.2f}%** across all time windows")

                        if insights:
                            for insight in insights:
                                st.write(insight)
                        else:
                            st.write("• No significant sector movements detected")
                    else:
                        st.warning("No return data available for sector analysis")
                else:
                    st.warning("Sector analysis data not available")
            except Exception as e:
                st.error(f"Error in sector analysis: {e}")
        else:
            st.info("Run analysis first to see sector analysis")

    # Tab 4 - Volatility Analysis
    with tab4:
        if run_analysis and 'results' in locals() and 'results' in globals():
            try:
                st.subheader("📈 Volatility Changes")

                vol_data = results.get('volatility_analysis')
                if vol_data is not None and not vol_data.empty:
                    # Filter for selected assets
                    vol_data_filtered = vol_data[vol_data['ticker'].isin(selected_assets)]

                    if not vol_data_filtered.empty:
                        # Create grouped bar chart
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            name='Pre-Event',
                            x=vol_data_filtered['ticker'],
                            y=vol_data_filtered['pre_event_vol'],
                            marker_color='#2196F3',
                            hovertemplate='%{x}: %{y:.1f}%<extra>Pre-Event Volatility</extra>'
                        ))
                        fig.add_trace(go.Bar(
                            name='Post-Event',
                            x=vol_data_filtered['ticker'],
                            y=vol_data_filtered['post_event_vol'],
                            marker_color='#FF9800',
                            hovertemplate='%{x}: %{y:.1f}%<extra>Post-Event Volatility</extra>'
                        ))
                        fig.update_layout(
                            title="Volatility Before vs After Release",
                            xaxis_title="Asset",
                            yaxis_title="Annualized Volatility (%)",
                            barmode='group',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Volatility changes table
                        st.subheader("📊 Volatility Changes Detail")

                        # Add change indicators
                        vol_display = vol_data_filtered.copy()
                        vol_display['change_pct'] = vol_display['vol_change_pct'].apply(lambda x: f"{x:+.1f}%")
                        vol_display['direction'] = vol_display['vol_change'].apply(
                            lambda x: "↗️ Increased" if x > 0 else "↘️ Decreased" if x < 0 else "➡️ Unchanged"
                        )

                        display_cols = ['ticker', 'pre_event_vol', 'post_event_vol', 'change_pct', 'direction']
                        display_names = ['Asset', 'Pre-Event Vol (%)', 'Post-Event Vol (%)', 'Change', 'Direction']

                        st.dataframe(
                            vol_display[display_cols],
                            column_config=dict(zip(display_cols, display_names)),
                            use_container_width=True
                        )
                    else:
                        st.warning("No volatility data available for selected assets")
                else:
                    st.warning("Volatility analysis data not available")
            except Exception as e:
                st.error(f"Error in volatility analysis: {e}")
        else:
            st.info("Run analysis first to see volatility analysis")

    # Tab 5 - Historical Trends
    with tab5:
        st.subheader("🕐 Historical Analysis")
        st.info("🚧 Historical trend analysis: Configure date range and run batch analysis")

        col1, col2 = st.columns(2)
        with col1:
            hist_start = st.date_input(
                "Start Date",
                value=datetime.datetime.now().date() - datetime.timedelta(days=365),
                help="Start date for historical analysis"
            )
        with col2:
            hist_end = st.date_input(
                "End Date",
                value=datetime.datetime.now().date(),
                help="End date for historical analysis"
            )

        if st.button("Run Historical Analysis"):
            with st.spinner("Analyzing historical events... This may take several minutes."):
                try:
                    fred_fetcher, yahoo_fetcher, analyzer, _ = initialize_components()
                    hist_results = analyzer.batch_analyze_historical(
                        selected_indicator,
                        datetime.datetime.combine(hist_start, datetime.datetime.min.time()),
                        datetime.datetime.combine(hist_end, datetime.datetime.min.time())
                    )

                    if hist_results.empty:
                        st.warning(f"⚠️ No historical data found for {selected_indicator} in the specified date range")
                    else:
                        st.success("✅ Historical analysis complete!")

                        # Display summary stats
                        if hasattr(hist_results, 'attrs') and 'summary_stats' in hist_results.attrs:
                            stats = hist_results.attrs['summary_stats']
                            st.subheader("📊 Summary Statistics")

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Events", stats.get('total_events', 0))
                            with col2:
                                st.metric("Avg Value", f"{stats.get('avg_value', 0):.2f}")
                            with col3:
                                st.metric("Avg Change", f"{stats.get('avg_change_pct', 0):+.2f}%")
                            with col4:
                                st.metric("Events Analyzed", len(hist_results))

                        # Display results table
                        st.subheader("📋 Historical Results")
                        st.dataframe(hist_results, use_container_width=True)

                        # Download button
                        csv = hist_results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Historical Results",
                            data=csv,
                            file_name=f"historical_{selected_indicator}_{hist_start}_{hist_end}.csv",
                            mime="text/csv"
                        )

                except Exception as e:
                    st.error(f"Error in historical analysis: {str(e)}")
                    st.info("💡 Try a shorter date range or check your internet connection")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
    <p>📊 <strong>Macro Event Impact Tracker</strong> • 100% Free & Open Source</p>
    <p>Data: FRED API & Yahoo Finance • No credit card required</p>
    <p>Built with ❤️ for finance students and professionals</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
