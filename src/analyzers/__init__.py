"""
Analyzers Package - Core Analysis Engine

This package contains the core analysis logic for processing macroeconomic events
and their impacts on financial markets. The ImpactAnalyzer provides comprehensive
event analysis with statistical testing, multi-asset correlations, and professional
reporting capabilities.

Key Components:
- ImpactAnalyzer: Main analysis engine for single events and historical batch processing
- Market reaction analysis across multiple timeframes (15min, 30min, 1hr)
- Statistical significance testing using t-tests
- Cross-asset correlation analysis during event windows
- Sector-level impact aggregation
- Volatility change analysis (pre vs post event)
- Historical trend analysis and comparison

Features:
- Real-time event impact assessment
- Multi-asset analysis (equities, bonds, commodities, FX, volatility)
- Statistical validation of market reactions
- Human-readable impact summaries and reports
- Batch processing for historical event analysis
- CSV export capabilities for further analysis

Analysis Windows:
- 15 Minutes: Immediate market reaction
- 30 Minutes: Short-term absorption
- 1 Hour: Medium-term stabilization
- 2 Hours: Longer-term reaction (when available)

Classes:
    ImpactAnalyzer: Core analysis engine with all impact analysis functionality

Usage:
    from src.analyzers.impact_analyzer import ImpactAnalyzer

    analyzer = ImpactAnalyzer()
    results = analyzer.analyze_single_event('CPI', datetime(2024, 1, 15))
    summary = analyzer.generate_impact_summary(results)
"""
