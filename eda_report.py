"""
Generate Sweetviz EDA report.
"""
import sweetviz as sv

def generate_report(df, output_path: str):
    """Create and save Sweetviz report."""
    try:
        print(df.head())  # Debugging line to check DataFrame content
        report = sv.analyze(df)
        report.show_html(filepath=output_path, open_browser=False)
    except Exception as e:
        print(f"Error generating report: {e}")
