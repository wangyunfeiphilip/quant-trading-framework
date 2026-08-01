import pandas as pd

from data.data_loader import generate_data_quality_report


def test_generate_data_quality_report_counts_duplicates() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-01", "2026-01-02"],
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "close": [100.0, 100.0, 101.0],
            "adjusted_close": [100.0, 100.0, 101.0],
        }
    )
    report = generate_data_quality_report(frame)
    assert report.loc[0, "duplicate_records"] == 1
