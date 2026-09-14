from dashboard.search import build_search_index, search_catalog


def test_search_catalog_matches_ticker_and_metric() -> None:
    catalog = build_search_index(["NVDA", "AAPL"])

    ticker_results = search_catalog("nvda", catalog)
    metric_results = search_catalog("sharpe", catalog)

    assert ticker_results[0].title == "NVDA"
    assert any(item.title == "Sharpe Ratio" for item in metric_results)


def test_search_catalog_matches_derivative_concept() -> None:
    catalog = build_search_index([])
    results = search_catalog("black scholes greeks", catalog)

    titles = {item.title for item in results}
    assert "Black-Scholes" in titles
    assert "Greeks" in titles


def test_search_catalog_matches_research_frameworks_and_a_share_code() -> None:
    catalog = build_search_index(["SZ.301321"])

    framework_results = search_catalog("财报 sec finbert", catalog)
    ashare_results = search_catalog("301321", catalog)

    assert any(item.title == "financial-research-agent" for item in framework_results)
    assert ashare_results[0].title == "SZ.301321"
