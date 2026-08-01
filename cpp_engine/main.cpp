#include "backtester.hpp"

#include <iostream>
#include <vector>

int main() {
    std::vector<quant::PriceBar> prices{
        {"2026-01-02", "AAPL", 100.0},
        {"2026-01-02", "MSFT", 200.0},
        {"2026-01-03", "AAPL", 102.0},
        {"2026-01-03", "MSFT", 198.0},
    };
    std::vector<quant::TargetWeight> weights{
        {"2026-01-02", "AAPL", 0.5},
        {"2026-01-02", "MSFT", 0.5},
    };

    quant::BacktestEngine engine(100000.0);
    const auto values = engine.run(prices, weights);
    for (const auto& row : values) {
        std::cout << row.date << "," << row.total_value << "\n";
    }
    return 0;
}
