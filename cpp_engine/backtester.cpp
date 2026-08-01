#include "backtester.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

namespace quant {

BacktestEngine::BacktestEngine(double initial_capital, ExecutionConfig execution)
    : portfolio_(initial_capital), execution_(execution) {}

const Portfolio& BacktestEngine::portfolio() const {
    return portfolio_;
}

std::vector<DailyValue> BacktestEngine::run(
    const std::vector<PriceBar>& prices,
    const std::vector<TargetWeight>& weights
) {
    std::map<std::string, std::unordered_map<std::string, double>> prices_by_date;
    std::map<std::string, std::unordered_map<std::string, double>> weights_by_date;

    for (const auto& bar : prices) {
        prices_by_date[bar.date][bar.ticker] = bar.close;
    }
    for (const auto& target : weights) {
        weights_by_date[target.date][target.ticker] = std::max(0.0, target.weight);
    }

    std::vector<DailyValue> output;
    std::unordered_map<std::string, double> active_weights;
    for (const auto& item : prices_by_date) {
        const auto& date = item.first;
        const auto& price_map = item.second;
        const auto weights_it = weights_by_date.find(date);
        if (weights_it != weights_by_date.end()) {
            active_weights = weights_it->second;
            rebalance(date, price_map, active_weights);
        }

        const double holdings = portfolio_.holdings_value(price_map);
        output.push_back(DailyValue{date, portfolio_.cash(), holdings, portfolio_.cash() + holdings});
    }
    return output;
}

void BacktestEngine::rebalance(
    const std::string& date,
    const std::unordered_map<std::string, double>& prices,
    const std::unordered_map<std::string, double>& weights
) {
    const double total = portfolio_.total_value(prices);
    std::set<std::string> tickers;
    for (const auto& item : prices) {
        tickers.insert(item.first);
    }
    for (const auto& item : weights) {
        tickers.insert(item.first);
    }

    std::unordered_map<std::string, int> desired_quantities;
    for (const auto& ticker : tickers) {
        const auto price_it = prices.find(ticker);
        if (price_it == prices.end() || price_it->second <= 0.0) {
            continue;
        }
        const double weight = weights.count(ticker) ? weights.at(ticker) : 0.0;
        desired_quantities[ticker] = static_cast<int>(std::floor(total * weight / price_it->second));
    }

    for (const auto& item : desired_quantities) {
        const int current = portfolio_.quantity(item.first);
        if (current > item.second) {
            Order order{date, item.first, OrderSide::Sell, current - item.second, prices.at(item.first)};
            portfolio_.apply_fill(execute_order(order, execution_));
        }
    }

    for (const auto& item : desired_quantities) {
        const int current = portfolio_.quantity(item.first);
        if (item.second > current) {
            const double price = prices.at(item.first);
            const double unit_cash_required =
                price
                * (1.0 + execution_.slippage_bps / 10000.0)
                * (1.0 + execution_.transaction_cost_bps / 10000.0);
            const int max_affordable = static_cast<int>(std::floor(portfolio_.cash() / unit_cash_required));
            const int quantity = std::min(item.second - current, max_affordable);
            if (quantity <= 0) {
                continue;
            }
            Order order{date, item.first, OrderSide::Buy, quantity, price};
            portfolio_.apply_fill(execute_order(order, execution_));
        }
    }
}

#ifdef QUANT_HAS_EIGEN
Eigen::VectorXd BacktestEngine::returns_from_values(const std::vector<DailyValue>& values) {
    if (values.size() < 2) {
        return Eigen::VectorXd();
    }
    Eigen::VectorXd returns(static_cast<int>(values.size() - 1));
    for (std::size_t i = 1; i < values.size(); ++i) {
        returns(static_cast<int>(i - 1)) = values[i].total_value / values[i - 1].total_value - 1.0;
    }
    return returns;
}
#endif

}  // namespace quant
