#include "portfolio.hpp"

#include <stdexcept>

namespace quant {

Portfolio::Portfolio(double initial_cash) : initial_cash_(initial_cash), cash_(initial_cash) {
    if (initial_cash <= 0.0) {
        throw std::invalid_argument("initial_cash must be positive");
    }
}

double Portfolio::cash() const {
    return cash_;
}

int Portfolio::quantity(const std::string& ticker) const {
    const auto it = positions_.find(ticker);
    return it == positions_.end() ? 0 : it->second.quantity();
}

double Portfolio::holdings_value(const std::unordered_map<std::string, double>& prices) const {
    double value = 0.0;
    for (const auto& item : positions_) {
        const auto price_it = prices.find(item.first);
        if (price_it != prices.end()) {
            value += item.second.market_value(price_it->second);
        }
    }
    return value;
}

double Portfolio::total_value(const std::unordered_map<std::string, double>& prices) const {
    return cash_ + holdings_value(prices);
}

const std::vector<TradeRecord>& Portfolio::trades() const {
    return trades_;
}

void Portfolio::apply_fill(const Fill& fill) {
    auto& position = positions_.try_emplace(fill.ticker, fill.ticker).first->second;
    const double gross = fill.notional();

    if (fill.side == OrderSide::Buy) {
        const double total_cost = gross + fill.transaction_cost;
        if (total_cost > cash_ + 1e-9) {
            throw std::runtime_error("insufficient cash");
        }
        position.buy(fill.quantity, fill.executed_price);
        cash_ -= total_cost;
    } else {
        position.sell(fill.quantity);
        cash_ += gross - fill.transaction_cost;
        if (position.quantity() == 0) {
            positions_.erase(fill.ticker);
        }
    }

    trades_.push_back(
        TradeRecord{
            fill.date,
            fill.ticker,
            fill.side,
            fill.quantity,
            fill.executed_price,
            fill.transaction_cost,
            cash_,
        }
    );
}

}  // namespace quant
