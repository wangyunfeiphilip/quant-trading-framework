#include "order.hpp"

#include <cmath>

namespace quant {

void Order::validate() const {
    if (ticker.empty()) {
        throw std::invalid_argument("ticker is empty");
    }
    if (quantity <= 0) {
        throw std::invalid_argument("quantity must be positive");
    }
    if (!(reference_price > 0.0) || !std::isfinite(reference_price)) {
        throw std::invalid_argument("reference_price must be positive and finite");
    }
}

double Fill::notional() const {
    return executed_price * static_cast<double>(quantity);
}

Fill execute_order(const Order& order, const ExecutionConfig& config) {
    order.validate();
    const double slippage_rate = config.slippage_bps / 10000.0;
    const double side_multiplier = order.side == OrderSide::Buy ? 1.0 : -1.0;
    const double executed_price = order.reference_price * (1.0 + side_multiplier * slippage_rate);
    const double notional = executed_price * static_cast<double>(order.quantity);

    Fill fill{
        order.date,
        order.ticker,
        order.side,
        order.quantity,
        order.reference_price,
        executed_price,
        std::abs(notional) * config.transaction_cost_bps / 10000.0,
        std::abs(executed_price - order.reference_price) * static_cast<double>(order.quantity),
    };
    return fill;
}

}  // namespace quant
