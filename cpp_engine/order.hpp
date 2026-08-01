#pragma once

#include <stdexcept>
#include <string>

namespace quant {

enum class OrderSide {
    Buy,
    Sell
};

struct Order {
    std::string date;
    std::string ticker;
    OrderSide side;
    int quantity;
    double reference_price;

    void validate() const;
};

struct ExecutionConfig {
    double transaction_cost_bps = 5.0;
    double slippage_bps = 2.0;
};

struct Fill {
    std::string date;
    std::string ticker;
    OrderSide side;
    int quantity;
    double reference_price;
    double executed_price;
    double transaction_cost;
    double slippage_cost;

    double notional() const;
};

Fill execute_order(const Order& order, const ExecutionConfig& config);

}  // namespace quant
