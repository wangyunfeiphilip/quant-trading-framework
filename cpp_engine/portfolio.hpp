#pragma once

#include "order.hpp"
#include "position.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace quant {

struct TradeRecord {
    std::string date;
    std::string ticker;
    OrderSide side;
    int quantity;
    double executed_price;
    double transaction_cost;
    double cash_after;
};

class Portfolio {
public:
    explicit Portfolio(double initial_cash = 100000.0);

    double cash() const;
    int quantity(const std::string& ticker) const;
    double holdings_value(const std::unordered_map<std::string, double>& prices) const;
    double total_value(const std::unordered_map<std::string, double>& prices) const;
    const std::vector<TradeRecord>& trades() const;

    void apply_fill(const Fill& fill);

private:
    double initial_cash_;
    double cash_;
    std::unordered_map<std::string, Position> positions_;
    std::vector<TradeRecord> trades_;
};

}  // namespace quant
