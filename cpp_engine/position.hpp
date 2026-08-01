#pragma once

#include <string>

namespace quant {

class Position {
public:
    explicit Position(std::string ticker = "");

    const std::string& ticker() const;
    int quantity() const;
    double average_cost() const;
    double market_value(double price) const;

    void buy(int quantity, double price);
    void sell(int quantity);

private:
    std::string ticker_;
    int quantity_ = 0;
    double average_cost_ = 0.0;
};

}  // namespace quant
