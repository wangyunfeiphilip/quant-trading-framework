#include "position.hpp"

#include <stdexcept>
#include <utility>

namespace quant {

Position::Position(std::string ticker) : ticker_(std::move(ticker)) {}

const std::string& Position::ticker() const {
    return ticker_;
}

int Position::quantity() const {
    return quantity_;
}

double Position::average_cost() const {
    return average_cost_;
}

double Position::market_value(double price) const {
    return static_cast<double>(quantity_) * price;
}

void Position::buy(int quantity, double price) {
    if (quantity <= 0 || price <= 0.0) {
        throw std::invalid_argument("buy quantity and price must be positive");
    }
    const int new_quantity = quantity_ + quantity;
    average_cost_ = (average_cost_ * quantity_ + price * quantity) / static_cast<double>(new_quantity);
    quantity_ = new_quantity;
}

void Position::sell(int quantity) {
    if (quantity <= 0) {
        throw std::invalid_argument("sell quantity must be positive");
    }
    if (quantity > quantity_) {
        throw std::invalid_argument("cannot sell more shares than held");
    }
    quantity_ -= quantity;
    if (quantity_ == 0) {
        average_cost_ = 0.0;
    }
}

}  // namespace quant
