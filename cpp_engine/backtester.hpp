#pragma once

#include "portfolio.hpp"

#include <string>
#include <unordered_map>
#include <vector>

#ifdef QUANT_HAS_EIGEN
#include <Eigen/Dense>
#endif

namespace quant {

struct PriceBar {
    std::string date;
    std::string ticker;
    double close;
};

struct TargetWeight {
    std::string date;
    std::string ticker;
    double weight;
};

struct DailyValue {
    std::string date;
    double cash;
    double holdings;
    double total_value;
};

class BacktestEngine {
public:
    BacktestEngine(double initial_capital = 100000.0, ExecutionConfig execution = {});

    std::vector<DailyValue> run(
        const std::vector<PriceBar>& prices,
        const std::vector<TargetWeight>& weights
    );

    const Portfolio& portfolio() const;

#ifdef QUANT_HAS_EIGEN
    static Eigen::VectorXd returns_from_values(const std::vector<DailyValue>& values);
#endif

private:
    void rebalance(
        const std::string& date,
        const std::unordered_map<std::string, double>& prices,
        const std::unordered_map<std::string, double>& weights
    );

    Portfolio portfolio_;
    ExecutionConfig execution_;
};

}  // namespace quant
