#include "payoff.hpp"
#include <algorithm>

namespace montecarlo{
    EuropeanCall::EuropeanCall(double strike)
        : strike_(strike)
    {
    }

    double EuropeanCall::operator()(double spot) const
    {
        return std::max(spot - strike_, 0.0);
    }

    EuropeanPut::EuropeanPut(double strike)
        : strike_(strike)
    {
    }

    double EuropeanPut::operator()(double spot) const
    {
        return std::max(strike_ - spot, 0.0);
    }
}
