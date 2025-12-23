#ifndef MONTECARLO_PRICER_PAYOFF_HPP
#define MONTECARLO_PRICER_PAYOFF_HPP

#include <memory>

namespace montecarlo{
    class Payoff{
        public:
        virtual double operator()(double spot) const = 0;
        virtual ~Payoff() = default;
    };

    class EuropeanCall : public Payoff{
    public:
        explicit EuropeanCall(double strike);
        double operator()(double spot) const override;
    private:
        double strike_;
    };
    
    class EuropeanPut : public Payoff{
    public:
        explicit EuropeanPut(double strike);
        double operator()(double spot) const override;
    private:
        double strike_;
    };

    inline std::unique_ptr<Payoff> make_call(double strike){
        return std::make_unique<EuropeanCall>(strike);
    }
    inline std::unique_ptr<Payoff> make_put(double strike){
        return std::make_unique<EuropeanPut>(strike);
    }
}

#endif // MONTECARLO_PRICER_PAYOFF_HPP