#include <solver/generated/stable_neo_hookean.h>

#include <iomanip>
#include <iostream>

int main()
{
    double mu, lambda;
    std::cout << std::setprecision(17);
    while(std::cin >> mu >> lambda)
    {
        Eigen::Matrix3d F;
        for(int i = 0; i < 3; ++i)
            for(int j = 0; j < 3; ++j)
                std::cin >> F(i, j);
        Eigen::Matrix<double, 9, 1> gradient;
        qipc::snh_gradient<double>(mu, lambda, F, gradient);
        std::cout << qipc::snh_energy<double>(mu, lambda, F);
        for(int i = 0; i < 3; ++i)
            for(int j = 0; j < 3; ++j)
                std::cout << ' ' << gradient(i + 3 * j);
        std::cout << '\n';
    }
}
