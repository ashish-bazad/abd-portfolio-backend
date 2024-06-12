import numpy as np
import random
import copy


def get_random_array(n):
    array = list(range(n))
    np.random.shuffle(array)

    return array

def get_adjusted_weights(portfolio_minimum_weights, portfolio_maximum_weights, basket_minimum_weights, basket_maximum_weights):
    
    total_minimum_sum = np.sum(basket_minimum_weights)
    portfolio_minimum_weights_sum = [np.sum(i) for i in portfolio_minimum_weights]

    tmp_basket_assignment = copy.deepcopy(basket_minimum_weights)
    tmp_weights_assignment = copy.deepcopy(portfolio_minimum_weights)

    left = 1 - total_minimum_sum
    current_basket_weights_sum = 0

    for i in range(len(basket_minimum_weights)):
        if i != len(basket_minimum_weights) - 1:
            tmp_basket_assignment[i] = np.random.uniform(low = basket_minimum_weights[i], high = min(basket_maximum_weights[i], basket_minimum_weights[i] + left))
            current_basket_weights_sum += tmp_basket_assignment[i]

        else:
            tmp_basket_assignment[i] = 1 - current_basket_weights_sum

            if tmp_basket_assignment[i] > basket_maximum_weights[i]:
                excess_weight = tmp_basket_assignment[i] - basket_maximum_weights[i]
                tmp_basket_assignment[i] = basket_maximum_weights[i]

                for k in range(len(basket_minimum_weights)):
                    gap = basket_maximum_weights[k] - tmp_basket_assignment[k]
                    tmp_basket_assignment[k] = min(tmp_basket_assignment[k] + excess_weight, basket_maximum_weights[k])
                    excess_weight = max(0, excess_weight - gap)

                    if excess_weight <= 0:
                        break

        left -= tmp_basket_assignment[i] - basket_minimum_weights[i]

        left_weights = tmp_basket_assignment[i] - portfolio_minimum_weights_sum[i]
        current_portfolio_weights_sum = 0

        for j in range(len(portfolio_minimum_weights[i])):

            if j != len(portfolio_minimum_weights[i]) - 1:
                tmp_weights_assignment[i][j] = np.random.uniform(low = portfolio_minimum_weights[i][j], high = min(portfolio_maximum_weights[i][j], portfolio_minimum_weights[i][j] + left_weights))
                current_portfolio_weights_sum += tmp_weights_assignment[i][j]

            else:
                tmp_weights_assignment[i][j] = tmp_basket_assignment[i] - current_portfolio_weights_sum

                if tmp_weights_assignment[i][j] > portfolio_maximum_weights[i][j]:
                    excess_weight = tmp_weights_assignment[i][j] - portfolio_maximum_weights[i][j]
                    tmp_weights_assignment[i][j] = portfolio_maximum_weights[i][j]
                    
                    
                    for k in get_random_array(len(portfolio_minimum_weights[i])):
                        gap = portfolio_maximum_weights[i][k] - tmp_weights_assignment[i][k]
                        tmp_weights_assignment[i][k] = min(tmp_weights_assignment[i][k] + excess_weight, portfolio_maximum_weights[i][k])
                        excess_weight = max(0, excess_weight - gap)

                        if excess_weight == 0:
                            break

            left_weights -= tmp_weights_assignment[i][j] - portfolio_minimum_weights[i][j]

    adjusted_weights = np.array([j for i in tmp_weights_assignment for j in i])
    adjusted_basket_weights = tmp_basket_assignment 

    return adjusted_weights, adjusted_basket_weights