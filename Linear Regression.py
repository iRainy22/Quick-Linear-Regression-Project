from numpy import *


def calc_error_for_line(b, m, points):
    #initaialize at 0
    totalError = 0

    for i in range(0,len(points)):
        #grab x and y values
        x = points[i,0]
        y = points[i,1]

        #get the difference, square it, and add it to the total
        totalError += (y - (m * x + b)) ** 2

    return totalError / float(len(points))

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    #initial b, m values
    b = starting_b
    m = starting_m

    #gradient descent
    for i in range(num_iterations):
        #update b, m with the new and more accurate b, m by performing gradeient step
        b, m= step_gradient(b,m, array(points), learning_rate)
    return [b, m]

def step_gradient(b_current, m_current, points, learning_rate):
    #starting point for our gradients
    b_gradient = 0
    m_gradient = 0

    N = float(len(points))

    for i in range(0, len(points)):
        x = points[i,0]
        y = points[i,1]
        #direction with respect to b and m
        #compute partial derivatives of our error function
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))

    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return[new_b, new_m]
    
def run():

    #Step 1 - collect our data
    points = genfromtxt('data.csv', delimiter=',')


    #step 2 - define our hyperparameters
    # how fast till our model converges
    learning_rate = 0.0001
    #for equation of line
    initial_b = 0
    initial_m = 0
    num_iterations = 1000
    
    #step 3 - train our model
    print('starting gradient descent at b = {0}, m = {1}, error = {2}'.format(initial_b,initial_m,calc_error_for_line(initial_b,initial_m,points)))
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print('ending gradient descent at b = {1}, m = {2}, error = {3}'.format(num_iterations, b,m,calc_error_for_line(b,m,points)))





if __name__ == '__main__':
    run()