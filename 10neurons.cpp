#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>

using namespace std;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

double loss(double y_true, double y_pred) {
    return 0.5 * pow(y_true - y_pred, 2);
}

double loss_derivative(double y_true, double y_pred) {
    return y_pred - y_true;
}

double neuron(double n1, double n2, double n3, double n4, double n5, double n6, double n7, double n8, double n9, double n10,
              double w1, double w2, double w3, double w4, double w5, double w6, double w7, double w8, double w9, double w10,
              double bias) {
    double weighted_sum = (w1 * n1) + (w2 * n2) + (w3 * n3) + (w4 * n4) + (w5 * n5) + (w6 * n6) + (w7 * n7) + (w8 * n8) + 
                          (w9 * n9) + (w10 * n10) + bias;
    return sigmoid(weighted_sum);
}

double random_weight() {
    return (rand() % 1001) / 1000.0 - 0.5; 
}

int main() {

    const int data_size = 1000;
    double x_train[data_size];
    double y_train[data_size];

    for (int i = 0; i < data_size; i++) {
        x_train[i] = (i * 2 * M_PI) / data_size;  
        y_train[i] = sin(x_train[i]);  
    }

    double learning_rate = 0.01;
    srand(time(0));

    // Initialize weights and biases Layer 1 (10 neurons)
    double w1_1 = random_weight(), w1_2 = random_weight(), w1_3 = random_weight(), w1_4 = random_weight(), w1_5 = random_weight(), w1_6 = random_weight(), w1_7 = random_weight(), w1_8 = random_weight(), w1_9 = random_weight(), w1_10 = random_weight();
    double bias_1_1 = random_weight(), bias_1_2 = random_weight(), bias_1_3 = random_weight(), bias_1_4 = random_weight(), bias_1_5 = random_weight(), bias_1_6 = random_weight(), bias_1_7 = random_weight(), bias_1_8 = random_weight(), bias_1_9 = random_weight(), bias_1_10 = random_weight();
    
    // Initialize weights and biases Layer 2 (10 neurons)
    double w2_1_1 = random_weight(), w2_1_2 = random_weight(), w2_1_3 = random_weight(), w2_1_4 = random_weight(), w2_1_5 = random_weight(), w2_1_6 = random_weight(), w2_1_7 = random_weight(), w2_1_8 = random_weight(), w2_1_9 = random_weight(), w2_1_10 = random_weight();
    double bias_2_1 = random_weight();

    double w2_2_1 = random_weight(), w2_2_2 = random_weight(), w2_2_3 = random_weight(), w2_2_4 = random_weight(), w2_2_5 = random_weight(), w2_2_6 = random_weight(), w2_2_7 = random_weight(), w2_2_8 = random_weight(), w2_2_9 = random_weight(), w2_2_10 = random_weight();
    double bias_2_2 = random_weight();

    double w2_3_1 = random_weight(), w2_3_2 = random_weight(), w2_3_3 = random_weight(), w2_3_4 = random_weight(), w2_3_5 = random_weight(), w2_3_6 = random_weight(), w2_3_7 = random_weight(), w2_3_8 = random_weight(), w2_3_9 = random_weight(), w2_3_10 = random_weight();
    double bias_2_3 = random_weight();

    double w2_4_1 = random_weight(), w2_4_2 = random_weight(), w2_4_3 = random_weight(), w2_4_4 = random_weight(), w2_4_5 = random_weight(), w2_4_6 = random_weight(), w2_4_7 = random_weight(), w2_4_8 = random_weight(), w2_4_9 = random_weight(), w2_4_10 = random_weight();
    double bias_2_4 = random_weight();

    double w2_5_1 = random_weight(), w2_5_2 = random_weight(), w2_5_3 = random_weight(), w2_5_4 = random_weight(), w2_5_5 = random_weight(), w2_5_6 = random_weight(), w2_5_7 = random_weight(), w2_5_8 = random_weight(), w2_5_9 = random_weight(), w2_5_10 = random_weight();
    double bias_2_5 = random_weight();

    double w2_6_1 = random_weight(), w2_6_2 = random_weight(), w2_6_3 = random_weight(), w2_6_4 = random_weight(), w2_6_5 = random_weight(), w2_6_6 = random_weight(), w2_6_7 = random_weight(), w2_6_8 = random_weight(), w2_6_9 = random_weight(), w2_6_10 = random_weight();
    double bias_2_6 = random_weight();

    double w2_7_1 = random_weight(), w2_7_2 = random_weight(), w2_7_3 = random_weight(), w2_7_4 = random_weight(), w2_7_5 = random_weight(), w2_7_6 = random_weight(), w2_7_7 = random_weight(), w2_7_8 = random_weight(), w2_7_9 = random_weight(), w2_7_10 = random_weight();
    double bias_2_7 = random_weight();

    double w2_8_1 = random_weight(), w2_8_2 = random_weight(), w2_8_3 = random_weight(), w2_8_4 = random_weight(), w2_8_5 = random_weight(), w2_8_6 = random_weight(), w2_8_7 = random_weight(), w2_8_8 = random_weight(), w2_8_9 = random_weight(), w2_8_10 = random_weight();
    double bias_2_8 = random_weight();

    double w2_9_1 = random_weight(), w2_9_2 = random_weight(), w2_9_3 = random_weight(), w2_9_4 = random_weight(), w2_9_5 = random_weight(), w2_9_6 = random_weight(), w2_9_7 = random_weight(), w2_9_8 = random_weight(), w2_9_9 = random_weight(), w2_9_10 = random_weight();
    double bias_2_9 = random_weight();

    double w2_10_1 = random_weight(), w2_10_2 = random_weight(), w2_10_3 = random_weight(), w2_10_4 = random_weight(), w2_10_5 = random_weight(), w2_10_6 = random_weight(), w2_10_7 = random_weight(), w2_10_8 = random_weight(), w2_10_9 = random_weight(), w2_10_10 = random_weight();
    double bias_2_10 = random_weight();

    double w_output1 = random_weight(), w_output2 = random_weight(), w_output3 = random_weight(), w_output4 = random_weight(), w_output5 = random_weight(), w_output6 = random_weight(), w_output7 = random_weight(), w_output8 = random_weight(), w_output9 = random_weight(), w_output10 = random_weight();
    double bias_output = random_weight();

    for (int epochs = 0; epochs < 500; epochs++) {
        double total_loss = 0;

        for (int i = 0; i < data_size; i++) {
            double x1 = x_train[i];
            double y = y_train[i];
            
            // Forward pass: Input -> Hidden layer 1           
            double hidden1_1  = neuron(x1, 0, 0, 0, 0, 0, 0, 0, 0, 0, w1_1, w1_2, w1_3, w1_4, w1_5, w1_6, w1_7, w1_8, w1_9, w1_10, bias_1_1);
            double hidden1_2  = neuron(x1, 0, 0, 0, 0, 0, 0, 0, 0, 0, w1_1, w1_2, w1_3, w1_4, w1_5, w1_6, w1_7, w1_8, w1_9, w1_10, bias_1_2);
            double hidden1_3  = neuron(x1, 0, 0, 0, 0, 0, 0, 0, 0, 0, w1_1, w1_2, w1_3, w1_4, w1_5, w1_6, w1_7, w1_8, w1_9, w1_10, bias_1_3);
            double hidden1_4  = neuron(x1, 0, 0, 0, 0, 0, 0, 0, 0, 0, w1_1, w1_2, w1_3, w1_4, w1_5, w1_6, w1_7, w1_8, w1_9, w1_10, bias_1_4);
            double hidden1_5  = neuron(x1, 0, 0, 0, 0, 0, 0, 0, 0, 0, w1_1, w1_2, w1_3, w1_4, w1_5, w1_6, w1_7, w1_8, w1_9, w1_10, bias_1_5);
            double hidden1_6  = neuron(x1, 0, 0, 0, 0, 0, 0, 0, 0, 0, w1_1, w1_2, w1_3, w1_4, w1_5, w1_6, w1_7, w1_8, w1_9, w1_10, bias_1_6);
            double hidden1_7  = neuron(x1, 0, 0, 0, 0, 0, 0, 0, 0, 0, w1_1, w1_2, w1_3, w1_4, w1_5, w1_6, w1_7, w1_8, w1_9, w1_10, bias_1_7);
            double hidden1_8  = neuron(x1, 0, 0, 0, 0, 0, 0, 0, 0, 0, w1_1, w1_2, w1_3, w1_4, w1_5, w1_6, w1_7, w1_8, w1_9, w1_10, bias_1_8);
            double hidden1_9  = neuron(x1, 0, 0, 0, 0, 0, 0, 0, 0, 0, w1_1, w1_2, w1_3, w1_4, w1_5, w1_6, w1_7, w1_8, w1_9, w1_10, bias_1_9);
            double hidden1_10 = neuron(x1, 0, 0, 0, 0, 0, 0, 0, 0, 0, w1_1, w1_2, w1_3, w1_4, w1_5, w1_6, w1_7, w1_8, w1_9, w1_10, bias_1_10);

            
            // Forward pass: Hidden layer 1 -> Hidden layer 2   

            double hidden2_1 = neuron(hidden1_1, hidden1_2, hidden1_3, hidden1_4, hidden1_5, hidden1_6, hidden1_7, hidden1_8, hidden1_9, hidden1_10, 
                          w2_1_1, w2_1_2, w2_1_3, w2_1_4, w2_1_5, w2_1_6, w2_1_7, w2_1_8, w2_1_9, w2_1_10, bias_2_1);

            double hidden2_2 = neuron(hidden1_1, hidden1_2, hidden1_3, hidden1_4, hidden1_5, hidden1_6, hidden1_7, hidden1_8, hidden1_9, hidden1_10, 
                          w2_2_1, w2_2_2, w2_2_3, w2_2_4, w2_2_5, w2_2_6, w2_2_7, w2_2_8, w2_2_9, w2_2_10, bias_2_2);

            double hidden2_3 = neuron(hidden1_1, hidden1_2, hidden1_3, hidden1_4, hidden1_5, hidden1_6, hidden1_7, hidden1_8, hidden1_9, hidden1_10, 
                          w2_3_1, w2_3_2, w2_3_3, w2_3_4, w2_3_5, w2_3_6, w2_3_7, w2_3_8, w2_3_9, w2_3_10, bias_2_3);

            double hidden2_4 = neuron(hidden1_1, hidden1_2, hidden1_3, hidden1_4, hidden1_5, hidden1_6, hidden1_7, hidden1_8, hidden1_9, hidden1_10, 
                          w2_4_1, w2_4_2, w2_4_3, w2_4_4, w2_4_5, w2_4_6, w2_4_7, w2_4_8, w2_4_9, w2_4_10, bias_2_4);

            double hidden2_5 = neuron(hidden1_1, hidden1_2, hidden1_3, hidden1_4, hidden1_5, hidden1_6, hidden1_7, hidden1_8, hidden1_9, hidden1_10, 
                          w2_5_1, w2_5_2, w2_5_3, w2_5_4, w2_5_5, w2_5_6, w2_5_7, w2_5_8, w2_5_9, w2_5_10, bias_2_5);

            double hidden2_6 = neuron(hidden1_1, hidden1_2, hidden1_3, hidden1_4, hidden1_5, hidden1_6, hidden1_7, hidden1_8, hidden1_9, hidden1_10, 
                          w2_6_1, w2_6_2, w2_6_3, w2_6_4, w2_6_5, w2_6_6, w2_6_7, w2_6_8, w2_6_9, w2_6_10, bias_2_6);

            double hidden2_7 = neuron(hidden1_1, hidden1_2, hidden1_3, hidden1_4, hidden1_5, hidden1_6, hidden1_7, hidden1_8, hidden1_9, hidden1_10, 
                          w2_7_1, w2_7_2, w2_7_3, w2_7_4, w2_7_5, w2_7_6, w2_7_7, w2_7_8, w2_7_9, w2_7_10, bias_2_7);

            double hidden2_8 = neuron(hidden1_1, hidden1_2, hidden1_3, hidden1_4, hidden1_5, hidden1_6, hidden1_7, hidden1_8, hidden1_9, hidden1_10, 
                          w2_8_1, w2_8_2, w2_8_3, w2_8_4, w2_8_5, w2_8_6, w2_8_7, w2_8_8, w2_8_9, w2_8_10, bias_2_8);

            double hidden2_9 = neuron(hidden1_1, hidden1_2, hidden1_3, hidden1_4, hidden1_5, hidden1_6, hidden1_7, hidden1_8, hidden1_9, hidden1_10, 
                          w2_9_1, w2_9_2, w2_9_3, w2_9_4, w2_9_5, w2_9_6, w2_9_7, w2_9_8, w2_9_9, w2_9_10, bias_2_9);

            double hidden2_10 = neuron(hidden1_1, hidden1_2, hidden1_3, hidden1_4, hidden1_5, hidden1_6, hidden1_7, hidden1_8, hidden1_9, hidden1_10, 
                           w2_10_1, w2_10_2, w2_10_3, w2_10_4, w2_10_5, w2_10_6, w2_10_7, w2_10_8, w2_10_9, w2_10_10, bias_2_10);


            // Forward pass: Hidden layer 2 -> Output
            double final_output = neuron(hidden2_1, hidden2_2, hidden2_3, hidden2_4, hidden2_5, hidden2_6, hidden2_7, hidden2_8, hidden2_9, hidden2_10,w_output1, w_output2, w_output3, w_output4, w_output5, w_output6, w_output7, w_output8, w_output9, w_output10,bias_output);

            double error_output = loss_derivative(y, final_output);
            total_loss += loss(y, final_output); 

            // Backpropagation: Update weights and biases of the output layer
            w_output1   -= learning_rate * error_output * hidden2_1 * sigmoid_derivative(final_output);
            w_output2   -= learning_rate * error_output * hidden2_2 * sigmoid_derivative(final_output);
            w_output3   -= learning_rate * error_output * hidden2_3 * sigmoid_derivative(final_output);
            w_output4   -= learning_rate * error_output * hidden2_4 * sigmoid_derivative(final_output);
            w_output5   -= learning_rate * error_output * hidden2_5 * sigmoid_derivative(final_output);
            w_output6   -= learning_rate * error_output * hidden2_6 * sigmoid_derivative(final_output);
            w_output7   -= learning_rate * error_output * hidden2_7 * sigmoid_derivative(final_output);
            w_output8   -= learning_rate * error_output * hidden2_8 * sigmoid_derivative(final_output);
            w_output9   -= learning_rate * error_output * hidden2_9 * sigmoid_derivative(final_output);
            w_output10  -= learning_rate * error_output * hidden2_10* sigmoid_derivative(final_output);
            bias_output -= learning_rate * error_output * sigmoid_derivative(final_output);

            // Backpropagation: Update hidden layer 2 weights
            double hidden2_1_error = error_output * w_output1 * sigmoid_derivative(hidden2_1);
            double hidden2_2_error = error_output * w_output2 * sigmoid_derivative(hidden2_2);
            double hidden2_3_error = error_output * w_output3 * sigmoid_derivative(hidden2_3);
            double hidden2_4_error = error_output * w_output4 * sigmoid_derivative(hidden2_4);
            double hidden2_5_error = error_output * w_output5 * sigmoid_derivative(hidden2_5);
            double hidden2_6_error = error_output * w_output6 * sigmoid_derivative(hidden2_6);
            double hidden2_7_error = error_output * w_output7 * sigmoid_derivative(hidden2_7);
            double hidden2_8_error = error_output * w_output8 * sigmoid_derivative(hidden2_8);
            double hidden2_9_error = error_output * w_output9 * sigmoid_derivative(hidden2_9);
            double hidden2_10_error = error_output * w_output10 * sigmoid_derivative(hidden2_10);

            // Update hidden layer 2 weights
            w2_1_1 -= learning_rate * hidden2_1_error * hidden1_1 * sigmoid_derivative(hidden2_1);
            w2_1_2 -= learning_rate * hidden2_1_error * hidden1_2 * sigmoid_derivative(hidden2_1);
            w2_1_3 -= learning_rate * hidden2_1_error * hidden1_3 * sigmoid_derivative(hidden2_1);
            w2_1_4 -= learning_rate * hidden2_1_error * hidden1_4 * sigmoid_derivative(hidden2_1);
            w2_1_5 -= learning_rate * hidden2_1_error * hidden1_5 * sigmoid_derivative(hidden2_1);
            w2_1_6 -= learning_rate * hidden2_1_error * hidden1_6 * sigmoid_derivative(hidden2_1);
            w2_1_7 -= learning_rate * hidden2_1_error * hidden1_7 * sigmoid_derivative(hidden2_1);
            w2_1_8 -= learning_rate * hidden2_1_error * hidden1_8 * sigmoid_derivative(hidden2_1);
            w2_1_9 -= learning_rate * hidden2_1_error * hidden1_9 * sigmoid_derivative(hidden2_1);
            w2_1_10 -= learning_rate * hidden2_1_error * hidden1_10 * sigmoid_derivative(hidden2_1);
            bias_2_1 -= learning_rate * hidden2_1_error * sigmoid_derivative(hidden2_1);

            w2_2_1 -= learning_rate * hidden2_2_error * hidden1_1 * sigmoid_derivative(hidden2_2);
            w2_2_2 -= learning_rate * hidden2_2_error * hidden1_2 * sigmoid_derivative(hidden2_2);
            w2_2_3 -= learning_rate * hidden2_2_error * hidden1_3 * sigmoid_derivative(hidden2_2);
            w2_2_4 -= learning_rate * hidden2_2_error * hidden1_4 * sigmoid_derivative(hidden2_2);
            w2_2_5 -= learning_rate * hidden2_2_error * hidden1_5 * sigmoid_derivative(hidden2_2);
            w2_2_6 -= learning_rate * hidden2_2_error * hidden1_6 * sigmoid_derivative(hidden2_2);
            w2_2_7 -= learning_rate * hidden2_2_error * hidden1_7 * sigmoid_derivative(hidden2_2);
            w2_2_8 -= learning_rate * hidden2_2_error * hidden1_8 * sigmoid_derivative(hidden2_2);
            w2_2_9 -= learning_rate * hidden2_2_error * hidden1_9 * sigmoid_derivative(hidden2_2);
            w2_2_10 -= learning_rate * hidden2_2_error * hidden1_10 * sigmoid_derivative(hidden2_2);
            bias_2_2 -= learning_rate * hidden2_2_error * sigmoid_derivative(hidden2_2);

            w2_3_1 -= learning_rate * hidden2_3_error * hidden1_1 * sigmoid_derivative(hidden2_3);
            w2_3_2 -= learning_rate * hidden2_3_error * hidden1_2 * sigmoid_derivative(hidden2_3);
            w2_3_3 -= learning_rate * hidden2_3_error * hidden1_3 * sigmoid_derivative(hidden2_3);
            w2_3_4 -= learning_rate * hidden2_3_error * hidden1_4 * sigmoid_derivative(hidden2_3);
            w2_3_5 -= learning_rate * hidden2_3_error * hidden1_5 * sigmoid_derivative(hidden2_3);
            w2_3_6 -= learning_rate * hidden2_3_error * hidden1_6 * sigmoid_derivative(hidden2_3);
            w2_3_7 -= learning_rate * hidden2_3_error * hidden1_7 * sigmoid_derivative(hidden2_3);
            w2_3_8 -= learning_rate * hidden2_3_error * hidden1_8 * sigmoid_derivative(hidden2_3);
            w2_3_9 -= learning_rate * hidden2_3_error * hidden1_9 * sigmoid_derivative(hidden2_3);
            w2_3_10 -= learning_rate * hidden2_3_error * hidden1_10 * sigmoid_derivative(hidden2_3);
            bias_2_3 -= learning_rate * hidden2_3_error * sigmoid_derivative(hidden2_3);

            w2_4_1 -= learning_rate * hidden2_4_error * hidden1_1 * sigmoid_derivative(hidden2_4);
            w2_4_2 -= learning_rate * hidden2_4_error * hidden1_2 * sigmoid_derivative(hidden2_4);
            w2_4_3 -= learning_rate * hidden2_4_error * hidden1_3 * sigmoid_derivative(hidden2_4);
            w2_4_4 -= learning_rate * hidden2_4_error * hidden1_4 * sigmoid_derivative(hidden2_4);
            w2_4_5 -= learning_rate * hidden2_4_error * hidden1_5 * sigmoid_derivative(hidden2_4);
            w2_4_6 -= learning_rate * hidden2_4_error * hidden1_6 * sigmoid_derivative(hidden2_4);
            w2_4_7 -= learning_rate * hidden2_4_error * hidden1_7 * sigmoid_derivative(hidden2_4);
            w2_4_8 -= learning_rate * hidden2_4_error * hidden1_8 * sigmoid_derivative(hidden2_4);
            w2_4_9 -= learning_rate * hidden2_4_error * hidden1_9 * sigmoid_derivative(hidden2_4);
            w2_4_10 -= learning_rate * hidden2_4_error * hidden1_10 * sigmoid_derivative(hidden2_4);
            bias_2_4 -= learning_rate * hidden2_4_error * sigmoid_derivative(hidden2_4);

            w2_5_1 -= learning_rate * hidden2_5_error * hidden1_1 * sigmoid_derivative(hidden2_5);
            w2_5_2 -= learning_rate * hidden2_5_error * hidden1_2 * sigmoid_derivative(hidden2_5);
            w2_5_3 -= learning_rate * hidden2_5_error * hidden1_3 * sigmoid_derivative(hidden2_5);
            w2_5_4 -= learning_rate * hidden2_5_error * hidden1_4 * sigmoid_derivative(hidden2_5);
            w2_5_5 -= learning_rate * hidden2_5_error * hidden1_5 * sigmoid_derivative(hidden2_5);
            w2_5_6 -= learning_rate * hidden2_5_error * hidden1_6 * sigmoid_derivative(hidden2_5);
            w2_5_7 -= learning_rate * hidden2_5_error * hidden1_7 * sigmoid_derivative(hidden2_5);
            w2_5_8 -= learning_rate * hidden2_5_error * hidden1_8 * sigmoid_derivative(hidden2_5);
            w2_5_9 -= learning_rate * hidden2_5_error * hidden1_9 * sigmoid_derivative(hidden2_5);
            w2_5_10 -= learning_rate * hidden2_5_error * hidden1_10 * sigmoid_derivative(hidden2_5);
            bias_2_5 -= learning_rate * hidden2_5_error * sigmoid_derivative(hidden2_5);

            w2_6_1 -= learning_rate * hidden2_6_error * hidden1_1 * sigmoid_derivative(hidden2_6);
            w2_6_2 -= learning_rate * hidden2_6_error * hidden1_2 * sigmoid_derivative(hidden2_6);
            w2_6_3 -= learning_rate * hidden2_6_error * hidden1_3 * sigmoid_derivative(hidden2_6);
            w2_6_4 -= learning_rate * hidden2_6_error * hidden1_4 * sigmoid_derivative(hidden2_6);
            w2_6_5 -= learning_rate * hidden2_6_error * hidden1_5 * sigmoid_derivative(hidden2_6);
            w2_6_6 -= learning_rate * hidden2_6_error * hidden1_6 * sigmoid_derivative(hidden2_6);
            w2_6_7 -= learning_rate * hidden2_6_error * hidden1_7 * sigmoid_derivative(hidden2_6);
            w2_6_8 -= learning_rate * hidden2_6_error * hidden1_8 * sigmoid_derivative(hidden2_6);
            w2_6_9 -= learning_rate * hidden2_6_error * hidden1_9 * sigmoid_derivative(hidden2_6);
            w2_6_10 -= learning_rate * hidden2_6_error * hidden1_10 * sigmoid_derivative(hidden2_6);
            bias_2_6 -= learning_rate * hidden2_6_error * sigmoid_derivative(hidden2_6);

            w2_7_1 -= learning_rate * hidden2_7_error * hidden1_1 * sigmoid_derivative(hidden2_7);
            w2_7_2 -= learning_rate * hidden2_7_error * hidden1_2 * sigmoid_derivative(hidden2_7);
            w2_7_3 -= learning_rate * hidden2_7_error * hidden1_3 * sigmoid_derivative(hidden2_7);
            w2_7_4 -= learning_rate * hidden2_7_error * hidden1_4 * sigmoid_derivative(hidden2_7);
            w2_7_5 -= learning_rate * hidden2_7_error * hidden1_5 * sigmoid_derivative(hidden2_7);
            w2_7_6 -= learning_rate * hidden2_7_error * hidden1_6 * sigmoid_derivative(hidden2_7);
            w2_7_7 -= learning_rate * hidden2_7_error * hidden1_7 * sigmoid_derivative(hidden2_7);
            w2_7_8 -= learning_rate * hidden2_7_error * hidden1_8 * sigmoid_derivative(hidden2_7);
            w2_7_9 -= learning_rate * hidden2_7_error * hidden1_9 * sigmoid_derivative(hidden2_7);
            w2_7_10 -= learning_rate * hidden2_7_error * hidden1_10 * sigmoid_derivative(hidden2_7);
            bias_2_7 -= learning_rate * hidden2_7_error * sigmoid_derivative(hidden2_7);

            w2_8_1 -= learning_rate * hidden2_8_error * hidden1_1 * sigmoid_derivative(hidden2_8);
            w2_8_2 -= learning_rate * hidden2_8_error * hidden1_2 * sigmoid_derivative(hidden2_8);
            w2_8_3 -= learning_rate * hidden2_8_error * hidden1_3 * sigmoid_derivative(hidden2_8);
            w2_8_4 -= learning_rate * hidden2_8_error * hidden1_4 * sigmoid_derivative(hidden2_8);
            w2_8_5 -= learning_rate * hidden2_8_error * hidden1_5 * sigmoid_derivative(hidden2_8);
            w2_8_6 -= learning_rate * hidden2_8_error * hidden1_6 * sigmoid_derivative(hidden2_8);
            w2_8_7 -= learning_rate * hidden2_8_error * hidden1_7 * sigmoid_derivative(hidden2_8);
            w2_8_8 -= learning_rate * hidden2_8_error * hidden1_8 * sigmoid_derivative(hidden2_8);
            w2_8_9 -= learning_rate * hidden2_8_error * hidden1_9 * sigmoid_derivative(hidden2_8);
            w2_8_10 -= learning_rate * hidden2_8_error * hidden1_10 * sigmoid_derivative(hidden2_8);
            bias_2_8 -= learning_rate * hidden2_8_error * sigmoid_derivative(hidden2_8);

            w2_9_1 -= learning_rate * hidden2_9_error * hidden1_1 * sigmoid_derivative(hidden2_9);
            w2_9_2 -= learning_rate * hidden2_9_error * hidden1_2 * sigmoid_derivative(hidden2_9);
            w2_9_3 -= learning_rate * hidden2_9_error * hidden1_3 * sigmoid_derivative(hidden2_9);
            w2_9_4 -= learning_rate * hidden2_9_error * hidden1_4 * sigmoid_derivative(hidden2_9);
            w2_9_5 -= learning_rate * hidden2_9_error * hidden1_5 * sigmoid_derivative(hidden2_9);
            w2_9_6 -= learning_rate * hidden2_9_error * hidden1_6 * sigmoid_derivative(hidden2_9);
            w2_9_7 -= learning_rate * hidden2_9_error * hidden1_7 * sigmoid_derivative(hidden2_9);
            w2_9_8 -= learning_rate * hidden2_9_error * hidden1_8 * sigmoid_derivative(hidden2_9);
            w2_9_9 -= learning_rate * hidden2_9_error * hidden1_9 * sigmoid_derivative(hidden2_9);
            w2_9_10 -= learning_rate * hidden2_9_error * hidden1_10 * sigmoid_derivative(hidden2_9);
            bias_2_9 -= learning_rate * hidden2_9_error * sigmoid_derivative(hidden2_9);

            // Hidden layer 2: Node 10
            w2_10_1 -= learning_rate * hidden2_10_error * hidden1_1 * sigmoid_derivative(hidden2_10);
            w2_10_2 -= learning_rate * hidden2_10_error * hidden1_2 * sigmoid_derivative(hidden2_10);
            w2_10_3 -= learning_rate * hidden2_10_error * hidden1_3 * sigmoid_derivative(hidden2_10);
            w2_10_4 -= learning_rate * hidden2_10_error * hidden1_4 * sigmoid_derivative(hidden2_10);
            w2_10_5 -= learning_rate * hidden2_10_error * hidden1_5 * sigmoid_derivative(hidden2_10);
            w2_10_6 -= learning_rate * hidden2_10_error * hidden1_6 * sigmoid_derivative(hidden2_10);
            w2_10_7 -= learning_rate * hidden2_10_error * hidden1_7 * sigmoid_derivative(hidden2_10);
            w2_10_8 -= learning_rate * hidden2_10_error * hidden1_8 * sigmoid_derivative(hidden2_10);
            w2_10_9 -= learning_rate * hidden2_10_error * hidden1_9 * sigmoid_derivative(hidden2_10);
            w2_10_10 -= learning_rate * hidden2_10_error * hidden1_10 * sigmoid_derivative(hidden2_10);
            bias_2_10 -= learning_rate * hidden2_10_error * sigmoid_derivative(hidden2_10);

            double hidden1_1_error = (hidden2_1_error * w2_1_1 + hidden2_2_error * w2_2_1 + hidden2_3_error * w2_3_1 + hidden2_4_error * w2_4_1 + hidden2_5_error * w2_5_1 + hidden2_6_error * w2_6_1 + hidden2_7_error * w2_7_1 + hidden2_8_error * w2_8_1 + hidden2_9_error * w2_9_1 + hidden2_10_error * w2_10_1) * sigmoid_derivative(hidden1_1);
            double hidden1_2_error = (hidden2_1_error * w2_1_2 + hidden2_2_error * w2_2_2 + hidden2_3_error * w2_3_2 + hidden2_4_error * w2_4_2 + hidden2_5_error * w2_5_2 + hidden2_6_error * w2_6_2 + hidden2_7_error * w2_7_2 + hidden2_8_error * w2_8_2 + hidden2_9_error * w2_9_2 + hidden2_10_error * w2_10_2) * sigmoid_derivative(hidden1_2);
            double hidden1_3_error = (hidden2_1_error * w2_1_3 + hidden2_2_error * w2_2_3 + hidden2_3_error * w2_3_3 + hidden2_4_error * w2_4_3 + hidden2_5_error * w2_5_3 + hidden2_6_error * w2_6_3 + hidden2_7_error * w2_7_3 + hidden2_8_error * w2_8_3 + hidden2_9_error * w2_9_3 + hidden2_10_error * w2_10_3) * sigmoid_derivative(hidden1_3);
            double hidden1_4_error = (hidden2_1_error * w2_1_4 + hidden2_2_error * w2_2_4 + hidden2_3_error * w2_3_4 + hidden2_4_error * w2_4_4 + hidden2_5_error * w2_5_4 + hidden2_6_error * w2_6_4 + hidden2_7_error * w2_7_4 + hidden2_8_error * w2_8_4 + hidden2_9_error * w2_9_4 + hidden2_10_error * w2_10_4) * sigmoid_derivative(hidden1_4);
            double hidden1_5_error = (hidden2_1_error * w2_1_5 + hidden2_2_error * w2_2_5 + hidden2_3_error * w2_3_5 + hidden2_4_error * w2_4_5 + hidden2_5_error * w2_5_5 + hidden2_6_error * w2_6_5 + hidden2_7_error * w2_7_5 + hidden2_8_error * w2_8_5 + hidden2_9_error * w2_9_5 + hidden2_10_error * w2_10_5) * sigmoid_derivative(hidden1_5);
            double hidden1_6_error = (hidden2_1_error * w2_1_6 + hidden2_2_error * w2_2_6 + hidden2_3_error * w2_3_6 + hidden2_4_error * w2_4_6 + hidden2_5_error * w2_5_6 + hidden2_6_error * w2_6_6 + hidden2_7_error * w2_7_6 + hidden2_8_error * w2_8_6 + hidden2_9_error * w2_9_6 + hidden2_10_error * w2_10_6) * sigmoid_derivative(hidden1_6);
            double hidden1_7_error = (hidden2_1_error * w2_1_7 + hidden2_2_error * w2_2_7 + hidden2_3_error * w2_3_7 + hidden2_4_error * w2_4_7 + hidden2_5_error * w2_5_7 + hidden2_6_error * w2_6_7 + hidden2_7_error * w2_7_7 + hidden2_8_error * w2_8_7 + hidden2_9_error * w2_9_7 + hidden2_10_error * w2_10_7) * sigmoid_derivative(hidden1_7);
            double hidden1_8_error = (hidden2_1_error * w2_1_8 + hidden2_2_error * w2_2_8 + hidden2_3_error * w2_3_8 + hidden2_4_error * w2_4_8 + hidden2_5_error * w2_5_8 + hidden2_6_error * w2_6_8 + hidden2_7_error * w2_7_8 + hidden2_8_error * w2_8_8 + hidden2_9_error * w2_9_8 + hidden2_10_error * w2_10_8) * sigmoid_derivative(hidden1_8);
            double hidden1_9_error = (hidden2_1_error * w2_1_9 + hidden2_2_error * w2_2_9 + hidden2_3_error * w2_3_9 + hidden2_4_error * w2_4_9 + hidden2_5_error * w2_5_9 + hidden2_6_error * w2_6_9 + hidden2_7_error * w2_7_9 + hidden2_8_error * w2_8_9 + hidden2_9_error * w2_9_9 + hidden2_10_error * w2_10_9) * sigmoid_derivative(hidden1_9);
            double hidden1_10_error =(hidden2_1_error * w2_1_10 + hidden2_2_error * w2_2_10 + hidden2_3_error * w2_3_10 + hidden2_4_error * w2_4_10 + hidden2_5_error * w2_5_10 + hidden2_6_error * w2_6_10 + hidden2_7_error * w2_7_10 + hidden2_8_error * w2_8_10 + hidden2_9_error * w2_9_10 + hidden2_10_error * w2_10_10) * sigmoid_derivative(hidden1_10);

            // Update hidden layer 1 weights and biases
            w1_1 -= learning_rate * hidden1_1_error * x1 * sigmoid_derivative(hidden1_1);
            bias_1_1 -= learning_rate * hidden1_1_error * sigmoid_derivative(hidden1_1);

            w1_2 -= learning_rate * hidden1_2_error * x1 * sigmoid_derivative(hidden1_2);
            bias_1_2 -= learning_rate * hidden1_2_error * sigmoid_derivative(hidden1_2);

            w1_3 -= learning_rate * hidden1_3_error * x1 * sigmoid_derivative(hidden1_3);
            bias_1_3 -= learning_rate * hidden1_3_error * sigmoid_derivative(hidden1_3);

            w1_4 -= learning_rate * hidden1_4_error * x1 * sigmoid_derivative(hidden1_4);
            bias_1_4 -= learning_rate * hidden1_4_error * sigmoid_derivative(hidden1_4);

            w1_5 -= learning_rate * hidden1_5_error * x1 * sigmoid_derivative(hidden1_5);
            bias_1_5 -= learning_rate * hidden1_5_error * sigmoid_derivative(hidden1_5);

            w1_6 -= learning_rate * hidden1_6_error * x1 * sigmoid_derivative(hidden1_6);
            bias_1_6 -= learning_rate * hidden1_6_error * sigmoid_derivative(hidden1_6);

            w1_7 -= learning_rate * hidden1_7_error * x1 * sigmoid_derivative(hidden1_7);
            bias_1_7 -= learning_rate * hidden1_7_error * sigmoid_derivative(hidden1_7);

            w1_8 -= learning_rate * hidden1_8_error * x1 * sigmoid_derivative(hidden1_8);
            bias_1_8 -= learning_rate * hidden1_8_error * sigmoid_derivative(hidden1_8);

            w1_9 -= learning_rate * hidden1_9_error * x1 * sigmoid_derivative(hidden1_9);
            bias_1_9 -= learning_rate * hidden1_9_error * sigmoid_derivative(hidden1_9);

            w1_10 -= learning_rate * hidden1_10_error * x1 * sigmoid_derivative(hidden1_10);
            bias_1_10 -= learning_rate * hidden1_10_error * sigmoid_derivative(hidden1_10);

        }
        cout << "Epoch: " << epochs + 1 << " Loss: " << total_loss / data_size << endl;
    }

    double inputs[3] =  {1, 0.5, 0};
    for (double input : inputs) {

        double output1 = neuron(input, 0, 0, 0, 0, 0, 0, 0, 0, 0, w1_1, 0, 0, 0, 0, 0, 0, 0, 0, 0, bias_1_1);
        double output2 = neuron(input, 0, 0, 0, 0, 0, 0, 0, 0, 0, w1_2, 0, 0, 0, 0, 0, 0, 0, 0, 0, bias_1_2);
        double output3 = neuron(input, 0, 0, 0, 0, 0, 0, 0, 0, 0, w1_3, 0, 0, 0, 0, 0, 0, 0, 0, 0, bias_1_3);
        double output4 = neuron(input, 0, 0, 0, 0, 0, 0, 0, 0, 0, w1_4, 0, 0, 0, 0, 0, 0, 0, 0, 0, bias_1_4);
        double output5 = neuron(input, 0, 0, 0, 0, 0, 0, 0, 0, 0, w1_5, 0, 0, 0, 0, 0, 0, 0, 0, 0, bias_1_5);
        double output6 = neuron(input, 0, 0, 0, 0, 0, 0, 0, 0, 0, w1_6, 0, 0, 0, 0, 0, 0, 0, 0, 0, bias_1_6);
        double output7 = neuron(input, 0, 0, 0, 0, 0, 0, 0, 0, 0, w1_7, 0, 0, 0, 0, 0, 0, 0, 0, 0, bias_1_7);
        double output8 = neuron(input, 0, 0, 0, 0, 0, 0, 0, 0, 0, w1_8, 0, 0, 0, 0, 0, 0, 0, 0, 0, bias_1_8);
        double output9 = neuron(input, 0, 0, 0, 0, 0, 0, 0, 0, 0, w1_9, 0, 0, 0, 0, 0, 0, 0, 0, 0, bias_1_9);
        double output10 = neuron(input, 0, 0, 0, 0, 0, 0, 0, 0, 0, w1_10, 0, 0, 0, 0, 0, 0, 0, 0, 0, bias_1_10);


        double output11 = neuron(output1, output2, output3, output4, output5, output6, output7, output8, output9, output10, w2_1_1, w2_1_2, w2_1_3, w2_1_4, w2_1_5, w2_1_6, w2_1_7, w2_1_8, w2_1_9, w2_1_10, bias_2_1);
        double output12 = neuron(output1, output2, output3, output4, output5, output6, output7, output8, output9, output10, w2_2_1, w2_2_2, w2_2_3, w2_2_4, w2_2_5, w2_2_6, w2_2_7, w2_2_8, w2_2_9, w2_2_10, bias_2_2);
        double output13 = neuron(output1, output2, output3, output4, output5, output6, output7, output8, output9, output10, w2_3_1, w2_3_2, w2_3_3, w2_3_4, w2_3_5, w2_3_6, w2_3_7, w2_3_8, w2_3_9, w2_3_10, bias_2_3);
        double output14 = neuron(output1, output2, output3, output4, output5, output6, output7, output8, output9, output10, w2_4_1, w2_4_2, w2_4_3, w2_4_4, w2_4_5, w2_4_6, w2_4_7, w2_4_8, w2_4_9, w2_4_10, bias_2_4);
        double output15 = neuron(output1, output2, output3, output4, output5, output6, output7, output8, output9, output10, w2_5_1, w2_5_2, w2_5_3, w2_5_4, w2_5_5, w2_5_6, w2_5_7, w2_5_8, w2_5_9, w2_5_10, bias_2_5);
        double output16 = neuron(output1, output2, output3, output4, output5, output6, output7, output8, output9, output10, w2_6_1, w2_6_2, w2_6_3, w2_6_4, w2_6_5, w2_6_6, w2_6_7, w2_6_8, w2_6_9, w2_6_10, bias_2_6);
        double output17 = neuron(output1, output2, output3, output4, output5, output6, output7, output8, output9, output10, w2_7_1, w2_7_2, w2_7_3, w2_7_4, w2_7_5, w2_7_6, w2_7_7, w2_7_8, w2_7_9, w2_7_10, bias_2_7);
        double output18 = neuron(output1, output2, output3, output4, output5, output6, output7, output8, output9, output10, w2_8_1, w2_8_2, w2_8_3, w2_8_4, w2_8_5, w2_8_6, w2_8_7, w2_8_8, w2_8_9, w2_8_10, bias_2_8);
        double output19 = neuron(output1, output2, output3, output4, output5, output6, output7, output8, output9, output10, w2_9_1, w2_9_2, w2_9_3, w2_9_4, w2_9_5, w2_9_6, w2_9_7, w2_9_8, w2_9_9, w2_9_10, bias_2_9);
        double output20 = neuron(output1, output2, output3, output4, output5, output6, output7, output8, output9, output10, w2_10_1, w2_10_2, w2_10_3, w2_10_4, w2_10_5, w2_10_6, w2_10_7, w2_10_8, w2_10_9, w2_10_10, bias_2_10);


        double final_output = neuron(output11, output12, output13, output14, output15, output16, output17, output18, output19, output20 ,w_output1, w_output2, w_output3, w_output4, w_output5, w_output6, w_output7, w_output8, w_output9, w_output10, bias_output);

        cout << "Prediction for input " << input  << ": " << final_output << endl;
    }

    return 0;
}
