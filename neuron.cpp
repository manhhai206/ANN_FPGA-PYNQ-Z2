#include <iostream>
#include <cmath>    
#include <cstdlib>
#include <ctime>

using namespace std;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1 - x); 
}

double loss(double y_true, double y_pred) {
    return 0.5 * pow(y_true - y_pred, 2);
}

double loss_derivative(double y_true, double y_pred) {
    return y_pred - y_true;
}

double neuron(double n1, double n2, double n3, double w1, double w2, double w3, double bias) {
    double weighted_sum = (w1 * n1) + (w2 * n2) + (w3 * n3) + bias;
    return sigmoid(weighted_sum);
}

double random_weight() {
    return (rand() % 1001) / 1000.0 - 0.5; 
}

int main() {
    double x_train[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}}; 
    double y_train[4] = {0, 0, 0, 1}; 
    double x_test[4][2] = {{1, 1}, {1, 1}, {0, 0}, {0, 1}}; 

    double learning_rate = 0.5;
    
    srand(time(0));

    double w1_hidden1_1 = random_weight(), w1_hidden2_1 = random_weight(), w1_hidden3_1 = random_weight();
    double w1_hidden1_2 = random_weight(), w1_hidden2_2 = random_weight(), w1_hidden3_2 = random_weight();
    double bias_hidden1 = random_weight(), bias_hidden2 = random_weight(), bias_hidden3 = random_weight();

    double w2_hidden1_1 = random_weight(), w2_hidden1_2 = random_weight(), w2_hidden1_3 = random_weight();
    double w2_hidden2_1 = random_weight(), w2_hidden2_2 = random_weight(), w2_hidden2_3 = random_weight();
    double w2_hidden3_1 = random_weight(), w2_hidden3_2 = random_weight(), w2_hidden3_3 = random_weight();
    double bias_hidden4 = random_weight(), bias_hidden5 = random_weight(), bias_hidden6 = random_weight();

    double w_output1 = random_weight(), w_output2 = random_weight(), w_output3 = random_weight();
    double bias_output = random_weight();

    for (int epochs = 0; epochs < 1000; epochs++) {
        double total_loss = 0;

        for (int i = 0; i < 4; i++) {
            double x1 = x_train[i][0];
            double x2 = x_train[i][1];
            double y = y_train[i];
            
            // Forward pass: Input -> Hidden layer 1
            double hidden1 = neuron(x1, x2, 0, w1_hidden1_1, w1_hidden1_2, 0, bias_hidden1);
            double hidden2 = neuron(x1, x2, 0, w1_hidden2_1, w1_hidden2_2, 0, bias_hidden2);
            double hidden3 = neuron(x1, x2, 0, w1_hidden3_1, w1_hidden3_2, 0, bias_hidden3);

            // Forward pass: Hidden layer 1 -> Hidden layer 2   
            double hidden4 = neuron(hidden1, hidden2, hidden3, w2_hidden1_1, w2_hidden1_2, w2_hidden1_3, bias_hidden4);
            double hidden5 = neuron(hidden1, hidden2, hidden3, w2_hidden2_1, w2_hidden2_2, w2_hidden2_3, bias_hidden5);
            double hidden6 = neuron(hidden1, hidden2, hidden3, w2_hidden3_1, w2_hidden3_2, w2_hidden3_3, bias_hidden6);

            // Forward pass: Hidden layer 3 -> Output
            double final_output = neuron(hidden4, hidden5, hidden6, w_output1, w_output2, w_output3, bias_output);

            double error_output = loss_derivative(y, final_output);
            total_loss += loss(y, final_output); 

            // Backpropagation: Update weights and biases of the output layer
            w_output1   -= learning_rate * error_output * hidden4 * sigmoid_derivative(final_output);
            w_output2   -= learning_rate * error_output * hidden5 * sigmoid_derivative(final_output);
            w_output3   -= learning_rate * error_output * hidden6 * sigmoid_derivative(final_output);
            bias_output -= learning_rate * error_output * sigmoid_derivative(final_output);

            // Backpropagation: Update hidden layer 2 weights
            double hidden4_error = error_output * w_output1 * sigmoid_derivative(hidden4);
            double hidden5_error = error_output * w_output2 * sigmoid_derivative(hidden5);
            double hidden6_error = error_output * w_output3 * sigmoid_derivative(hidden6);

            // Update hidden layer 2 weights
            w2_hidden1_1 -= learning_rate * hidden4_error * hidden1 * sigmoid_derivative(hidden4);
            w2_hidden1_2 -= learning_rate * hidden4_error * hidden2 * sigmoid_derivative(hidden4);
            w2_hidden1_3 -= learning_rate * hidden4_error * hidden3 * sigmoid_derivative(hidden4);
            bias_hidden4 -= learning_rate * hidden4_error * sigmoid_derivative(hidden4);

            w2_hidden2_1 -= learning_rate * hidden5_error * hidden1 * sigmoid_derivative(hidden5);
            w2_hidden2_2 -= learning_rate * hidden5_error * hidden2 * sigmoid_derivative(hidden5);
            w2_hidden2_3 -= learning_rate * hidden5_error * hidden3 * sigmoid_derivative(hidden5);
            bias_hidden5 -= learning_rate * hidden5_error * sigmoid_derivative(hidden5);

            w2_hidden3_1 -= learning_rate * hidden6_error * hidden1 * sigmoid_derivative(hidden6);
            w2_hidden3_2 -= learning_rate * hidden6_error * hidden2 * sigmoid_derivative(hidden6);
            w2_hidden3_3 -= learning_rate * hidden6_error * hidden3 * sigmoid_derivative(hidden6);
            bias_hidden6 -= learning_rate * hidden6_error * sigmoid_derivative(hidden6);

            // Backpropagation: Update hidden layer 1 weights
            double hidden1_error = (hidden4_error * w2_hidden1_1 + hidden5_error * w2_hidden2_1 + hidden6_error * w2_hidden3_1) * sigmoid_derivative(hidden1);
            double hidden2_error = (hidden4_error * w2_hidden1_2 + hidden5_error * w2_hidden2_2 + hidden6_error * w2_hidden3_2) * sigmoid_derivative(hidden2);
            double hidden3_error = (hidden4_error * w2_hidden1_3 + hidden5_error * w2_hidden2_3 + hidden6_error * w2_hidden3_3) * sigmoid_derivative(hidden3);

            // Update hidden layer 1 weights
            w1_hidden1_1 -= learning_rate * hidden1_error * x1 * sigmoid_derivative(hidden1);
            w1_hidden1_2 -= learning_rate * hidden1_error * x2 * sigmoid_derivative(hidden1);
            bias_hidden1 -= learning_rate * hidden1_error * sigmoid_derivative(hidden1);

            w1_hidden2_1 -= learning_rate * hidden2_error * x1 * sigmoid_derivative(hidden2);
            w1_hidden2_2 -= learning_rate * hidden2_error * x2 * sigmoid_derivative(hidden2);
            bias_hidden2 -= learning_rate * hidden2_error * sigmoid_derivative(hidden2);

            w1_hidden3_1 -= learning_rate * hidden3_error * x1 * sigmoid_derivative(hidden3);
            w1_hidden3_2 -= learning_rate * hidden3_error * x2 * sigmoid_derivative(hidden3);
            bias_hidden3 -= learning_rate * hidden3_error * sigmoid_derivative(hidden3);
        }

        cout << "Epoch: " << epochs + 1 << " Loss: " << total_loss / 4 << endl;
    }

    // cout << "\nAfter training:\n";
    // cout << endl;
    // cout << "w1_hidden1_1: " << w1_hidden1_1 << ", w1_hidden1_2: " << w1_hidden1_2 << ", bias_hidden1: " << bias_hidden1 << endl;
    // cout << "w1_hidden2_1: " << w1_hidden2_1 << ", w1_hidden2_2: " << w1_hidden2_2 << ", bias_hidden2: " << bias_hidden2 << endl;
    // cout << "w1_hidden3_1: " << w1_hidden3_1 << ", w1_hidden3_2: " << w1_hidden3_2 << ", bias_hidden3: " << bias_hidden3 << endl;

    // cout << "w2_hidden1_1: " << w2_hidden1_1 << ", w2_hidden1_2: " << w2_hidden1_2 << ", w2_hidden1_3: " << w2_hidden1_3 << ", bias_hidden4: " << bias_hidden4 << endl;
    // cout << "w2_hidden2_1: " << w2_hidden2_1 << ", w2_hidden2_2: " << w2_hidden2_2 << ", w2_hidden2_3: " << w2_hidden2_3 << ", bias_hidden5: " << bias_hidden5 << endl;
    // cout << "w2_hidden3_1: " << w2_hidden3_1 << ", w2_hidden3_2: " << w2_hidden3_2 << ", w2_hidden3_3: " << w2_hidden3_3 << ", bias_hidden6: " << bias_hidden6 << endl;

    // cout << "w_output1: " << w_output1 << ", w_output2: " << w_output2 << ", w_output3: " << w_output3 << ", bias_output: " << bias_output << endl;
    // cout << endl;

    for (int i = 0; i < 4; i++) 
    {
        double x1 = x_test[i][0];
        double x2 = x_test[i][1];

        double output1 = neuron(x1, x2, 0, w1_hidden1_1, w1_hidden1_2, 0, bias_hidden1);
        double output2 = neuron(x1, x2, 0, w1_hidden2_1, w1_hidden2_2, 0, bias_hidden2);
        double output3 = neuron(x1, x2, 0, w1_hidden3_1, w1_hidden3_2, 0, bias_hidden3);

        double output4 = neuron(output1, output2, output3, w2_hidden1_1, w2_hidden1_2, w2_hidden1_3, bias_hidden4);
        double output5 = neuron(output1, output2, output3, w2_hidden2_1, w2_hidden2_2, w2_hidden2_3, bias_hidden5);
        double output6 = neuron(output1, output2, output3, w2_hidden3_1, w2_hidden3_2, w2_hidden3_3, bias_hidden6);

        double final_output = neuron(output4, output5, output6, w_output1, w_output2, w_output3, bias_output);

        cout << "Prediction for input (" << x1 << ", " << x2 << "): " << final_output << endl;
    }

    return 0;
}
