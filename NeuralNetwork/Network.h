#ifndef NETWORK_H
#define NETWORK_H
#include "RequiresVector.h"
#include "Matrix.h"


struct NetworkConfig {
	double eta;		// Learning Rate
	double lambda;	// Regularisation Parameter
	size_t epochs;
	size_t mini_batch_size;

	CostFunction cost_function;
};


class Network {
public:
	Network(const std::vector<size_t>& sizes, const NetworkConfig& config);
	void train(std::vector<ImageTuple> training, const std::vector<ImageTuple>& test, const std::vector<ImageTuple>& validation);

private:
	Vector feedforward(Vector input_activations) const;
	std::pair<size_t, double> evaluate(const std::vector<ImageTuple>& validation_data);
	
	void update_mini_batch(const std::vector<ImageTuple>& mini_batch, const size_t& training_size);
	std::pair<std::vector<Vector>, std::vector<Matrix>> backprop(const ImageTuple& image);
	
private:
	NetworkConfig config;
	std::vector<size_t> sizes;
	std::vector<Vector> biases;
	std::vector<Matrix> weights;
};

#endif
