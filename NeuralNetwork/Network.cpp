#include "Network.h"

Network::Network(const std::vector<size_t>& sizes, const NetworkConfig& config) : config(config), sizes(sizes), biases(sizes.size()), weights(sizes.size()){

	for (size_t layer = 1; layer < sizes.size(); ++layer) {
		Vector new_bias(sizes.at(layer));
		new_bias.fill(FillType::RANDOM);
		biases[layer] = new_bias;

		Matrix new_weight(sizes.at(layer), sizes.at(layer - 1));
		new_weight.fill(FillType::RANDOM);
		weights[layer] = new_weight;
	}
}

Vector Network::feedforward(Vector activations) const {
	// Returns the output of the network
	assert(weights.size() == biases.size());

	for (size_t layer = 1; layer < sizes.size(); ++layer) {
		activations = sigmoid(weights.at(layer) * activations + biases.at(layer));
	}

	return activations;
}

void Network::train(std::vector<ImageTuple> training, const std::vector<ImageTuple>& test, const std::vector<ImageTuple>& validation) {
	for (size_t epoch = 0; epoch < config.epochs; ++epoch) {
		Random::shuffle<ImageTuple>(training);
		const std::vector<std::vector<ImageTuple>> mini_batches = split_training_data(training, config.mini_batch_size);

		for (size_t i = 0; i < mini_batches.size(); ++i) {
			update_mini_batch(mini_batches.at(i), training.size());
		}

		std::pair<size_t, double> training_evaluation = evaluate(training);
		std::pair<size_t, double> validation_evaluation = evaluate(validation);

		std::cout << "Epoch " << epoch + 1 << " of " << config.epochs << ": " << std::endl;

		std::cout << "\tTraining:" << std::endl;
		std::cout << "\t\t" << training_evaluation.first << " / " << training.size() << "\t= " << 100.0 * training_evaluation.first / training.size() << "%" <<  std::endl;
		std::cout << "\t\t" << training_evaluation.second << std::endl;

		std::cout << "\tValidation:" << std::endl;
		std::cout << "\t\t" << validation_evaluation.first << " / " << validation.size() << "\t= " << 100.0 * validation_evaluation.first / validation.size() << "%" << std::endl;
		std::cout << "\t\t" << validation_evaluation.second << std::endl << std::endl;
	}

	std::cout << "Finished" << std::endl;
}

std::pair<size_t, double> Network::evaluate(const std::vector<ImageTuple>& data) {
	size_t correct = 0;
	double summed_cost = 0.0;

	for (size_t i = 0; i < data.size(); ++i) {
		size_t desired_output_value = get_highest_index(data.at(i).desired_output);
		Vector actual_output = feedforward(data.at(i).image_vector);

		if (desired_output_value == get_highest_index(actual_output)) {
			correct++;
		}

		summed_cost += config.cost_function.function(actual_output, data.at(i).desired_output);
	}

	return std::pair<size_t, double>(correct, summed_cost / static_cast<double>(data.size()));
}

void Network::update_mini_batch(const std::vector<ImageTuple>& mini_batch, const size_t& training_size) {
	std::vector<Vector> nabla_B(sizes.size());
	std::vector<Matrix> nabla_W(sizes.size());
	for (size_t layer = 1; layer < sizes.size(); ++layer) {
		nabla_B[layer] = Vector(sizes.at(layer));
		nabla_W[layer] = Matrix(sizes.at(layer), sizes.at(layer - 1));
	}

	for (size_t i = 0; i < mini_batch.size(); ++i) {
		std::pair<std::vector<Vector>, std::vector<Matrix>> delta_nablas = backprop(mini_batch.at(i));
		std::vector<Vector> delta_nabla_B = delta_nablas.first;
		std::vector<Matrix> delta_nabla_W = delta_nablas.second;

		for (size_t layer = 1; layer < sizes.size(); ++layer) {
			nabla_B[layer] = nabla_B.at(layer) + delta_nabla_B.at(layer);
			nabla_W[layer] = nabla_W.at(layer) + delta_nabla_W.at(layer);
		}
	}

	const double learning_constant = config.eta / static_cast<double>(mini_batch.size());
	for (size_t layer = 1; layer < sizes.size(); ++layer) {
		// Learning rule for weights changed due to regularisation
		//weights[layer] = weights.at(layer) - (nabla_W.at(layer) * learning_constant);
		
		const double regularisation_constant = (1.0 - (config.eta * config.lambda / static_cast<double>(training_size)));
		weights[layer] = (weights.at(layer) * regularisation_constant) - (nabla_W.at(layer) * learning_constant);
		biases[layer] = biases.at(layer) - (nabla_B.at(layer) * learning_constant);
	}
}

std::pair<std::vector<Vector>, std::vector<Matrix>> Network::backprop(const ImageTuple& image) {
	std::vector<Vector> nabla_B(sizes.size());
	std::vector<Matrix> nabla_W(sizes.size());
	for (size_t layer = 1; layer < sizes.size(); ++layer) {
		nabla_B[layer] = Vector(sizes.at(layer));
		nabla_W[layer] = Matrix(sizes.at(layer), sizes.at(layer - 1));
	}

	// Step 1: Input
	std::vector<Vector> activations(sizes.size());
	activations[0] = image.image_vector;

	// Step 2: Feedforward
	std::vector<Vector> z_values(sizes.size());
	for (size_t layer = 1; layer < sizes.size(); ++layer) {
		z_values[layer] = weights.at(layer) * activations.at(layer - 1) + biases.at(layer);
		activations[layer] = sigmoid(z_values.at(layer));
	}

	// Step 3: Output Error (Compute Delta Last (delta ^ L) and nabla_B (same thing))
	std::vector<Vector> delta(sizes.size());
	delta[sizes.size() - 1] = config.cost_function.bias_derivative(z_values.at(sizes.size() - 1), activations.at(sizes.size() - 1), image.desired_output);
	nabla_B[sizes.size() - 1] = delta.at(sizes.size() - 1);

	// Step 4: Backpropagate the Error
	for (size_t layer = (sizes.size() - 2); layer > 0; --layer) {
		delta[layer] = Vector::hadamard(weights.at(layer + 1).transpose() * delta.at(layer + 1), sigmoid_prime(z_values.at(layer)));
		nabla_B[layer] = delta.at(layer);
	}

	// Step 5: Output (Find nabla_W)
	for (size_t layer = 1; layer < sizes.size(); ++layer) {
		for (size_t k = 0; k < sizes.at(layer); ++k) {
			for (size_t j = 0; j < sizes.at(layer - 1); ++j) {
				nabla_W[layer][k][j] = activations.at(layer - 1).at(j) * delta.at(layer).at(k);
			}
		}
	}

	return std::pair<std::vector<Vector>, std::vector<Matrix>>(nabla_B, nabla_W);
}
