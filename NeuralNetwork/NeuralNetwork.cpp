#include "Helpers.h"
#include "Network.h"
#include "RequiresVector.h"


void run_network() {
	try {
		std::tuple<std::vector<ImageTuple>, std::vector<ImageTuple>, std::vector<ImageTuple>> all_data = load_data();
		std::vector<ImageTuple> training_data = std::get<0>(all_data);
		std::vector<ImageTuple> test_data = std::get<1>(all_data);
		std::vector<ImageTuple> validation_data = std::get<2>(all_data);

		NetworkConfig config{
			0.001,			// Learning Rate (eta)
			5.0,			// Regularisation Parameter (lambda)
			150,			// Epoch Count
			5,				// Mini-Batch Size
			CrossEntropy	// Cost Function
		};

		Network network({ 28 * 28, 64, 64, 10 }, config);
		network.train(training_data, test_data, validation_data);

	} catch (const std::exception& exception) {
		std::cout << exception.what() << std::endl;
	}
}

int main(int argc, char* argv[]) {
	if (argc < 1) { std::cout << "Less than one argument? Idk how this happened" << std::endl; return EXIT_FAILURE; }
	FileSystem::set(argv[0]);

	run_network();

	std::cin.get();
    return EXIT_SUCCESS;
}

