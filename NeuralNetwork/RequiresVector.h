#ifndef REQUIRESVECTOR_H
#define REQUIRESVECTOR_H
#include "Vector.h"

#include <fstream>

static Vector sigmoid_prime(const Vector& x);

//
//
//	Cost Functions
//
//

// a = Actual last layer activations
// y = Desired last layer activations

struct MSE {	// Mean square error (quadratic cost)
	static inline double function(const Vector& a, const Vector& y) {
		return std::pow((a - y).magnitude(), 2.0) / 2.0;
	}

	// Bias Derivative
	static inline Vector delta(const Vector& z, const Vector& a, const Vector& y) {
		return Vector::hadamard((a - y), sigmoid_prime(z));
	}
};

struct CEE {	// Cross Entropy Error
	static double function(const Vector& a, const Vector& y) {
		assert(a.size() == y.size());

		double sum = 0.0;
		for (size_t i = 0; i < a.size(); ++i) {
			sum += (y.at(i) * ln(a.at(i))) + ((1.0 - y.at(i)) * ln(1.0 - a.at(i)));
		}

		return (-sum) / static_cast<double>(a.size());
	}

	static inline Vector delta(const Vector& z, const Vector& a, const Vector& y) {
		return a - y;
	}
};

typedef double(*Function)(const Vector& a, const Vector& y);
typedef Vector(*Delta)(const Vector& z, const Vector& a, const Vector& y);

struct CostFunction {
	Function function;
	Delta bias_derivative;
};

static const CostFunction Quadratic{ MSE::function, MSE::delta };
static const CostFunction CrossEntropy{ CEE::function, CEE::delta };

//
//
//	Misc.
//
//

struct ImageTuple {
	Vector image_vector;
	Vector desired_output;
};

static Vector apply(const Vector& x, double (*func)(const double&)) {
	Vector result(x.size());
	for (Vector::size_type index = 0; index < x.size(); ++index) {
		result.set(index, func(x.at(index)));
	}

	return result;

}

static Vector sigmoid(const Vector& x) {
	return apply(x, &sigmoid);
}

static Vector sigmoid_prime(const Vector& x) {
	return apply(x, &sigmoid_prime);
}

static Vector ln(const Vector& x) {
	return apply(x, &ln);
}

static std::vector<std::vector<ImageTuple>> split_training_data(const std::vector<ImageTuple>& training_data, const std::vector<ImageTuple>::size_type& mini_batch_size) {
	std::vector<std::vector<ImageTuple>> mini_batches;

	for (std::vector<ImageTuple>::size_type i = 0; i < training_data.size(); i += mini_batch_size) {
		mini_batches.push_back(std::vector<ImageTuple>(training_data.begin() + i, training_data.begin() + i + mini_batch_size));
	}

	return mini_batches;
}

static size_t get_highest_index(const Vector& vector) {
	std::vector<double> vector_values = vector.to_vector();
	return std::distance(vector_values.begin(), std::max_element(vector_values.begin(), vector_values.end()));
}

static std::vector<Vector> load_image_data(const std::string& file_name) {
	std::fstream image_file(FileSystem::get_directory() + file_name, std::ios::in | std::ios::binary);
	if (!image_file.is_open()) {
		throw std::runtime_error("Failed to open " + file_name);
	}

	// 4 Bytes == 32 Bit Integer
	char magic_number_buffer[4];
	char image_count_buffer[4];
	char row_count_buffer[4];
	char column_count_buffer[4];

	image_file.read(magic_number_buffer, 4);
	image_file.read(image_count_buffer, 4);
	image_file.read(row_count_buffer, 4);
	image_file.read(column_count_buffer, 4);

	int magic_number = convert_to_big_endian(magic_number_buffer);
	unsigned short image_count = convert_to_big_endian(image_count_buffer);
	int image_row_count = convert_to_big_endian(row_count_buffer);
	int image_column_count = convert_to_big_endian(column_count_buffer);

	std::vector<Vector> images;
	char current_char[1];

	for (unsigned short image = 0; image < image_count; ++image) {
		Vector current_image(image_row_count * image_column_count);

		for (int pixel = 0; pixel < (image_row_count * image_column_count); ++pixel) {
			image_file.read(current_char, 1);
			unsigned int value = static_cast<unsigned int>(static_cast<unsigned char>(current_char[0]));
			current_image.set(pixel, value);
		}

		images.push_back(current_image);
	}

	return images;
}

static std::vector<Vector> load_label_data(const std::string& file_name) {
	std::fstream image_file(FileSystem::get_directory() + file_name, std::ios::in | std::ios::binary);
	if (!image_file.is_open()) {
		throw std::runtime_error("Failed to open " + file_name);
	}

	// 4 Bytes == 32 Bit Integer
	char magic_number_buffer[4];
	char image_count_buffer[4];

	image_file.read(magic_number_buffer, 4);
	image_file.read(image_count_buffer, 4);

	int magic_number = convert_to_big_endian(magic_number_buffer);
	unsigned short image_count = convert_to_big_endian(image_count_buffer);

	std::vector<Vector> images;
	char current_char[1];

	for (unsigned short label = 0; label < image_count; ++label) {
		Vector current_image(10);

		image_file.read(current_char, 1);
		unsigned int value = static_cast<unsigned int>(static_cast<unsigned char>(current_char[0]));
		current_image.set(value, 1.0);
		images.push_back(current_image);
	}

	return images;
}

static std::tuple<std::vector<ImageTuple>, std::vector<ImageTuple>, std::vector<ImageTuple>> load_data() {
	std::vector<Vector> training_image_vectors = load_image_data("training_images");
	std::cout << "Loaded training image vectors" << std::endl;

	std::vector<Vector> training_label_vectors = load_label_data("training_labels");
	std::cout << "Loaded training label vectors" << std::endl;

	std::vector<Vector> validation_image_vectors = load_image_data("validation_images");
	std::cout << "Loaded validation image vectors" << std::endl;

	std::vector<Vector> validation_label_vectors = load_label_data("validation_labels");
	std::cout << "Loaded validation label vectors" << std::endl << std::endl;

	assert(training_image_vectors.size() == training_label_vectors.size());
	assert(validation_image_vectors.size() == validation_label_vectors.size());

	std::vector<ImageTuple> training_data;
	std::vector<ImageTuple> test_data;
	std::vector<ImageTuple> validation_data;
	for (size_t i = 0; i < training_image_vectors.size(); ++i) {
		ImageTuple image_tuple = { training_image_vectors.at(i), training_label_vectors.at(i) };
		if (i < 55000) {
			training_data.push_back(image_tuple);
		} else {
			test_data.push_back(image_tuple);
		}
	}

	for (size_t i = 0; i < validation_image_vectors.size(); ++i) {
		validation_data.push_back({ validation_image_vectors.at(i), validation_label_vectors.at(i) });
	}

	return std::make_tuple(training_data, test_data, validation_data);
}

#endif
