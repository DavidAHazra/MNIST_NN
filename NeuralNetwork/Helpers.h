#ifndef HELPERS_H
#define HELPERS_H

#define _USE_MATH_DEFINES

#include <algorithm>
#include <iostream>
#include <numeric>
#include <cassert>
#include <random>
#include <time.h>
#include <math.h>
#include <tuple>

//
//
//	Enums
//
//

enum FillType {
	ZERO,
	RANDOM
};

//
//
//	Classes
//
//

class Random {
public:
	Random(const Random& other) = delete;
	Random& operator=(const Random& other) = delete;

	static Random* get_instance() {
		static Random instance;
		return &instance;
	}

public:
	static double get_gaussian_distribution(const double& mean, const double& std_dev) {
		std::normal_distribution<double> distribution(mean, std_dev);
		return distribution(Random::get_instance()->generator);
	}

	template <typename T>
	static void shuffle(std::vector<T>& vector) {
		std::shuffle(std::begin(vector), std::end(vector), Random::get_instance()->generator);
	}

private:
	Random() { generator.seed(static_cast<unsigned int>(time(nullptr))); }

	std::default_random_engine generator;
};


class FileSystem {
public:
	FileSystem(const FileSystem& other) = delete;
	FileSystem& operator=(const FileSystem& other) = delete;

	static FileSystem* get_instance() {
		static FileSystem instance;
		return &instance;
	}

	static inline std::string get() { return get_instance()->path; }
	static inline void set(const std::string& new_path) { get_instance()->path = new_path; }

	static inline std::string get_directory() {
		return get_instance()->get().replace(get_instance()->get().find("NeuralNetwork.exe"), std::string("NeuralNetwork.exe").size(), "");
	}

private:
	std::string path;

	FileSystem() {}
};

//
//
//	Functions
//
//

static inline double sigmoid(const double& x) {
	return 1.0 / (1.0 + std::exp(-x));
}

static double sigmoid_prime(const double& x) {
	const double sigmoid_x = sigmoid(x);
	return sigmoid_x * (1.0 - sigmoid_x);
}

static double ln(const double& x) {
	return std::log(x);
}

static int convert_to_big_endian(const char* buffer) {
	return 
		static_cast<int>(buffer[3]) |
		static_cast<int>(buffer[2]) << 8 |
		static_cast<int>(buffer[1]) << 16 |
		static_cast<int>(buffer[0]) << 24;
}

#endif