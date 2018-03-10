#include "Vector.h"

Vector::Vector(const Vector::size_type& size) : values(size) {}

void Vector::fill(const FillType& fill) {
	switch (fill) {
		case (FillType::ZERO) : {
			std::fill(values.begin(), values.end(), 0.0);
			break;
		}

		case (FillType::RANDOM) : {
			for (Vector::size_type index = 0; index < values.size(); ++index) {
				values[index] = Random::get_gaussian_distribution(0.0, 1.0);
			}

			break;
		}
	}
}

double Vector::magnitude() const {
	double sum_of_squares = 0.0;

	for (Vector::size_type index = 0; index < this->values.size(); ++index) {
		sum_of_squares += this->values.at(index) * this->values.at(index);
	}

	return std::sqrt(sum_of_squares);
}

Vector Vector::operator+(const Vector& other) const {
	assert(this->values.size() == other.values.size());

	Vector result(this->values.size());
	for (Vector::size_type index = 0; index < this->values.size(); ++index) {
		result.set(index, this->values.at(index) + other.values.at(index));
	}

	return result;
}

Vector Vector::operator-(const Vector& other) const {
	assert(this->values.size() == other.values.size());

	Vector result(this->values.size());
	for (Vector::size_type index = 0; index < this->values.size(); ++index) {
		result.set(index, this->values.at(index) - other.values.at(index));
	}

	return result;
}

Vector Vector::operator*(const double& scalar) const {
	assert(this->values.size() > 0);

	Vector result(this->values.size());
	for (Vector::size_type index = 0; index < this->values.size(); ++index) {
		result.set(index, this->values.at(index) * scalar);
	}

	return result;
}

Vector Vector::operator-() const {
	Vector result(this->values.size());
	for (Vector::size_type index = 0; index < this->values.size(); ++index) {
		result.set(index, -this->values.at(index));
	}

	return result;
}

double Vector::dot(const std::vector<double>& v1, const std::vector<double>& v2) {
	assert(v1.size() == v2.size());
	
	double result = 0.0;
	for (Vector::size_type index = 0; index < v1.size(); ++index) {
		result += v1.at(index) * v2.at(index);
	}

	return result;
}

Vector Vector::hadamard(const Vector& v1, const Vector& v2) {
	assert(v1.values.size() == v2.values.size());

	Vector result(v1.values.size());
	for (Vector::size_type index = 0; index < v1.values.size(); ++index) {
		result.set(index, v1.values.at(index) * v2.values.at(index));
	}

	return result;
}
