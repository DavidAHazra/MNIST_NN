#include "Matrix.h"

Matrix::Matrix(const Matrix::size_type& row_count, const Matrix::size_type& column_count) : values(row_count) {
	for (Matrix::size_type index = 0; index < values.size(); ++index) {
		values[index].resize(column_count);
	}
}

void Matrix::fill(const FillType& fill) {
	switch (fill) {
		case (FillType::ZERO) : {
			for (Matrix::size_type index = 0; index < values.size(); ++index) {
				std::fill(values[index].begin(), values[index].end(), 0.0);
			}

			break;
		}

		case (FillType::RANDOM) : {
			for (Matrix::size_type i = 0; i < values.size(); ++i) {
				for (Matrix::size_type j = 0; j < values.at(i).size(); ++j) {
					values[i][j] = Random::get_gaussian_distribution(0.0, 1.0 / std::sqrt(this->values.at(0).size()));
				}
			}

			break;
		}
	}
}

Matrix Matrix::transpose() const {
	Matrix result(values.at(0).size(), values.size());

	for (Matrix::size_type row = 0; row < values.size(); ++row) {
		for (Matrix::size_type col = 0; col < values.at(row).size(); ++col) {
			result[col][row] = values.at(row).at(col);
		}
	}

	return result;
}

double Matrix::sum() const {
	double sum = 0.0;
	for (Matrix::size_type i = 0; i < this->values.size(); ++i) {
		for (Matrix::size_type j = 0; j < this->values.at(i).size(); ++j) {
			sum += this->values.at(i).at(j);
		}
	}

	return sum;
}

Matrix Matrix::operator+(const Matrix& other) const {
	assert((this->values.size() == other.values.size()) && (this->values.at(0).size() == other.values.at(0).size()));

	Matrix result(this->values.size(), this->values.at(0).size());
	for (Matrix::size_type i = 0; i < this->values.size(); ++i) {
		for (Matrix::size_type j = 0; j < this->values.at(i).size(); ++j) {
			result[i][j] = this->values.at(i).at(j) + other.values.at(i).at(j);
		}
	}

	return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
	assert((this->values.size() == other.values.size()) && (this->values.at(0).size() == other.values.at(0).size()));

	Matrix result(this->values.size(), this->values.at(0).size());
	for (Matrix::size_type i = 0; i < this->values.size(); ++i) {
		for (Matrix::size_type j = 0; j < this->values.at(i).size(); ++j) {
			result[i][j] = this->values.at(i).at(j) - other.values.at(i).at(j);
		}
	}

	return result;
}

Vector Matrix::operator*(const Vector& vector) const {
	assert((this->values.size() > 0) && (vector.size() > 0));
	assert(this->values.at(0).size() == vector.size());

	Vector result(this->values.size());
	for (Vector::size_type row = 0; row < this->values.size(); ++row) {
		result.set(row, Vector::dot(vector.to_vector(), this->values.at(row)));
	}

	return result;
}

Matrix Matrix::operator*(const double& scalar) const {
	assert(this->values.size() > 0);

	Matrix result(this->values.size(), this->values.at(0).size());
	for (Matrix::size_type i = 0; i < this->values.size(); ++i) {
		for (Matrix::size_type j = 0; j < this->values.at(i).size(); ++j) {
			result[i][j] = this->values.at(i).at(j) * scalar;
		}
	}

	return result;
}