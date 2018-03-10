#ifndef MATRIX_H
#define MATRIX_H
#include "Helpers.h"
#include "Vector.h"

class Matrix {
public:
	typedef std::vector<std::vector<double>>::size_type size_type;

public:
	Matrix() : values(0) {}
	Matrix(const Matrix::size_type& row_count, const Matrix::size_type& column_count);

	void fill(const FillType& fill);
	double sum() const;

	inline std::vector<double>& operator[](int index) { return values[index]; }

	Matrix transpose() const;

	Matrix operator+(const Matrix& other) const;
	Matrix operator-(const Matrix& other) const;
	Vector operator*(const Vector& vector) const;
	Matrix operator*(const double& scalar) const;

private:
	std::vector<std::vector<double>> values;
};

/*

[ [x, x, x],
  [x, x, x],
  [x, x, x] ]

Row Count = Number of inner arrays
Column Count = Number of items in each inner array

*/

#endif