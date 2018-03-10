#ifndef VECTOR_H
#define VECTOR_H
#include <vector>
#include "Helpers.h"

class Vector {
public:
	typedef std::vector<double>::size_type size_type;

public:
	Vector() : values(0) {}
	Vector(const Vector::size_type& size);

	inline Vector::size_type size() const { return values.size(); }
	inline std::vector<double> to_vector() const { return values; }
	inline double at(const Vector::size_type& index) const { return values.at(index); }
	inline void set(const Vector::size_type& index, const double& value) { values[index] = value; }

	void fill(const FillType& fill);
	double magnitude() const;

	Vector operator+(const Vector& other) const;
	Vector operator-(const Vector& other) const;
	Vector operator*(const double& scalar) const;
	Vector operator-() const;

	static double dot(const std::vector<double>& v1, const std::vector<double>& v2);
	static Vector hadamard(const Vector& v1, const Vector& v2);

private:
	std::vector<double> values;
};

#endif
