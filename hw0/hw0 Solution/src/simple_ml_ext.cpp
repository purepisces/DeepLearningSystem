#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    // Temporary arrays to store logits and gradients
    std::vector<float> logits(m * k, 0.0);
    std::vector<float> grad(n * k, 0.0);

    // Run through each mini-batch
    for (size_t b = 0; b < m; b += batch) {
        // Zero out the gradient
        std::fill(grad.begin(), grad.end(), 0.0);

        // Size of the current mini-batch
        size_t bsize = std::min(batch, m - b);

        // Compute logits
        for (size_t i = 0; i < bsize; ++i) {
            for (size_t j = 0; j < k; ++j) {
                float sum = 0.0;
                for (size_t l = 0; l < n; ++l) {
                    sum += X[(b + i) * n + l] * theta[l * k + j];
                }
                logits[i * k + j] = sum;
            }
        }

        // Compute gradient
        for (size_t i = 0; i < bsize; ++i) {
            for (size_t j = 0; j < k; ++j) {
                float exp_sum = 0.0;
                for (size_t l = 0; l < k; ++l) {
                    exp_sum += std::exp(logits[i * k + l]);
                }
                float softmax = std::exp(logits[i * k + j]) / exp_sum;
                float delta = (j == y[b + i]) ? 1.0 : 0.0;
                for (size_t l = 0; l < n; ++l) {
                    grad[l * k + j] += (softmax - delta) * X[(b + i) * n + l];
                }
            }
        }

        // Update theta
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < k; ++j) {
                theta[i * k + j] -= lr * grad[i * k + j] / bsize;
            }
        }
    }
}
    /// END YOUR CODE
  


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
