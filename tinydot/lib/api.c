#include <stdlib.h>
#include <math.h>
#include "tensor.h"

struct Tensor *api_sqrt(const struct Tensor *tensor) {
  struct Tensor *result = tensor_init(tensor->rank, tensor->shape);
  double *data = malloc(tensor->length * sizeof(double));

  for (unsigned int i = 0; i < tensor->length; i++)
    data[i] = sqrt(tensor->data[i]);

  tensor_set(result, data);

  return result;
}

struct Tensor *api_random(unsigned int rank, unsigned int *shape) {
  struct Tensor *tensor = tensor_init(rank, shape);
  
  for (unsigned int i = 0; i < tensor->length; i++) {
    tensor->data[i] = (double)rand() / RAND_MAX;
  }

  return tensor;
}

struct Tensor *api_uniform(unsigned int rank, unsigned int *shape, double low, double high) {
  struct Tensor *tensor = tensor_init(rank, shape);

  for (unsigned int i = 0; i < tensor->length; i++) {
    tensor->data[i] = low + (high - low) * (double)rand() / RAND_MAX;
  }

  return tensor;
}

float api_prod(const struct Tensor *tensor) {
  float res = 1.0f;

  for (unsigned int i = 0; i < tensor->length; ++i) {
    res *= tensor->data[i];
  }

  return res;
}

struct Tensor *zeros(unsigned int rank, unsigned int *shape) {
  struct Tensor *tensor = tensor_init(rank, shape);
  tensor->data = calloc(tensor->length, sizeof(double));

  return tensor;
}

struct Tensor *ones(unsigned int rank, unsigned int *shape) {
  struct Tensor *tensor = tensor_init(rank, shape);

  for (unsigned int i = 0; i < tensor->length; i++)
    tensor->data[i] = 1.0;
  
  return tensor;
}

struct Tensor *maximum(const struct Tensor *t1, const struct Tensor *t2) {
  struct Tensor *result = tensor_init(t1->rank, t1->shape);
  double *data = malloc(t1->length * sizeof(double));

  for (unsigned int i = 0; i < t1->length; i++)
    data[i] = t1->data[i] > t2->data[i] ? t1->data[i] : t2->data[i];

  tensor_set(result, data);

  return result;
}

struct Tensor *maximum_scalar(const struct Tensor *tensor, const int max) {
  struct Tensor *result = tensor_init(tensor->rank, tensor->shape);
  double *data = malloc(tensor->length * sizeof(double));

  for (unsigned int i = 0; i < tensor->length; i++)
    data[i] = tensor->data[i] > max ? tensor->data[i] : max;

  tensor_set(result, data);

  return result;
}

struct Tensor *exponent(const struct Tensor *tensor) {
  struct Tensor *result = tensor_init(tensor->rank, tensor->shape);
  double *data = malloc(tensor->length * sizeof(double));

  for (unsigned int i = 0; i < tensor->length; i++)
    data[i] = exp(tensor->data[i]);

  tensor_set(result, data);

  return result;
}

// (2 , 3 , 4) 
// array([[[ 0,  1,  2,  3],
//         [ 4,  5,  6,  7],
//         [ 8,  9, 10, 11]],

//        [[12, 13, 14, 15],
//         [16, 17, 18, 19],
//         [20, 21, 22, 23]]])

// sum flat array over a given axis
// for example if the array is [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, r, s, t, u, w, x, y, z]
// and axis is 0, then the result is [sum(a, m), sum(b, n), sum(c, o), sum(d, p), sum(e, r), sum(f, s), sum(g, t), sum(h, u), sum(i, w), sum(j, x), sum(k, y), sum(l, z)]
// and axis is 1, then the result is [sum(a, e, i), sum(b, f, j), sum(c, g, k), sum(d, h, l), sum(e, i, m), sum(f, j, n), sum(g, k, o), sum(h, l, p), sum(i, m, r), sum(j, n, s), sum(k, o, t), sum(l, p, u), sum(m, r, w), sum(n, s, x), sum(o, t, y), sum(p, u, z)]
// and axis is 2, then the result is [sum(a, b, c, d), sum(e, f, g, h), sum(i, j, k, l), sum(m, n, o, p), sum(r, s, t, u), sum(w, x, y, z)]
// 0 < axis < rank

struct Tensor *arange(int start, int stop, unsigned int rank, unsigned int *shape, int step) {
  struct Tensor *tensor = tensor_init(rank, shape);
  double *data = malloc(tensor->length * sizeof(double));

  for (unsigned int i = 0; i < stop; i++)
    data[i] = start + i * step;

  tensor_set(tensor, data);

  return tensor;
}

// struct Tensor *linspace(double start, double stop, unsigned int num) {
//   struct Tensor *tensor = tensor_init(1, &num);
//   double *data = malloc(num * sizeof(double));

//   for (unsigned int i = 0; i < num; i++)
//     data[i] = start + (stop - start) * (double)i / (double)(num - 1);

//   tensor_set(tensor, data);

//   return tensor;
// }
