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

void sum(const struct Tensor *tensor, const unsigned int axis) {
  // TODO - sum over axis
}