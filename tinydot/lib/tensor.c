#include <stdlib.h>
#include "tensor.h"

struct Tensor *tensor_init(unsigned int rank, unsigned int *shape) {
  struct Tensor *tensor = malloc(sizeof(struct Tensor));

  tensor->rank = rank;
  tensor->shape = malloc(rank * sizeof(unsigned int));

  tensor->length = 1;
  for (unsigned int i = 0; i < rank; i++) {
    tensor->shape[i] = shape[i];
    tensor->length *= shape[i];
  }

  tensor->data = malloc(tensor->length * sizeof(double));
  return tensor;
}

void tensor_destroy(struct Tensor *tensor) {
  free(tensor);
}

struct Tensor *tensor_copy(struct Tensor *tensor) {
  struct Tensor *new_tensor = zeros(tensor->rank, tensor->shape);
  new_tensor->data = tensor->data;
  return new_tensor;
}

void tensor_set(struct Tensor *tensor, double *data) {
  for (unsigned int i = 0; i < tensor->length; i++)
    tensor->data[i] = data[i];
}

struct Tensor *tensor_add(struct Tensor *t1, struct Tensor *t2) {
  struct Tensor *result = tensor_init(t1->rank, t1->shape);
  double *data = malloc(t1->length * sizeof(double));

  for (unsigned int i = 0; i < t1->length; i++)
    data[i] = t1->data[i] + t2->data[i]; 

  tensor_set(result, data);

  return result;
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

