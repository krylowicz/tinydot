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

void tensor_set(struct Tensor *tensor, double *data) {
  for (unsigned int i = 0; i < tensor->length; i++)
    tensor->data[i] = data[i];
}

