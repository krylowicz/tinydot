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
