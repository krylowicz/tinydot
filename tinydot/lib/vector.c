#include <stdlib.h>
#include "tensor.h"

double vector_dot(struct Tensor *t1, struct Tensor *t2) {
  double result = 0.0;

  for (unsigned int i = 0; i < t1->length; i++)
    result += t1->data[i] * t2->data[i];

  return result;
}

struct Tensor *vector_sub(struct Tensor *t1, struct Tensor *t2) {
  struct Tensor *result = tensor_init(t1->rank, t1->shape);
  double *data = calloc(t1->length, sizeof(double));

  for (unsigned int i = 0; i < t1->length; i++)
    data[i] = t1->data[i] - t2->data[i];

  tensor_set(result, data);

  return result;
}