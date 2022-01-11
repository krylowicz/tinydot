#include <stdlib.h>
#include <math.h>
#include "tensor.h"

double vector_dot(struct Tensor *t1, struct Tensor *t2) {
  double result = 0.0;

  for (unsigned int i = 0; i < t1->length; i++)
    result += t1->data[i] * t2->data[i];

  return result;
}

double vector_norm(struct Tensor *t) {
  double result = 0.0;

  for (unsigned int i = 0; i < t->length; i++)
    result += t->data[i] * t->data[i];

  return pow(result, 0.5);
}

struct Tensor *vector_sub(struct Tensor *t1, struct Tensor *t2) {
  struct Tensor *result = tensor_init(t1->rank, t1->shape);
  double *data = calloc(t1->length, sizeof(double));

  for (unsigned int i = 0; i < t1->length; i++)
    data[i] = t1->data[i] - t2->data[i];

  tensor_set(result, data);

  return result;
}

void vector_rotate(struct Tensor *vector, double theta) {
  double x = vector->data[0];
  double y = vector->data[1];
  theta = theta * (M_PI / 180);
  vector->data[0] = x * cos(theta) + y * sin(theta);
  vector->data[1] = y * cos(theta) - x * sin(theta);
}

double vector_angle(struct Tensor *v1, struct Tensor *v2, unsigned int degrees) {
  double dot = vector_dot(v1, v2);
  double v1_norm = vector_norm(v1);
  double v2_norm = vector_norm(v2);
  double theta = acos(dot / (v1_norm * v2_norm));

  if (degrees)
    theta = theta * (180 / M_PI);

  return theta;
}