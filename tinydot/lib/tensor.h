#pragma once

struct Tensor {
  unsigned int rank;
  unsigned int length;
  unsigned int *shape;
  double *data;
};

struct Tensor *tensor_init(unsigned int rank, unsigned int *shape);
void tensor_destroy(struct Tensor *tensor);
void tensor_set(struct Tensor *tensor, double *data);

