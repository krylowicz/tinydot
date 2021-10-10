#pragma once

struct Tensor {
  unsigned int rank;
  unsigned int length;
  unsigned int *shape;
  double *data;
};

struct Tensor *tensor_init(unsigned int rank, unsigned int *shape);
void tensor_destroy(struct Tensor *tensor);
struct Tensor *tensor_copy(struct Tensor *tensor);
void tensor_set(struct Tensor *tensor, double *data);
struct Tensor *tensor_add(struct Tensor *t1, struct Tensor *t2);
struct Tensor *zeros(unsigned int rank, unsigned int *shape);
struct Tensor *ones(unsigned int rank, unsigned int *shape);
