#include <stdlib.h>
#include <math.h>
#include "tensor.h"

double matrix_norm(struct Tensor *tensor) {
  int rows = tensor->shape[0];
  int cols = tensor->shape[1];
  double *res = calloc(cols, sizeof(double));

  for (unsigned int i = 0; i < rows * cols; i += cols) {
    for (unsigned int j = 0; j < cols; j++)
      res[j] += fabs(tensor->data[i + j]);
  }

  for (unsigned int i = 1; i < cols; i++) {
    if (res[0] < res[i])
      res[0] = res[i];
  }

  return res[0];
}

double matrix_trace(struct Tensor *tensor) {
  int rows = tensor->shape[0];
  int cols = tensor->shape[1];
  double trace = 0.0;

  int offset = 0;
  for (unsigned int i = 0; i < rows * cols; i += rows) {
    trace += tensor->data[i + offset];
    offset++;
  }

  return trace;
}

unsigned int *matrix_transpose(struct Tensor *tensor) {
  int rows = tensor->shape[0];
  int cols = tensor->shape[1];
  double *data = calloc(rows * cols, sizeof(double));

  for (unsigned int i = 0; i < cols; i++) {
    for (unsigned int j = 0; j < rows; j++) 
      data[i * rows + j] = tensor->data[j * cols + i];
  }

  tensor->shape[0] = cols;
  tensor->shape[1] = rows;

  free(tensor->data);
  tensor->data = data;

 return tensor->shape;
}

struct Tensor *matrix_identity(unsigned int rank, unsigned int *shape) {
  struct Tensor *matrix = zeros(rank, shape);
  int rows = matrix->shape[0];

  for (unsigned int i = 0; i < rows; i++) 
    matrix->data[i * rows + i] = 1.0;

  return matrix;
}

double matrix_determinant(struct Tensor *tensor) {
  int rows = tensor->shape[0];
  struct Tensor *ct = tensor_copy(tensor);

  for (unsigned int fd = 0; fd < rows; fd++) {
    for (unsigned int i = fd + 1; i < rows; i++) {
      if (ct->data[fd * rows + fd] == 0)
        ct->data[fd * rows + fd] = 1.0e-18;

      double cr = ct->data[i * rows + fd] / ct->data[fd * rows + fd];
      for (unsigned int j = 0; j < rows; j++)
        ct->data[i * rows + j] = ct->data[i * rows + j] - cr * ct->data[fd * rows + j];
    }
  }
  
  double det = 1.0;
  for (unsigned int i = 0; i < rows; i++)
    det *= ct->data[i * rows + i];

  return det;
}

struct Tensor *matrix_inverse(struct Tensor *matrix) { 
  int rows = matrix->shape[0];
  int cols = matrix->shape[1];

  struct Tensor *AM = tensor_copy(matrix);
  struct Tensor *I = matrix_identity(matrix->rank, matrix->shape);

  for (unsigned int fd = 0; fd < rows; ++fd) {
    double fd_scalar = 1.0 / AM->data[fd * rows + fd];

    for (unsigned int j = 0; j < cols; ++j) {
      AM->data[fd * rows + j] *= fd_scalar;
      I->data[fd * rows + j] *= fd_scalar;
    }

    for (unsigned int i = 0; i < rows; ++i) {
      if (i != fd) {
        double cr_scalar = AM->data[i * rows + fd];

        for (unsigned int j = 0; j < cols; ++j) {
          AM->data[i * rows + j] -= cr_scalar * AM->data[fd * rows + j];
          I->data[i * rows + j] -= cr_scalar * I->data[fd * rows + j];
        }
      }
    }
  }

  return I;
}

struct Tensor *matmul(struct Tensor *A, struct Tensor *B) {
  int A_rows = A->shape[0];
  int A_cols = A->shape[1];
  int B_rows = A->shape[0];
  int B_cols = B->shape[1];
  unsigned int shape[2] = {A_rows, B_cols};
  struct Tensor *C = zeros(2, shape);

  for (unsigned int i = 0; i < A_rows; i++) {
    for (unsigned int j = 0; j < B_cols; j++) {
      double total = 0.0;
      for (unsigned int ii = 0; ii < A_cols; ii++)
        total += A->data[i * A_rows + ii] * B->data[ii * B_rows + j];
      C->data[i * A_rows + j] = total;
    }
  }

  return C;
}
