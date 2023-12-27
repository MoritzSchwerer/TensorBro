#include <stdio.h>
#include <stdlib.h>


void reduce_tensor_dim(int *out, int *data, int dim) {
    int shape[] = {2, 2, 3};
    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
            for (int k = 0; k < shape[2]; k++) {
                int idx = i * 2 * 3 + j * 3 + k;
                int redIdx;
                if (dim == 0) {
                    // remove i and nothing else
                    redIdx = j * 3 + k;
                } else if (dim == 1) {
                    // remove j and j factor (2)
                    redIdx = i * 3 + k;
                } else if (dim == 2) {
                    // remove k and k factor (3)
                    redIdx = i * 2 + j;
                }
                out[redIdx] += data[idx];
            }
        }
    }
}

void reduce_tensor_dim(int *out, int *data, int dim) {
    int shape[] = {2, 2, 3, 4};
    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
            for (int k = 0; k < shape[2]; k++) {
                for (int l = 0; l < shape[3]; l++) {
                    int idx = i * 2 * 3 * 4 + j * 3 * 4 + k * 4 + l;
                    int redIdx;
                    if (dim == 0) {
                        // remove i and nothing else
                        redIdx = j * 3 * 4 + k * 4 + l;
                    } else if (dim == 1) {
                        // remove j and j factor (2)
                        redIdx = i * 3 * 4 + k * 4 + l;
                    } else if (dim == 2) {
                        // remove k and k factor (3)
                        redIdx = i * 2 * 4 + j * 4 + l;
                    } else if (dim == 3) {
                        // remove l and l factor (4)
                        redIdx = i * 2 * 3 + j * 3 + k;
                    }
                    out[redIdx] += data[idx];
                }
            }
        }
    }
}

void reduce_tensor_dim0(int *out, int *data) {
    int shape[] = {2, 2, 3};
    for (int j = 0; j < shape[1]; j++) {
        for (int k = 0; k < shape[2]; k++) {
            for (int i = 0; i < shape[0]; i++) {
                int idx = i * 2 * 3 + j * 3 + k;
                int redIdx = j * 3 + k; 
                out[redIdx] += data[idx];
            }
        }
    }
}

void reduce_tensor_dim1(int *out, int *data) {
    int shape[] = {2, 2, 3};
    for (int i = 0; i < shape[0]; i++) {
        for (int k = 0; k < shape[2]; k++) {
            for (int j = 0; j < shape[1]; j++) {
                int idx = i * 2 * 3 + j * 3 + k;
                int redIdx = i * 3 + k; 
                out[redIdx] += data[idx];
            }
        }
    }
}

void reduce_tensor_dim2(int *out, int *data) {
    int shape[] = {2, 2, 3};
    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
            for (int k = 0; k < shape[2]; k++) {
                int idx = i * 2 * 3 + j * 3 + k;
                int redIdx = i * 2 + j; 
                out[redIdx] += data[idx];
            }
        }
    }
}


int main() {
    // shape: (2, 2, 3)
    int data[] = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4};
    int shape[] = {2, 2, 3};

    int out[] = {0, 0, 0, 0, 0, 0};
    // int out[] = {0, 0, 0, 0};

    reduce_tensor_dim(out, data, 0);
    for (int i = 0; i < 6; i++) {
        printf("%d ", out[i]);
    }
    return 0;
}
