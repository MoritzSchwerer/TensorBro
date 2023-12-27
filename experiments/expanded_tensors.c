#include <stdio.h>
#include <stdlib.h>

void multiply_tensors(int *data1, int *stride1, int *data2, int *stride2, int *result_data) {
    int shape[] = {2, 2, 3};
    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
            for (int k = 0; k < shape[2]; k++) {
                int idx1 = (i / stride1[0]) * shape[1] * shape[2] + (j / stride1[1]) * shape[2] + (k / stride1[2]);
                int idx2 = (i / stride2[0]) * shape[1] * shape[2] + (j / stride2[1]) * shape[2] + (k / stride2[2]);
                result_data[idx1] = data1[idx1] * data2[idx2];
            }
        }
    }
}

int main() {
    // shape: (2, 2, 3)
    int data1[] = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4};
    int stride1[] = {1, 1, 1};

    // shape: (1, 1, 3)
    int data2[] = {3, 4, 5};
    int stride2[] = {2, 2, 1};

    int result_data[12];

    multiply_tensors(data1, stride1, data2, stride2, result_data);
    for (int i = 0; i < 12; i++) {
        printf("%d ", result_data[i]);
    }
    return 0;
}
