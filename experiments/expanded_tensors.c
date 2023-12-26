#include <stdio.h>
#include <stdlib.h>

int calc_index(int *stride, int *shape, int *index) {
    int idx = 0;
    int mul = 1;
    for (int i = 2; i >= 0; i--) {
        idx += index[i] / stride[i] * mul;
        mul *= shape[i];
        // mul *= (i > 0) ? shape[i] : 1;
    }
    return idx;
}

void multiply_tensors(int *data1, int *stride1, int *data2, int *stride2, int *result_data) {
    int index[] = {0,0,0};
    int shape[] = {2, 2, 3};
    for (int i = 0; i < 2; i++) { // first dim
        index[0] = i;
        for (int j = 0; j < 2; j++) {  // second dim
            index[1] = j;
            for (int k = 0; k < 3; k++) { // third dim
                index[2] = k;
                int idx1 = i*3*2 + j*3 + k;
                int idx2 = k /stride2[2] + (j / stride2[1]) * 3 + (i / stride2[0]) * 3 * 2;
                printf("%d, %d\n", idx2, calc_index(stride2, shape, index));
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
