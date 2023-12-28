#include <stdio.h>
#include <stdlib.h>


void permute_tensor(int *out, int *data) {
    for (int a = 0; a < 2; a++) {
        for (int b = 0; b < 4; b++) {
            for (int c = 0; c < 3; c++) {
                int idx = a * 4 * 3 + b * 3 + c;
                int newIdx = a * 4 * 3 + c * 4 + b;
                out[newIdx] = data[idx];
            }
        }
    }
}

int main() {
    // shape: (2, 4, 3)
    int data[] = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8};
    const int size = 24;
    // perm (0, 2, 1)
    // to shape (2, 3, 4)

    int out[size];

    permute_tensor(out, data);
    printf("data: ");
    for (int i = 0; i < size; i++) {
        printf("%d ", data[i]);
    }
    printf("\n");
    printf("out:  ");
    for (int i = 0; i < size; i++) {
        printf("%d ", out[i]);
    }
    printf("\n");
    return 0;
}
