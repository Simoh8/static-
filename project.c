#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ROWS 10
#define COLS 10

int main() {
    int maze[ROWS][COLS] = {0};
    int start_row, start_col, end_row, end_col;

    // Initialize random seed
    srand(time(NULL));

    // Set start and end positions
    start_row = rand() % ROWS;
    start_col = 0;
    end_row = rand() % ROWS;
    end_col = COLS - 1;

    // Generate maze walls
    for (int i = 0; i < ROWS * COLS / 3; i++) {
        int row = rand() % ROWS;
        int col = rand() % COLS;
        if ((row != start_row || col != start_col) && (row != end_row || col != end_col)) {
            maze[row][col] = 1;
        }
    }

    // Print maze
    for (int row = 0; row < ROWS; row++) {
        for (int col = 0; col < COLS; col++) {
            if (row == start_row && col == start_col) {
                printf("S");
            } else if (row == end_row && col == end_col) {
                printf("E");
            } else if (maze[row][col] == 1) {
                printf("#");
            } else {
                printf(" ");
            }
        }
        printf("\n");
    }

    return 0;
}