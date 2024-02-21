// Author: APD team, except where source was noted

#include "helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>

#define CONTOUR_CONFIG_COUNT    16
#define FILENAME_MAX_SIZE       50
#define STEP                    8
#define SIGMA                   200
#define RESCALE_X               2048
#define RESCALE_Y               2048

#define CLAMP(v, min, max) if(v < min) { v = min; } else if(v > max) { v = max; }

// Creates a map between the binary configuration (e.g. 0110_2) and the corresponding pixels
// that need to be set on the output image. An array is used for this map since the keys are
// binary numbers in 0-15. Contour images are located in the './contours' directory.
ppm_image **init_contour_map() {
    ppm_image **map = (ppm_image **)malloc(CONTOUR_CONFIG_COUNT * sizeof(ppm_image *));
    if (!map) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        char filename[FILENAME_MAX_SIZE];
        sprintf(filename, "./contours/%d.ppm", i);
        map[i] = read_ppm(filename);
    }

    return map;
}

// Updates a particular section of an image with the corresponding contour pixels.
// Used to create the complete contour image.
void update_image(ppm_image *image, ppm_image *contour, int x, int y) {
    for (int i = 0; i < contour->x; i++) {
        for (int j = 0; j < contour->y; j++) {
            int contour_pixel_index = contour->x * i + j;
            int image_pixel_index = (x + i) * image->y + y + j;

            image->data[image_pixel_index].red = contour->data[contour_pixel_index].red;
            image->data[image_pixel_index].green = contour->data[contour_pixel_index].green;
            image->data[image_pixel_index].blue = contour->data[contour_pixel_index].blue;
        }
    }
}

// Calls `free` method on the utilized resources.
void free_resources(ppm_image *image, ppm_image **contour_map, unsigned char **grid, int step_x) {
    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        free(contour_map[i]->data);
        free(contour_map[i]);
    }
    free(contour_map);

    for (int i = 0; i <= image->x / step_x; i++) {
        free(grid[i]);
    }
    free(grid);

    free(image->data);
    free(image);
}

typedef struct {
    ppm_image *image;
    ppm_image *scaled_image;
    unsigned char **grid;
    ppm_image **contour_map;
    int thread_count;
    int thread_id;
    pthread_barrier_t *barrier;
    const char *filename;
    int ok;
} STRUCT_DATA;

void *f(void *arg) {
    uint8_t sample[3];

    // variabila unde sunt stocate datele pentru thread-uri
    STRUCT_DATA *data = (STRUCT_DATA *)arg;

    int thread_id = data->thread_id;

    int start, end;

    // numarul de thread-uri
    int P = data->thread_count;

    // bariera
    pthread_barrier_t *barrier = ((STRUCT_DATA *)data)->barrier;

    ppm_image *image = ((STRUCT_DATA *)data)->image;

    ppm_image *scaled_image = ((STRUCT_DATA *)data)->scaled_image;

    ppm_image **contour_map = ((STRUCT_DATA *)data)->contour_map;

    unsigned char **grid = ((STRUCT_DATA *)data)->grid;

    // fisierul unde se va scrie imaginea
    const char *filename = ((STRUCT_DATA *)data)->filename;

    // variabila pentru a face interpolare bicubica doar la imaginile de dimensiune mare
    int ok = ((STRUCT_DATA *)data)->ok;

    // rescale_image

    if (ok == 0) {
        start = thread_id * ((double)scaled_image->x / P);
        end = scaled_image->x;
        if ((thread_id + 1) * ((double)scaled_image->x / P) < end) end = (thread_id + 1) * ((double)scaled_image->x / P);

        for (int i = start; i < end; i++) {
            for (int j = 0; j < scaled_image->y; j++) {
                float u = (float)i / (float)(scaled_image->x - 1);
                float v = (float)j / (float)(scaled_image->y - 1);
                sample_bicubic(image, u, v, sample);


                scaled_image->data[i * scaled_image->y + j].red = sample[0];
                scaled_image->data[i * scaled_image->y + j].green = sample[1];
                scaled_image->data[i * scaled_image->y + j].blue = sample[2];
            }
        }

        // bariera pentru a ne asigura ca s-au terminat de executat toate thread-urile inainte de a dezaloca memoria imaginii
        pthread_barrier_wait(barrier);

        if (thread_id == 0) {
            free(image->data);
            free(image);
        }
    }

    // sample_grid

    int p_grid = scaled_image->x / STEP;
    int q_grid = scaled_image->y / STEP;

    start = thread_id * ((double)p_grid / P);
    end = p_grid;
    if ((thread_id + 1) * ((double)p_grid / P) < end) end = (thread_id + 1) * ((double)p_grid / P);

    for (int i = start; i < end; i++) {
        for (int j = 0; j < q_grid; j++) {
            ppm_pixel curr_pixel = scaled_image->data[i * STEP * scaled_image->y + j * STEP];

            unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

            if (curr_color > SIGMA) {
                grid[i][j] = 0;
            } else {
                grid[i][j] = 1;
            }
        }
    }

    // este nevoie de un singur thread pentru a stoca acea valoare in matrice
    if (thread_id == 0) {
        grid[p_grid][q_grid] = 0;
    }

    pthread_barrier_wait(barrier);

    // last sample points have no neighbors below / to the right, so we use pixels on the
    // last row / column of the input image for them
    for (int i = start; i < end; i++) {
        ppm_pixel curr_pixel = scaled_image->data[i * STEP * scaled_image->y + scaled_image->x - 1];
        ppm_pixel curr_pixel_j = scaled_image->data[(scaled_image->x - 1) * scaled_image->y + i * STEP];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;
        unsigned char curr_color_j = (curr_pixel_j.red + curr_pixel_j.green + curr_pixel_j.blue) / 3;

        if (curr_color > SIGMA) {
            grid[i][q_grid] = 0;
        } else {
            grid[i][q_grid] = 1;
        }

        if (curr_color_j > SIGMA) {
            grid[p_grid][i] = 0;
        } else {
            grid[p_grid][i] = 1;
        }
    }

    pthread_barrier_wait(barrier);

    // march

    int p = scaled_image->x / STEP;
    int q = scaled_image->y / STEP;

    start = thread_id * ((double)p / P);
    end = p;
    if ((thread_id + 1) * ((double)p / P) < end) end = (thread_id + 1) * ((double)p / P);

    for (int i = start; i < end; i++) {
        for (int j = 0; j < q; j++) {
            unsigned char k = 8 * grid[i][j] + 4 * grid[i][j + 1] + 2 * grid[i + 1][j + 1] + 1 * grid[i + 1][j];
            update_image(scaled_image, contour_map[k], i * STEP, j * STEP);
        }
    }

    pthread_barrier_wait(barrier);

    write_ppm(scaled_image, filename);

    pthread_barrier_wait(barrier);

    if (thread_id == 0) {
        free_resources(scaled_image, contour_map, grid, STEP);
    }

    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: ./tema1 <in_file> <out_file>\n");
        return 1;
    }

    ppm_image *image = read_ppm(argv[1]);

    // 0. Initialize contour map
    ppm_image **contour_map = init_contour_map();

    // numarul de thread-uri
    int thread_count = atoi(argv[3]);

    // se aloca memorie pentru vectorul de thread-uri, structura cu informatiile despre thread-uri si bariera
    pthread_t *threads = (pthread_t *) malloc(thread_count * sizeof(pthread_t));

    STRUCT_DATA *info = (STRUCT_DATA *) malloc(thread_count * sizeof(STRUCT_DATA));

    pthread_barrier_t *barrier = (pthread_barrier_t *) malloc(1 * sizeof(pthread_barrier_t));

    ppm_image *scaled_image;

    int ok = 0;

    // se initiaza bariera
    if (pthread_barrier_init(barrier, NULL, thread_count) != 0)
    {
        printf("Eroare la crearea barierei!\n");
        exit(-1);
    }

    // aceasta secventa de cod va fi executata de thread-ul principal (main)
    scaled_image = (ppm_image *)malloc(sizeof(ppm_image));
    if (image->x <= RESCALE_X && image->y <= RESCALE_Y) {
        scaled_image = image;
        ok = 1;
    } else {
        if (!scaled_image) {
            fprintf(stderr, "Unable to allocate memory\n");
            exit(1);
        }
        scaled_image->x = RESCALE_X;
        scaled_image->y = RESCALE_Y;

        scaled_image->data = (ppm_pixel*) malloc(RESCALE_X * RESCALE_Y * sizeof(ppm_pixel));
        if (!scaled_image) {
            fprintf(stderr, "Unable to allocate memory\n");
            exit(1);
        }
    }

    unsigned char **grid = (unsigned char **)malloc((257) * sizeof(unsigned char*));
    if (!grid) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i <= 256; i++) {
        grid[i] = (unsigned char *)malloc((257) * sizeof(unsigned char));
        if (!grid[i]) {
            fprintf(stderr, "Unable to allocate memory\n");
            exit(1);
        }
    }

    // se creeaza thread-urile
    for (int i = 0; i < thread_count; i++) {
        info[i].thread_id = i;
        info[i].thread_count = thread_count;
        info[i].image = image;
        info[i].scaled_image = scaled_image;
        info[i].contour_map = contour_map;
        info[i].grid = grid;
        info[i].barrier = barrier;
        info[i].filename = argv[2];
        info[i].ok = ok;

        int r = pthread_create(&threads[i], NULL, f, &info[i]);

        if (r) {
            printf("Eroare la crearea thread-ului %d\n", i);
            exit(-1);
        }
    }

    for (int i = 0; i < thread_count; i++) {
        int r = pthread_join(threads[i], NULL);

        if (r) {
            printf("Eroare la asteptarea thread-ului %d\n", i);
            exit(-1);
        }
    }

    // se distruge bariera
    pthread_barrier_destroy(barrier);

    // se elibereaza memoria
    free(barrier);
    free(info);
    free(threads);

    return 0;
}