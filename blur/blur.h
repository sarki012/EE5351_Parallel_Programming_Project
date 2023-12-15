#ifndef _IMAGE_H_
#define _IMAGE_H_

// Thread block size
// KERNEL_SIZE must be an odd number
#define KERNEL_SIZE 5
#define KS_DIV_2 2
#define IN_TILE_SIZE 16
#define OUT_TILE_SIZE (IN_TILE_SIZE - KERNEL_SIZE + 1)
//#define BLOCK_SIZE (IN_TILE_SIZE + KERNEL_SIZE - 1)


#endif // _VECTORADD_H_

