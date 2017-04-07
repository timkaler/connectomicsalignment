

#ifndef SIMPLE_MUTEX_T_H
#define SIMPLE_MUTEX_T_H
typedef volatile int64_t simple_mutex_t;


static void simple_mutex_init(simple_mutex_t* m) {
  *m = 0;
}

static void simple_acquire(simple_mutex_t* m) {
  while (!__sync_bool_compare_and_swap(m, 0, 1)) {
    continue;
  }
}

static void simple_release(simple_mutex_t* m) {
  if(!__sync_bool_compare_and_swap(m,1,0)) {
     printf("Concurrency Error! Mutex released, but the mutex is not currently acquired!\n");
     abort();
  }
}
#endif
