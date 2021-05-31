
#include <time.h>

enum class Clock { CPU_PROC, CPU_THREAD, WALL, TSC };
enum class Unit { NANOS, MICROS, MILLIS, SECS, TSC };

struct Ticks {
  long tic_tsc = 0;
  long toc_tsc = 0;
  struct timespec tic_ts;
  struct timespec toc_ts;
  clockid_t clk_id;

  void Reset() {
    tic_tsc = 0;
    toc_tsc = 0;
  }

  double Elapsed(Unit unit) {
    if (tic_tsc) {
      if (unit == Unit::TSC)
        return (double)(toc_tsc - tic_tsc);

      // Covert tsc to nanos and then return.
    }

    struct timespec t;
    if ((toc_ts.tv_nsec - tic_ts.tv_nsec) < 0) {
      t.tv_sec = toc_ts.tv_sec - tic_ts.tv_sec - 1;
      t.tv_nsec = 1000000000 + toc_ts.tv_nsec - tic_ts.tv_nsec;
    } else {
      t.tv_sec = toc_ts.tv_sec - tic_ts.tv_sec;
      t.tv_nsec = toc_ts.tv_nsec - tic_ts.tv_nsec;
    }

    switch (unit) {
    case Unit::NANOS:
      return (double)(t.tv_sec * 1000000000 + t.tv_nsec);
    case Unit::MICROS:
      return (t.tv_sec * 1000000 + (double)t.tv_nsec / 1000);
    case Unit::MILLIS:
      return (t.tv_sec * 1000 + (double)t.tv_nsec / 1000000);
    case Unit::SECS:
      return (t.tv_sec + (double)t.tv_nsec / 1000000000);
    }

    return -1.0;
  }
};

inline long tsc() { return 0; }

inline void Tic(Ticks* t, Clock clk = Clock::WALL) {
  switch (clk) {
  case Clock::WALL:
    t->clk_id = CLOCK_MONOTONIC;
    break;
  case Clock::CPU_PROC:
    t->clk_id = CLOCK_PROCESS_CPUTIME_ID;
    break;
  case Clock::CPU_THREAD:
    t->clk_id = CLOCK_THREAD_CPUTIME_ID;
    break;
  case Clock::TSC:
    t->tic_tsc = tsc();
    return;
  }

  clock_gettime(t->clk_id, &(t->tic_ts));
}

inline void Toc(Ticks* t) {
  if (t->tic_tsc) {
    t->toc_tsc = tsc();
    return;
  }

  clock_gettime(t->clk_id, &(t->toc_ts));
}
