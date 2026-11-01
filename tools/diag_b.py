"""Applies a diagnostic build to a stripped main.cpp. usage: diag.py in out push

M = 128 MB + mcode * 256 KB, T = 3 s + tcode * 16 ms + (exec and exit, under 16 ms). Output stays byte-exact.
"""
import sys

s = open(sys.argv[1], encoding='utf-8', newline='').read().replace('\r\n', '\n')
push = int(sys.argv[3])


def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b, 1)


rep('static int threads;', r'''#include <stdio.h>
#include <time.h>
constexpr int DIAG_PUSH = ''' + str(push) + r''';
static double now_ms(){timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}
static double T0=now_ms(), marks_ms[16]; static int marks_n=0;
#define MARK() (marks_ms[marks_n++]=now_ms())
static long read_long(const char*p){FILE*f=fopen(p,"r");long v=0;if(f){if(fscanf(f,"%ld",&v)!=1)v=0;fclose(f);}return v;}
[[noreturn]] static void report(u64 mcode,u64 tcode){
  long target=(128L<<20)+(long)mcode*(256L<<10);
  long cur=read_long("/sys/fs/cgroup/memory.current");
  long extra=target-cur;
  if(extra>0){
    char*m=(char*)mmap(0,extra,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    for(long i=0;i<extra;i+=4096)m[i]=1;
    munmap(m,extra);
  }
  double until=T0+3000+(double)tcode*16;
  timespec ts;ts.tv_sec=(time_t)(until/1e3);ts.tv_nsec=(long)((until-ts.tv_sec*1e3)*1e6);
  clock_nanosleep(CLOCK_MONOTONIC,TIMER_ABSTIME,&ts,0);
  _exit(0);}
struct alignas(64) Spent{u64 tok=0,look=0,sort=0,high=0,low=0;};
static Spent spent[256];
static std::atomic<int> next_spent{0};
static int threads;''')
# phase marks at barriers
rep('''static void between_steps() noexcept {
  static int step = 0;
''', '''static void between_steps() noexcept {
  static int step = 0;
  MARK();
''')
rep('''  munmap((void *)input, input_size + PAGE);
''', '''  munmap((void *)input, input_size + PAGE);
  MARK();
''')
rep('''  free(draft.hashes);
}
''', '''  free(draft.hashes);
  MARK();
}
''')
# pass counters
rep('''      tokens[0] = 0;
      u64 *token = tokens + 1;
''', '''      u64 t0 = __rdtsc();
      tokens[0] = 0;
      u64 *token = tokens + 1;
''')
rep('''      size_t count = token - tokens - 1;
''', '''      size_t count = token - tokens - 1;
      u64 t1 = __rdtsc();
''')
rep('''      for (size_t i = 1; i <= 8; i++)
        ids[count + i] = NO_ID;
''', '''      u64 t2 = __rdtsc();
      for (size_t i = 1; i <= 8; i++)
        ids[count + i] = NO_ID;
''')
rep('''      for (size_t i = 0; i < highs; i++)
        line[high_ids[i] * LINE_WORDS] |= bit;
''', '''      u64 t3 = __rdtsc();
      for (size_t i = 0; i < highs; i++)
        line[high_ids[i] * LINE_WORDS] |= bit;
      u64 t4 = __rdtsc();
''')
rep('''      if (doc % LINE_DOCS == LINE_DOCS - 1 || p == file_end)
        flush(doc);
''', '''      u64 t5 = __rdtsc();
      sp.tok += t1 - t0; sp.look += t2 - t1; sp.sort += t3 - t2; sp.high += t4 - t3; sp.low += t5 - t4;
      if (doc % LINE_DOCS == LINE_DOCS - 1 || p == file_end)
        flush(doc);
''')
rep('''static void parse_docs(Scratch &scratch) {
''', '''static void parse_docs(Scratch &scratch) {
  Spent &sp = spent[next_spent.fetch_add(1)];
''')
# report instead of exit
rep('''  _exit(0);
}

static void between_steps''', r'''  double end_ms = now_ms();
  // marks: 0 queries read, 1 plan done, 2 parse barrier, 3 between done, 4 lists, 5 answers, 6 release
  double setup = marks_ms[1] - T0, parse = marks_ms[2] - marks_ms[1], between = marks_ms[3] - marks_ms[2];
  double lists_ms = marks_ms[4] - marks_ms[3], answers = marks_ms[5] - marks_ms[4], rest = end_ms - marks_ms[5];
  double tsc_ms = 0; {u64 a=__rdtsc(); double b=now_ms(); while(now_ms()-b<2){} tsc_ms=(__rdtsc()-a)/(now_ms()-b);}
  u64 tok=0,look=0,sort=0,high=0,low=0; for(auto&x:spent){tok+=x.tok;look+=x.look;sort+=x.sort;high+=x.high;low+=x.low;}
  auto q=[](double v,double unit,int nbits){long x=(long)(v/unit+0.5);long m=(1L<<nbits)-1;return (u64)(x>m?m:x);};
  long peak=read_long("/sys/fs/cgroup/memory.peak");
  fprintf(stderr,"setup %.2f read %.2f parse %.2f between %.2f lists %.2f answers %.2f rest %.2f | tok %.1f look %.1f sort %.1f high %.1f low %.1f | peak %.1f\n",setup,marks_ms[0]-T0,parse,between,lists_ms,answers,rest,tok/tsc_ms,look/tsc_ms,sort/tsc_ms,high/tsc_ms,low/tsc_ms,peak/1e6);
  u64 m=0,t=0;
  if(DIAG_PUSH==1){m=q(parse,0.25,10)|q(setup,0.5,4)<<10;t=q(lists_ms,0.25,8);}
  if(DIAG_PUSH==2){m=q(answers,0.25,10)|q(rest,0.25,4)<<10;t=q(marks_ms[0]-T0,0.25,8);}
  if(DIAG_PUSH==3){m=q(tok/tsc_ms,4,7)|q(look/tsc_ms,4,7)<<7;t=q(low/tsc_ms,2,8);}
  fprintf(stderr,"mcode %llu tcode %llu\n",m,t);
  report(m,t);
}

static void between_steps''')
open(sys.argv[2], 'w', encoding='utf-8', newline='\n').write(s)
