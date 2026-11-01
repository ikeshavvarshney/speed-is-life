// Build: g++ -std=c++20 -O2 -march=x86-64-v3 -maes. Names come from hash_check_ac_names.py.
// Read-only risk check of the hashes in tracks A and C, on their real names and on targeted changes.
// usage: acheck <file of names, one per line, escaped> a|c
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
using u8 = uint8_t;
using u64 = unsigned long long;
using v4 = __m256i;
using v2 = __m128i;
alignas(64) static u8 low_table[128];

static const u8 *low_bytes_a(size_t len) { return low_table + 64 - len; }
static v4 load(const void *p) { return _mm256_loadu_si256((const v4 *)p); }

// Copied from track-a/main.cpp, unchanged.
static u64 hash_a(const u8 *p, size_t len) {
  const u8 *cut = low_bytes_a(std::min<size_t>(len, 64));
  v4 a = _mm256_and_si256(load(p), load(cut));
  v4 b = _mm256_and_si256(load(p + 32), load(cut + 32));
  v2 seed = _mm_set_epi64x(0x2D358DCCAA6C78A5, 0x8BB84B93962EACC9);
  v2 first = _mm_aesenc_si128(_mm_xor_si128(_mm256_castsi256_si128(a), seed), _mm256_extracti128_si256(a, 1));
  v2 h = _mm_aesenc_si128(first, _mm_aesenc_si128(_mm256_castsi256_si128(b), _mm256_extracti128_si256(b, 1)));
  for (size_t i = 64; i < len; i += 16) {
    auto tail = (const v2 *)low_bytes_a(std::min<size_t>(len - i, 16));
    h = _mm_aesenc_si128(h, _mm_and_si128(_mm_loadu_si128((const v2 *)(p + i)), _mm_loadu_si128(tail)));
  }
  return _mm_cvtsi128_si64(_mm_aesenc_si128(_mm_aesenc_si128(h, seed), seed));
}

// Copied from track-c/main.cpp, unchanged.
static u64 hash_c(const u8 *p, size_t len) {
  auto mask = (const v4 *)(low_table + 64 - 32 - std::min<size_t>(len, 32) + 32 - 32 + 32);
  mask = (const v4 *)(low_table + 64 - std::min<size_t>(len, 32));
  v4 bytes = _mm256_and_si256(_mm256_loadu_si256((const v4 *)p), _mm256_loadu_si256(mask));
  v2 key = _mm_set_epi64x(0x8BB84B93962EACC9, 0x2D358DCCAA6C78A5 ^ (long long)len);
  v2 low = _mm_xor_si128(_mm256_castsi256_si128(bytes), key);
  v2 state = _mm_aesenc_si128(low, _mm256_extracti128_si256(bytes, 1));
  for (size_t i = 32; i < len; i += 16) {
    auto tail = low_table + 64 - std::min<size_t>(len - i, 16);
    state = _mm_aesenc_si128(state, _mm_and_si128(_mm_loadu_si128((const v2 *)(p + i)), _mm_loadu_si128((const v2 *)tail)));
  }
  state = _mm_aesenc_si128(state, key);
  return _mm_cvtsi128_si64(_mm_aesenc_si128(state, key)) | 1;
}

static char buffer[512];
static char track;
static u64 h_of(const std::string &s) {
  memset(buffer, 'q', sizeof buffer);
  memcpy(buffer, s.data(), s.size());
  return track == 'a' ? hash_a((const u8 *)buffer, s.size()) : hash_c((const u8 *)buffer, s.size());
}

int main(int argc, char **argv) {
  memset(low_table, 255, 64);
  track = argv[2][0];
  // Bits the tracks actually keep: A compares 40 bits, C all 63.
  const u64 keep = track == 'a' ? 0xFFFFFFFFFF000000ull : ~0ull;
  std::ifstream in(argv[1], std::ios::binary);
  std::vector<std::string> names;
  std::string line;
  while (std::getline(in, line))
    if (!line.empty())
      names.push_back(line);
  std::sort(names.begin(), names.end());
  names.erase(std::unique(names.begin(), names.end()), names.end());
  std::unordered_map<u64, std::string> seen;
  size_t real_collisions = 0;
  std::string example;
  size_t longest = 0;
  for (auto &n : names) {
    longest = std::max(longest, n.size());
    auto [it, fresh] = seen.try_emplace(h_of(n) & keep, n);
    if (!fresh && it->second != n) {
      real_collisions++;
      if (example.empty())
        example = it->second + " = " + n;
    }
  }
  printf("%s track %c: %zu distinct names, longest %zu, %zu real collisions %s\n", argv[1], track, names.size(), longest, real_collisions, example.c_str());
  // Changes along AES diagonals in the bytes that get the fewest rounds: how often do variants of real names collide?
  std::mt19937_64 rng(5);
  static const int DIAG[4][4] = {{0, 5, 10, 15}, {4, 9, 14, 3}, {8, 13, 2, 7}, {12, 1, 6, 11}};
  size_t base = track == 'a' ? 48 : 16;
  size_t tried = 0, hits = 0;
  std::string hit_example;
  for (int round = 0; round < 3000; round++) {
    std::string s = names[rng() % names.size()];
    if (s.size() < base + 12)
      s.resize(base + 16, 'a');
    int d = rng() % 4;
    std::unordered_map<u64, std::string> local;
    for (int v = 0; v < 2000; v++) {
      std::string x = s;
      for (int k = 0; k < 3; k++)
        x[base + DIAG[d][k]] = 'a' + rng() % 26;
      if (x == s)
        continue;
      tried++;
      auto [it, fresh] = local.try_emplace(h_of(x) & keep, x);
      if (!fresh && it->second != x) {
        hits++;
        if (hit_example.empty())
          hit_example = it->second + " = " + x;
      }
    }
  }
  printf("  synthetic variants on one diagonal from byte %zu: %zu tried, %zu collisions %s\n", base, tried, hits, hit_example.c_str());
}
