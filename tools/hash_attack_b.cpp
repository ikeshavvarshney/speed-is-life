// Build: g++ -std=c++20 -O2 -march=x86-64-v3 -maes. Run: hash_attack_b <docs.txt> ...
// Attacks candidate token hashes with structured differences, the way AES is weak when truncated.
// usage: hashtest2 <docs.txt> ...
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using u8 = uint8_t;
using u64 = unsigned long long;
using v2 = __m128i;
alignas(64) static u8 low_bytes[64];
static const u64 IDENTITY = ~(1ull << 19);
static int variant = 0;

static inline u64 mix_rounds(v2 state) {
  state = _mm_aesenc_si128(state, _mm_set_epi64x(0x2D358DCCAA6C78A5, 0x8BB84B93));
  state = _mm_aesenc_si128(state, _mm_set_epi64x(0x4B33A62ED433D4A3, 0xC2B2AE3D));
  return _mm_cvtsi128_si64(
      _mm_aesenc_si128(state, _mm_set_epi64x(0x9E3779B97F4A7C15, 0xBF58476D)));
}

// Up to 32 bytes are all covered by the first and last 16, and longer tokens differ
// from any term by their length.
static u64 long_token_hash(const u8 *start, size_t len) {
  v2 head = _mm_loadu_si128((const v2 *)start);
  v2 tail = _mm_loadu_si128((const v2 *)(start + len - 16));
  v2 lower = _mm_set1_epi8(0x20);
  head = _mm_aesenc_si128(_mm_or_si128(head, lower), _mm_or_si128(tail, lower));
  return mix_rounds(_mm_xor_si128(head, _mm_set1_epi64x(len)));
}

// Hashes a token where it lies. Token bytes are letters and digits, so OR-ing 0x20
// lowercases them, and masking removes whatever follows. AES spreads every byte in far
// fewer micro-ops than a wide multiply, but it takes three rounds. After two, half of
// the state still collides for tokens differing only along one of AES's diagonals:
// `wodeand` and `woweandvi` both occur in the seed 2 corpus, and had the same hash. The
// hash stands in for the bytes: with about 600,000 distinct tokens and 76,000 terms,
// the chance of any collision is about 1 in 100 million.
static inline u64 token_hash(const u8 *start, size_t len) {
  if (len > 16)
    return long_token_hash(start, len);
  v2 bytes = _mm_or_si128(_mm_loadu_si128((const v2 *)start), _mm_set1_epi8(0x20));
  return mix_rounds(
      _mm_and_si128(bytes, _mm_loadu_si128((const v2 *)(low_bytes + 32 - len))));
}

static char buffer[256];
static u64 h_of(const std::string &s) {
  memset(buffer, 'q', sizeof buffer);
  memcpy(buffer, s.data(), s.size());
  return token_hash((const u8 *)buffer, s.size()) & IDENTITY;
}

static std::unordered_map<u64, std::string> seen_hash;
static size_t collisions, total;
static std::string example;
static void add(const std::string &s) {
  u64 h = h_of(s);
  total++;
  auto [it, fresh] = seen_hash.try_emplace(h, s);
  if (!fresh && it->second != s) {
    collisions++;
    if (example.empty())
      example = it->second + " = " + s;
  } else if (!fresh) {
    total--;
  }
}
static void report(const char *name) {
  printf("  %-44s %10zu distinct, %6zu collisions %s\n", name, total, collisions, example.c_str());
  seen_hash.clear();
  collisions = total = 0;
  example.clear();
}

int main(int argc, char **argv) {
  memset(low_bytes, 255, 32);
  static const char ALPHA[] = "abcdefghijklmnopqrstuvwxyz0123456789";
  // The state diagonals AES mixes in its first round.
  static const int DIAG[4][4] = {{0, 5, 10, 15}, {4, 9, 14, 3}, {8, 13, 2, 7}, {12, 1, 6, 11}};
  const char *names[] = {"the code to push"};
  for (variant = 0; variant < 1; variant++) {
    printf("%s\n", names[variant]);
    std::mt19937_64 rng(7);
    // Differences on one diagonal, 3 and 4 positions.
    for (int d = 0; d < 4; d++) {
      for (int base = 0; base < 25; base++) {
        std::string s(16, 'a');
        for (auto &c : s)
          c = ALPHA[rng() % 36];
        for (int a = 0; a < 36; a++)
          for (int b = 0; b < 36; b++)
            for (int c = 0; c < 36; c += 2)
              for (int e = 0; e < 36; e += 18) {
                std::string x = s;
                x[DIAG[d][0]] = ALPHA[a];
                x[DIAG[d][1]] = ALPHA[b];
                x[DIAG[d][2]] = ALPHA[c];
                x[DIAG[d][3]] = ALPHA[e];
                add(x);
              }
      }
    }
    report("same diagonal, 16 chars");
    // Shorter: only the diagonal positions below the length vary.
    for (int len = 9; len <= 15; len++) {
      for (int d = 0; d < 4; d++) {
        int pos[4], n = 0;
        for (int i = 0; i < 4; i++)
          if (DIAG[d][i] < len)
            pos[n++] = DIAG[d][i];
        if (n < 2)
          continue;
        for (int base = 0; base < 40; base++) {
          std::string s(len, 'a');
          for (auto &c : s)
            c = 'a' + rng() % 26;
          for (int a = 0; a < 26; a++)
            for (int b = 0; b < 26; b++)
              for (int c = 0; c < (n > 2 ? 26 : 1); c++) {
                std::string x = s;
                x[pos[0]] = 'a' + a;
                x[pos[1]] = 'a' + b;
                if (n > 2)
                  x[pos[2]] = 'a' + c;
                add(x);
              }
        }
      }
    }
    report("same diagonal, 9 to 15 letters");
    // All pairs of changed positions, to the longest token the spec allows.
    for (int len = 5; len <= 32; len++) {
      for (int base = 0; base < 6; base++) {
        std::string s(len, 'a');
        for (auto &c : s)
          c = ALPHA[rng() % 36];
        for (int i = 0; i < len; i++)
          for (int j = i + 1; j < len; j++)
            for (int a = 0; a < 36; a += 5)
              for (int b = 0; b < 36; b += 5) {
                std::string x = s;
                x[i] = ALPHA[a];
                x[j] = ALPHA[b];
                add(x);
              }
      }
    }
    report("two changes, 5 to 32 chars");
    // Numbers and years: digits, zero padding, at many lengths.
    for (int n = 0; n < 2000000; n++) {
      char t[32];
      add(std::string(t, snprintf(t, sizeof t, "%d", n)));
      if (n < 400000) {
        add(std::string(t, snprintf(t, sizeof t, "%07d", n)));
        add(std::string(t, snprintf(t, sizeof t, "%011d", n)));
      }
    }
    report("numbers, zero padded too");
    // Real tokens.
    for (int a = 1; a < argc; a++) {
      std::ifstream in(argv[a], std::ios::binary);
      std::string line;
      std::unordered_set<std::string> tokens;
      while (std::getline(in, line)) {
        size_t tab = line.find('\t');
        std::string cur;
        for (size_t i = tab + 1; i <= line.size(); i++) {
          unsigned char c = i < line.size() ? line[i] : ' ';
          if (c >= 'A' && c <= 'Z')
            c += 32;
          if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
            cur.push_back(c);
          } else if (!cur.empty()) {
            tokens.insert(cur);
            cur.clear();
          }
        }
      }
      for (auto &x : tokens)
        add(x);
      report(argv[a]);
    }
  }
}
