FROM swift:6.2@sha256:d6f8f6112e9a63da0a4a151d6b93acd16699d35c0b82531ac1a245f5af210197 AS build
COPY main.cpp .
RUN clang++ -std=c++20 -O3 -flto -fuse-ld=lld -fno-pic -march=x86-64-v3 -mtune=alderlake -maes -fno-exceptions -static -o solution main.cpp
FROM debian:13-slim@sha256:62bd423aeee912639ca8ee681c2b762cb3e25137e0a92b20be455898a9f034ad
COPY --from=build /solution /app/solution
ENTRYPOINT ["/app/solution"]
