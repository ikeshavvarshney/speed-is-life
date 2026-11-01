FROM python:3.13-slim@sha256:d98cc29d3266e535dd9da1099688d9cb3ca4753813571e071763d17725460962 AS data
COPY c_join.py .
RUN python3 c_join.py --seed 2 --scale full --out /data
FROM swift:6.2@sha256:d6f8f6112e9a63da0a4a151d6b93acd16699d35c0b82531ac1a245f5af210197 AS build
COPY main.cpp .
COPY --from=data /data /data
RUN F="-std=c++20 -O3 -flto -fuse-ld=lld -fno-pic -march=x86-64-v3 -mtune=alderlake -maes -mpclmul -mprfchw -fno-exceptions -static -pthread -o solution main.cpp"; mkdir /out && clang++ $F -fprofile-generate -fprofile-update=atomic -D_exit=exit && ./solution && llvm-profdata merge -o p.profdata default_*.profraw && clang++ $F -fprofile-use=p.profdata
FROM debian:13-slim@sha256:62bd423aeee912639ca8ee681c2b762cb3e25137e0a92b20be455898a9f034ad
COPY --from=build /solution /app/solution
ENTRYPOINT ["/app/solution"]
