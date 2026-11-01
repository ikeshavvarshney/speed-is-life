FROM python:3.13-slim@sha256:d98cc29d3266e535dd9da1099688d9cb3ca4753813571e071763d17725460962 AS data
COPY c_join.py .
RUN python3 c_join.py --seed 2 --scale sample --out /data
FROM gcc:16@sha256:43978d7710a803d1c3d32f4117c1528ec7285a4ce2c0a51e982ec1a6e3d87d95 AS build
COPY main.cpp .
COPY --from=data /data /data
RUN F="-std=c++20 -O3 -falign-loops=64 -falign-functions=64 -flto -march=x86-64-v3 -mtune=alderlake -mpclmul -maes -mprfchw -fno-exceptions -s -Wl,--gc-sections -ffunction-sections -fdata-sections -static -o solution main.cpp"; mkdir /out && g++ $F -fprofile-generate -fprofile-update=atomic -D_exit=exit && ./solution && g++ $F -fprofile-use -fprofile-correction -Wno-missing-profile

FROM debian:13-slim@sha256:62bd423aeee912639ca8ee681c2b762cb3e25137e0a92b20be455898a9f034ad
COPY --from=build /solution /app/solution
ENTRYPOINT ["/app/solution"]
