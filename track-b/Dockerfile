FROM python:3.13-slim@sha256:d98cc29d3266e535dd9da1099688d9cb3ca4753813571e071763d17725460962 AS generator
COPY b_index.py /b_index.py
RUN python3 /b_index.py --seed 7 --scale sample --out /data

FROM gcc:16@sha256:43978d7710a803d1c3d32f4117c1528ec7285a4ce2c0a51e982ec1a6e3d87d95 AS build
WORKDIR /src
COPY main.cpp .
COPY --from=generator /data /data
ENV FLAGS="-std=c++20 -O3 -flto -march=x86-64-v3 -maes -mtune=alderlake -fno-exceptions -static -pthread -o solution main.cpp"
RUN mkdir /out && g++ $FLAGS -fprofile-generate -fprofile-update=prefer-atomic -D_exit=exit && ./solution && g++ $FLAGS -fprofile-use -fprofile-correction -Wno-missing-profile

FROM debian:13-slim@sha256:62bd423aeee912639ca8ee681c2b762cb3e25137e0a92b20be455898a9f034ad
COPY --from=build /src/solution /app/solution
ENTRYPOINT ["/app/solution"]
