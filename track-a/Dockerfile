FROM gcc:16@sha256:43978d7710a803d1c3d32f4117c1528ec7285a4ce2c0a51e982ec1a6e3d87d95 AS build
WORKDIR /src
COPY main.cpp a_ingest.py .
RUN F="-std=c++20 -O3 -flto -march=x86-64-v3 -mtune=alderlake -maes -fno-exceptions -static -o solution main.cpp"; mkdir /data /out && python3 a_ingest.py --seed 2 --scale sample --out /data && g++ $F -fprofile-generate -fprofile-update=prefer-atomic -D_exit=exit && ./solution && g++ $F -fprofile-use -fprofile-correction -Wno-missing-profile
FROM debian:13-slim@sha256:62bd423aeee912639ca8ee681c2b762cb3e25137e0a92b20be455898a9f034ad
COPY --from=build /src/solution /app/solution
ENTRYPOINT ["/app/solution"]
