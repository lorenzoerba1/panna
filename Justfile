test: build-tests
    build/tests "CrossPolytope*"

build-tests:
    cmake --build build -j --config Debug --target tests

clean:
    cd build && make clean

setup-cmake:
    test -d build || mkdir build
    cd build && cmake ..

generate-compile-commands:
    just clean
    bear -- just build-tests

scan-build:
    just clean
    scan-build just build
