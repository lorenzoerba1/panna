example: build-example
    build/fashion

profile-example: build-example
    samply record build/fashion

build-example:
    cmake --build build -j --target fashion

test: build-tests
    build/tests "serialization"

build-tests:
    cmake --build build -j --target tests

clean:
    cd build && make clean

setup-cmake:
    test -d build || mkdir build
    cd build && cmake -DCMAKE_BUILD_TYPE=Debug ..

generate-compile-commands:
    just clean
    bear -- just build-tests

scan-build:
    just clean
    scan-build just build
