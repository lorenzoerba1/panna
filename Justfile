example: build-example
    build/glove

build-example:
    cmake --build build -j --target glove

test: build-tests
    build/tests 

build-tests:
    cmake --build build -j --target tests

clean:
    cd build && make clean

setup-cmake:
    test -d build || mkdir build
    cd build && cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..

generate-compile-commands:
    just clean
    bear -- just build-tests

scan-build:
    just clean
    scan-build just build
