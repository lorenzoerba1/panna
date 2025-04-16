install-python-extension:
    pip install --no-build-isolation -ve .

example: build-example
    build/fashion

profile-example: build-example
    samply record build/fashion

build-example:
    cmake --build build -j --target fashion

test: build-tests
    build/tests "angular distance trie index"

build-tests:
    cmake --build build -j --target tests

clean:
    cd build && make clean

setup-cmake:
    test -d build || mkdir build
    cd build && cmake -DCMAKE_BUILD_TYPE=Debug ..

generate-compile-commands:
    just clean
    bear -- cmake --build build -j

scan-build:
    just clean
    scan-build just build
