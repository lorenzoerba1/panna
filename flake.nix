# TODO: reintroduce the dbg macro somehow
{
  description = "PANNA: Playground for Approximate Nearest Neighbor Algorithms";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.hl = {
    url = "github:pamburus/hl";
    inputs.nixpkgs.follows = "nixpkgs";
  };
  inputs.flake-utils = {
    url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    hl,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {inherit system;};
        hl-bin = hl.packages.${system}.default;
        # get a python build with optimizations enabled, following
        # this suggestion: https://discourse.nixos.org/t/why-is-the-nix-compiled-python-slower/18717/9
        python = pkgs.python312;
        # .override {
        #   enableOptimizations = true;
        #   reproducibleBuild = false;
        #   self = python;
        # };

        fast-hdbscan = python.pkgs.buildPythonPackage rec {
          pname = "fast_hdbscan";
          version = "0.2.2";
          src = python.pkgs.fetchPypi {
            inherit pname version;
            hash = "sha256-6/2iCMhdM4FBzGyKa5XJ8vHqg9O5mC+YZLJdc7B9nMg=";
          };
          # do not run tests
          doCheck = false;
          # specific to buildPythonPackage, see its reference
          pyproject = true;
          dependencies = with python.pkgs; [
            numba
            numpy
            scikit-learn
          ];
          build-system = with python.pkgs; [
            setuptools
            wheel
          ];
        };

        panna-python = python.pkgs.buildPythonPackage {
          pname = "panna";
          version = "0.0.1";
          pyproject = true;
          src = ./.;

          # as stated here, one should disable the cmake setup:
          # https://discourse.nixos.org/t/building-python-package-with-scikit-build-core-and-cmake-dependencies-die-python/69665/2
          dontUseCmakeConfigure = true;
          dontUseCmakeBuild = true;
          dontUseCmakeInstall = true;

          build-system = with python.pkgs; [
            scikit-build-core
            nanobind
          ];
          dependencies = with python.pkgs; [
            numpy
            h5py
          ];
          buildInputs = with pkgs; [
            python.pkgs.build
            catch2_3
            cereal
            hdf5
            highfive
          ];
          nativeBuildInputs = with pkgs; [
            cmake
            git
            ninja
          ];
        };

        container = pkgs.singularity-tools.buildImage {
          name = "panna";
          runScript = "#!${pkgs.stdenv.shell}\npython $@";
          contents = [
            (python.withPackages
              (ppkgs:
                with ppkgs; [
                  numpy
                  h5py
                  panna-python
                  fast-hdbscan
                  icecream
                ]))
          ];
          diskSize = 1024 * 3; # necessary to fit the packages, otherwise the build fails
        };
      in {
        packages.default = panna-python;
        packages.container = container;
        devShells.default = (pkgs.mkShell.override {stdenv = pkgs.clangStdenv;}) {
          venvDir = ".venv";

          packages = with pkgs;
            [
              gcc
              lldb
              clang-tools
              python
              hdf5
              sqlite-interactive
              cmake
              just
              bear # To generate compile_commands.json files
              llvmPackages.openmp
              llvmPackages.libcxx
              rr
              gdbgui
              valgrind
              highfive
              samply
              boost
              cereal
              catch2_3
              fast-hdbscan
            ]
            ++ (with python312Packages; [
              venvShellHook
              numpy
              h5py
              nanobind
              icecream
              scikit-build-core
            ])
            ++ [hl-bin];

          NIX_ENFORCE_NO_NATIVE = false;
        };
      }
    );
}
