{
  description = "PUFFINN - Parameterless and Universal Fast FInding of Nearest Neighbors";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.hl.url = "github:pamburus/hl";

  outputs = {
    self,
    nixpkgs,
    hl,
  }: let
    supportedSystems = ["x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin"];
    forEachSupportedSystem = f:
      nixpkgs.lib.genAttrs supportedSystems (system:
        f {
          pkgs = import nixpkgs {inherit system;};
          hlbin = hl.packages.${system}.default;
        });
  in {
    devShells = forEachSupportedSystem ({
      pkgs,
      hlbin,
    }: {
      default = (pkgs.mkShell.override {stdenv = pkgs.clangStdenv;}) {
        venvDir = ".venv";

        packages = with pkgs;
          [
            gcc
            lldb
            clang-tools
            python312
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
          ]
          ++ (with python312Packages; [
            venvShellHook
            numpy
            h5py
            nanobind
            icecream
            scikit-build-core
          ])
          ++ [hlbin];

        NIX_ENFORCE_NO_NATIVE = false;
      };
    });
  };
}
