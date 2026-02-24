{
  description = "jamtrack-rs development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f system);
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        {
          default = pkgs.mkShell {
            name = "jamtrack";
            packages = with pkgs; [
              rustc
              cargo
              ffmpeg_7-full
              pkg-config
              python3
              # For visualizer GUI
              clang
              llvmPackages.libclang
              libxkbcommon
              wayland
              xorg.libX11
              xorg.libXcursor
              xorg.libXrandr
              xorg.libXi
              vulkan-loader
              libGL
            ];
            shellHook = ''
              export PS1="(jamtrack) $PS1"
              export LIBCLANG_PATH="${pkgs.llvmPackages.libclang.lib}/lib"
              export PKG_CONFIG_PATH="${pkgs.ffmpeg_7-full.dev}/lib/pkgconfig:$PKG_CONFIG_PATH"
              export BINDGEN_EXTRA_CLANG_ARGS="-I${pkgs.ffmpeg_7-full.dev}/include"
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
                pkgs.libxkbcommon
                pkgs.wayland
                pkgs.xorg.libX11
                pkgs.xorg.libXcursor
                pkgs.xorg.libXrandr
                pkgs.xorg.libXi
                pkgs.vulkan-loader
                pkgs.libGL
                pkgs.ffmpeg_7-full.lib
              ]}:$LD_LIBRARY_PATH"
            '';
          };
        }
      );
    };
}
