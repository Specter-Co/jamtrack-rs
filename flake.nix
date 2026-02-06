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
              ffmpeg
              python3
            ];
            shellHook = ''
              export PS1="(jamtrack) $PS1"
            '';
          };
        }
      );
    };
}
