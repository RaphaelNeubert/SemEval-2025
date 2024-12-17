{
  description = "Python environment with flakes in zsh";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs";

  outputs =
    { self, nixpkgs }:
    {
      devShells.default = nixpkgs.lib.mkShell {
        packages = with nixpkgs.pkgs; [
          python3
          python3Packages.numpy
          python3Packages.tqdm
        ];
      };
    };
}
