{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs-brainflow.url = "github:Pandapip1/nixpkgs/brainflow";
    nixpkgs-optree.url = "github:Pandapip1/nixpkgs/init-pythonpackages-optree";
  };

  outputs = { self, nixpkgs, nixpkgs-brainflow, nixpkgs-optree }: {
    packages.x86_64-linux.bfivrc = with
    import nixpkgs {
      system = "x86_64-linux";
    }; let
      pkgs-brainflow = import nixpkgs-brainflow {
        system = "x86_64-linux";
        inherit nixpkgs;
      };
      pkgs-optree = import nixpkgs-optree {
        system = "x86_64-linux";
        inherit nixpkgs;
      };
    in stdenv.mkDerivation {
      name = "bfivrc";
      buildInputs = (with pkgs; [
        python311
      ]) ++ (with pkgs.python311Packages; [
        numpy
        python-osc
        scipy
        setuptools
        matplotlib
        pillow
        scikit-learn
        tensorflow
        keras
        rich
        ml-dtypes
      ]) ++ (with pkgs-brainflow.python311Packages; [
        brainflow
      ]) ++ (with pkgs-optree.python311Packages; [
        optree
      ]);
    };

    packages.x86_64-linux.default = self.packages.x86_64-linux.bfivrc;
  };
}
