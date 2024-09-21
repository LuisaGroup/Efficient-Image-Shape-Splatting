# Efficient Image-Space Shape Splatting for Monte Carlo Rendering
Code for SIGGRAPH Asia 2024 Paper ["Efficient Image-Space Shape Splatting for Monte Carlo Rendering"](https://cs.uwaterloo.ca/~xtong/assets/pdf/shape_splatting.pdf).

The code is based on [AkariRender](https://github.com/shiinamiyuki/akari_render). The two main files are:
-  [shape_splat_pt.rs](crates/akari_integrator/src/shape_splat_pt.rs) contains the implementation of the shape splatting integrator on top of path tracing. This integrator is designed to run on GPU and uses the wavefront scheduling proposed in the paper. Running the integrator on CPU is not recommended.
- [shape_splat_mcmc.rs](crates/akari_integrator/src/shape_splat_mcmc.rs) contains the implementation of the shape splatting integrator on top of a unidirectional PSSMLT. It should be run on CPU only.

## Build
## Building:
If you are using < Windows 10, please upgrade to Windows 10 or above.
- Rust 1.81.0+
- CMake > 3.23
- Ninja
- Clone Blender 4.0 source code from `blender-v4.0-release` branch
- Put path to blender source in `blender_src_path.txt` at project root

To run on CPU, the following runtime requirement must be satisfied:
- clang++ in `PATH`
- llvm dynamic library of the same version (for Windows users, it is the `LLVM-C.dll`.
) should be in `PATH` as well.

## Run
```bash
# Run on GPU
cargo run --release --bin akari-cli -- -s scenes/fireplace-room/scene.json  -m config/pt.json -d cuda
cargo run --release --bin akari-cli -- -s scenes/fireplace-room/scene.json  -m config/shape-splatting-pt.json -d cuda

# Run on CPU
cargo run --release --bin akari-cli -- -s scenes/fireplace-room/scene.json  -m config/pssmlt.json
cargo run --release --bin akari-cli -- -s scenes/fireplace-room/scene.json  -m config/shape-splatting-pssmlt.json
```

