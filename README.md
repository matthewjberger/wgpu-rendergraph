# Rust / Winit / Egui / Wgpu Rendergraph

This project demonstrates how to setup an app using [rust](https://www.rust-lang.org/) and [wgpu](https://wgpu.rs/) to render a scene using a rendergraph, supporting
both webgl and webgpu [wasm](https://webassembly.org/) as well as native.

<img width="2560" height="1392" alt="rendergraph" src="https://github.com/user-attachments/assets/2724809f-41b1-4657-89e8-a852fa365d06" />

## Quickstart

```
# native
cargo run -r

# webgpu
trunk serve --features webgpu --open

# webgl
trunk serve --features webgl --open
```

> All chromium-based browsers like Brave, Vivaldi, Chrome, etc support wgpu.
> Firefox also [supports wgpu](https://mozillagfx.wordpress.com/2025/07/15/shipping-webgpu-on-windows-in-firefox-141/) now starting with version `141`.

## Prerequisites (web)

* [trunk](https://trunkrs.dev/)
