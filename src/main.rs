// #![windows_subsystem = "windows"] // uncomment this to suppress terminal on windows

fn main() -> Result<(), winit::error::EventLoopError> {
    let event_loop = winit::event_loop::EventLoop::builder().build()?;

    #[cfg(not(target_arch = "wasm32"))]
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    #[cfg(target_arch = "wasm32")]
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);

    let mut app = app_core::App::default();
    event_loop.run_app(&mut app)?;
    Ok(())
}
