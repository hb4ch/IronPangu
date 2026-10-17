//! Consume termination synchronously, before native libraries install handlers.
//! No runtime cleanup runs in an async signal handler or chains into CANN's handler.
use tokio_util::sync::CancellationToken;

#[cfg(unix)]
pub fn install() -> anyhow::Result<CancellationToken> {
    let stop = CancellationToken::new();
    // Called by main before creating any threads. Every worker inherits the mask.
    let mut signals = unsafe { std::mem::zeroed::<libc::sigset_t>() };
    unsafe {
        libc::sigemptyset(&mut signals);
        libc::sigaddset(&mut signals, libc::SIGINT);
        libc::sigaddset(&mut signals, libc::SIGTERM);
        let error = libc::pthread_sigmask(libc::SIG_BLOCK, &signals, std::ptr::null_mut());
        if error != 0 { return Err(std::io::Error::from_raw_os_error(error).into()); }
    }
    let signal_stop = stop.clone();
    std::thread::Builder::new().name("shutdown-signals".into()).spawn(move || {
        let mut received = 0;
        let error = unsafe { libc::sigwait(&signals, &mut received) };
        if error != 0 { eprintln!("signal wait failed: {}", std::io::Error::from_raw_os_error(error)); }
        else { eprintln!("shutdown signal {received} received; draining frontend"); }
        signal_stop.cancel();
    })?;
    Ok(stop)
}

#[cfg(not(unix))]
pub fn install() -> anyhow::Result<CancellationToken> {
    let stop = CancellationToken::new();
    let signal_stop = stop.clone();
    std::thread::Builder::new().name("shutdown-signals".into()).spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().expect("signal runtime");
        runtime.block_on(async { let _ = tokio::signal::ctrl_c().await; });
        signal_stop.cancel();
    })?;
    Ok(stop)
}
