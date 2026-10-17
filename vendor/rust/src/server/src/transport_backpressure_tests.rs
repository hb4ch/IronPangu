//! Real TCP backpressure through the production HTTP connection loop.
//! Synthetic body source: this does not qualify LLM scheduling or NPU output.
use crate::listener::{Listener, MaybeTlsListener};
use crate::{ConnectionTimeouts, HttpListenerMode, serve_connections};
use axum::{Router, body::Body, routing::get};
use bytes::Bytes;
use std::convert::Infallible;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio_util::sync::CancellationToken;

struct Lifetime(Arc<AtomicBool>);
impl Drop for Lifetime {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

#[tokio::test]
async fn stalled_tcp_consumer_bounds_polling_keeps_peer_live_and_drops_stream() {
    let pulled = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(AtomicBool::new(false));
    let source_count = pulled.clone();
    let source_drop = dropped.clone();
    let app = Router::new().route("/health", get(|| async { "ok" })).route(
        "/stream",
        get(move || {
            let count = source_count.clone();
            let lifetime = Lifetime(source_drop.clone());
            async move {
                Body::from_stream(futures::stream::unfold(lifetime, move |lifetime| {
                    count.fetch_add(1, Ordering::AcqRel);
                    async move {
                        static CHUNK: [u8; 65536] = [b'x'; 65536];
                        Some((Ok::<_, Infallible>(Bytes::from_static(&CHUNK)), lifetime))
                    }
                }))
            }
        }),
    );
    let listener = Listener::bind(&HttpListenerMode::BindTcp {
        host: "127.0.0.1".into(),
        port: 0,
    })
    .await
    .unwrap();
    let addr = listener.local_addr_display().unwrap();
    let shutdown = CancellationToken::new();
    let stopped = shutdown.clone();
    let server = tokio::spawn(async move {
        serve_connections(
            MaybeTlsListener::plain(listener),
            app,
            stopped.cancelled_owned(),
            ConnectionTimeouts {
                header_read: Duration::from_secs(5),
                keep_alive_enabled: true,
            },
        )
        .await
    });
    let mut slow = TcpStream::connect(&addr).await.unwrap();
    slow.write_all(b"GET /stream HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .await
        .unwrap();
    // Read one byte to establish that the streaming response has begun, then stop.
    tokio::time::timeout(Duration::from_secs(3), slow.read_exact(&mut [0u8; 1]))
        .await
        .unwrap()
        .unwrap();
    tokio::time::timeout(Duration::from_secs(3), async {
        let mut last = pulled.load(Ordering::Acquire);
        let mut stable = 0;
        while stable < 5 {
            tokio::time::sleep(Duration::from_millis(20)).await;
            let now = pulled.load(Ordering::Acquire);
            assert!(
                now < 1024,
                "unread connection consumed over 64 MiB of generated body"
            );
            stable = if now == last { stable + 1 } else { 0 };
            last = now;
        }
    })
    .await
    .expect("output polling must stop while the TCP client is stalled");
    assert!(pulled.load(Ordering::Acquire) > 0);
    assert!(!dropped.load(Ordering::Acquire));
    let mut peer = TcpStream::connect(&addr).await.unwrap();
    peer.write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .await
        .unwrap();
    let mut response = String::new();
    tokio::time::timeout(Duration::from_secs(3), peer.read_to_string(&mut response))
        .await
        .unwrap()
        .unwrap();
    assert!(response.starts_with("HTTP/1.1 200"));
    let before_resume = pulled.load(Ordering::Acquire);
    tokio::time::timeout(Duration::from_secs(3), async {
        let mut buffer = [0u8; 65536];
        while pulled.load(Ordering::Acquire) == before_resume {
            assert!(
                slow.read(&mut buffer).await.unwrap() > 0,
                "stream ended while resuming"
            );
        }
    })
    .await
    .expect("body polling must resume when the consumer drains its socket");
    drop(slow);
    tokio::time::timeout(Duration::from_secs(3), async {
        while !dropped.load(Ordering::Acquire) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("disconnect must drop the body source");
    shutdown.cancel();
    tokio::time::timeout(Duration::from_secs(3), server)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}
