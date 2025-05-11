
use std::sync::Arc;
use rustfft::{FftPlanner, num_complex::Complex, Fft};
use serde::Deserialize;

use std::convert::Infallible;
use std::net::SocketAddr;

use http_body_util::Full;
use hyper::body::{Bytes, Incoming};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;

const DEFAULT_QUERY: &str = r#"
sum(rate(node_network_receive_bytes_total{device=~"en[ops].*"}[30s]))+sum(rate(node_network_transmit_bytes_total{device=~"en[ops].*"}[30s]))
"#;

const DEFAYLT_QUERY_IN: &str = r#"
sum(rate(node_network_receive_bytes_total{device=~"en[ops].*"}[30s]))
"#;

const DEFAULT_QUERY_OUT: &str = r#"
sum(rate(node_network_transmit_bytes_total{device=~"en[ops].*"}[30s]))
"#;

fn get_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs()
}

#[derive(Debug, Clone, Deserialize)]
pub struct PrometheusResponse {
    pub status: String,
    pub data: PrometheusData,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PrometheusData {
    pub result_type: String,
    pub result: Vec<PrometheusResult>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PrometheusResult {
    pub values: Vec<(u64, String)>,
}

impl PrometheusResult {
    pub fn get_values(&self) -> Vec<f32> {
        self.values.iter().map(|(_, v)| v.parse().unwrap_or(0.0f32)).collect()
    }
}

async fn fetch_prometheus_data(url: &str, query: &str) -> anyhow::Result<PrometheusResponse> {
    let now = get_timestamp();
    let start = now - 60 * 60 * 2; // 2 hour ago
    let end = now;

    let client = reqwest::Client::new();
    let response = client
        .get(format!("{}/api/v1/query_range", url))
        .query(&[("query", query)])
        .query(&[("start", start.to_string())])
        .query(&[("end", end.to_string())])
        .query(&[("step", "15s")])
        .send()
        .await?
        .json::<PrometheusResponse>()
        .await?;

    Ok(response)
}

pub struct DataFetcher {
    url: String,
    query: String,
    fft: Arc<dyn Fft<f32>>,
}

impl DataFetcher {
    pub fn new(url: String, query: String) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(240);
        Self { url, query, fft }
    }

    pub async fn fetch(&self) -> anyhow::Result<PrometheusResponse> {
        fetch_prometheus_data(&self.url, &self.query).await
    }

    pub async fn fetch_and_process(&self) -> anyhow::Result<Vec<f32>> {
        let response = self.fetch().await?;

        for result in response.data.result {
            let values = result.get_values();
            // return last 240 samples
            let last_240_samples = values.iter().rev().take(240).rev().cloned().collect::<Vec<f32>>();

            // 1Gbps
            let min_max = 1_000_000_000.0;

            let max = last_240_samples.iter().cloned().fold(min_max, f32::max);

            let normalized_samples: Vec<f32> = last_240_samples.iter().map(|&x| x / max).collect();

            let mut buf: Vec<Complex<f32>> = normalized_samples.iter().map(|&x| Complex::new(x, 0.0)).collect();
            self.fft.process(&mut buf);
            let magnitudes: Vec<f32> = buf.iter().map(|c| c.norm()).collect();
            return Ok(magnitudes);
        }

        Err(anyhow::anyhow!("No data found"))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let prometheus_url = std::env::var("PROMETHEUS_URL")
        .unwrap_or("http://127.0.0.1:9090".to_string());

    let prometheus_query = std::env::var("PROMETHEUS_QUERY")
        .unwrap_or(DEFAULT_QUERY.trim().to_string());

    let prometheus_query_in = std::env::var("PROMETHEUS_QUERY_IN")
        .unwrap_or(DEFAYLT_QUERY_IN.trim().to_string());

    let prometheus_query_out = std::env::var("PROMETHEUS_QUERY_OUT")
        .unwrap_or(DEFAULT_QUERY_OUT.trim().to_string());

    let listen_addr: SocketAddr = std::env::var("LISTEN_ADDR")
        .unwrap_or("127.0.0.1:3333".to_string())
        .parse()
        .expect("Invalid LISTEN_ADDR");

    let listener = TcpListener::bind(listen_addr).await?;
    println!("Listening on {}", listen_addr);


    let data_fetcher = Arc::new(DataFetcher::new(prometheus_url.clone(), prometheus_query.clone()));

    let data_fetcher_in = Arc::new(DataFetcher::new(prometheus_url.clone(), prometheus_query_in.clone()));
    let data_fetcher_out = Arc::new(DataFetcher::new(prometheus_url.clone(), prometheus_query_out.clone()));

    let make_service = move |req: Request<Incoming>| {
        let data_fetcher = data_fetcher.clone();
        let data_fetcher_in = data_fetcher_in.clone();
        let data_fetcher_out = data_fetcher_out.clone();

        async move {
            let path = req.uri().path();

            if path == "/stereo" {
                let in_data = data_fetcher_in.clone().fetch_and_process().await.unwrap_or_else(|_| vec![]);
                let out_data = data_fetcher_out.clone().fetch_and_process().await.unwrap_or_else(|_| vec![]);

                let data = serde_json::json!({
                    "in": in_data,
                    "out": out_data,
                });
                let data = serde_json::to_string(&data).unwrap().into_bytes();
                let mut res = Response::new(Full::new(Bytes::from(data)));
                *res.status_mut() = hyper::StatusCode::OK;
                res.headers_mut().append(
                    "Content-Type",
                    hyper::header::HeaderValue::from_static("application/json"),
                );
                res.headers_mut().append(
                    "Access-Control-Allow-Origin",
                    hyper::header::HeaderValue::from_static("*"),
                );
                return Ok::<_, Infallible>(res);
            }

            let data = data_fetcher.clone().fetch_and_process().await.unwrap_or_else(|_| vec![]);
            println!("Processed data: {:?}", data);
            let data = serde_json::to_string(&data).unwrap().into_bytes();

            let mut res = Response::new(Full::new(Bytes::from(data)));

            *res.status_mut() = hyper::StatusCode::OK;
            res.headers_mut().append(
                "Content-Type",
                hyper::header::HeaderValue::from_static("application/json"),
            );
            res.headers_mut().append(
                "Access-Control-Allow-Origin",
                hyper::header::HeaderValue::from_static("*"),
            );
            Ok::<_, Infallible>(res)
        }
    };

    loop {
        let (socket, _) = listener.accept().await?;
        let io = TokioIo::new(socket);

        let make_service_clone = make_service.clone();
        let http1 = http1::Builder::new()
            .serve_connection(io, service_fn(make_service_clone));
        tokio::spawn(http1);
    }

}
